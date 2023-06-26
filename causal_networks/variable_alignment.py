from functools import partial
from typing import Callable, Iterable, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import IterableDataset, DataLoader, TensorDataset

import numpy as np

from tqdm import tqdm

from transformer_lens.hook_points import HookedRootModule, HookPoint

from .dag import DeterministicDAG


class ParametrisedOrthogonalMatrix(nn.Module):
    """A parametrised orthogonal matrix"""

    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.theta = nn.Parameter(torch.randn(size, size))

    def get_orthogonal_matrix(self) -> torch.Tensor:
        Q, R = torch.linalg.qr(self.theta)
        return Q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.get_orthogonal_matrix()

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.get_orthogonal_matrix().T

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.theta = self.theta.to(*args, **kwargs)
        return self


class InterchangeInterventionDataset(IterableDataset):
    """A dataset for doing interchange intervention

    At each step it selects a random subset of the DAG nodes and random base
    and source inputs. It always returns the same number of source inputs,
    filling in for missing nodes with the base input.

    Parameters
    ----------
    dag_nodes : list[str]
        The names of the nodes (variables) of the DAG to be aligned
    max_steps : int
        The maximum number of steps to run the training for
    """

    def __init__(self, inputs: torch.Tensor, dag_nodes: list[str], num_steps: int):
        super().__init__()
        self.inputs = inputs
        self.dag_nodes = dag_nodes
        self.num_steps = num_steps
        self.num_nodes = len(dag_nodes)

    def __iter__(self):
        for _ in range(self.num_steps):
            node_selected_mask = torch.randint(
                0, 2, (len(self.dag_nodes),), dtype=torch.bool
            )
            nodes = [
                node
                for node, selected in zip(self.dag_nodes, node_selected_mask)
                if selected
            ]
            inputs_perm = torch.randperm(self.inputs.shape[0])
            base_input = self.inputs[inputs_perm[0]]
            source_inputs = self.inputs[inputs_perm[1 : self.num_nodes + 1]]
            source_inputs[node_selected_mask] = base_input
            yield nodes, base_input, source_inputs


class VariableAlignment:
    """A variable alignment between a DAG and a hooked model

    Parameters
    ----------
    dag : DeterministicDAG
        The DAG to align to the model
    low_level_model : HookedRootModule
        The hooked low-level model to align to the DAG
    dag_nodes : list[str]
        The names of the nodes (variables) of the DAG to align
    input_alignment : callable
        A function mapping (batches of) model input tensors to DAG input
        dictionaries
    output_alignment : callable
        A function mapping (batches of) model output tensors to DAG output
        values
    intervene_model_hooks : list[str]
        The names of the model hooks together give the whole activation space
        for alignment the DAG nodes
    subspaces_sizes : list[int]
        The sizes of the subspaces to use for each DAG nodes
    device : Union[str, torch.device], default=None
        The device to put everything on. If None, use the device of the model
    verbosity : int, default=1
        The verbosity level
    progress_bar : bool, default=True
        Whether to show a progress bar during training
    """

    def __init__(
        self,
        dag: DeterministicDAG,
        low_level_model: HookedRootModule,
        dag_nodes: list[str],
        input_alignment: callable,
        output_alignment: callable,
        intervene_model_hooks: str,
        subspaces_sizes: list[int],
        device: Optional[Union[str, torch.device]] = None,
        verbosity: int = 1,
        progress_bar: bool = True,
    ):
        if len(dag_nodes) != len(subspaces_sizes):
            raise ValueError(
                f"Expected {len(dag_nodes)} subspaces sizes, got {len(subspaces_sizes)}"
            )

        if device is None:
            device = low_level_model.device

        self.dag = dag
        self.low_level_model = low_level_model
        self.dag_nodes = dag_nodes
        self.input_alignment = input_alignment
        self.output_alignment = output_alignment
        self.intervene_model_hooks = intervene_model_hooks
        self.subspaces_sizes = subspaces_sizes
        self.device = device
        self.verbosity = verbosity
        self.progress_bar = progress_bar

        self.layer_sizes = self._determine_layer_sizes()
        self.total_space_size = sum(self.layer_sizes)

        self.subspaces_sizes_with_y0 = [
            self.total_space_size - sum(self.subspaces_sizes)
        ] + self.subspaces_sizes

        if sum(self.subspaces_sizes) > self.total_space_size:
            raise ValueError(
                f"Sum of subspace sizes ({sum(self.subspaces_sizes)}) "
                f"exceeds activation space size ({self.total_space_size})"
            )

        self.rotation = ParametrisedOrthogonalMatrix(self.total_space_size)
        self.rotation.to(self.device)

    def _determine_layer_sizes(self) -> list[int]:
        """Run the model to determine the sizes of the activation spaces"""

        if self.verbosity > 0:
            print("Running model to determine activation space size...")

        sizes = []

        def counter_hook(value, hook: HookPoint, sizes):
            sizes.append(value.shape[1])

        partial_counter_hook = partial(counter_hook, sizes=sizes)

        x = torch.zeros(1, self.low_level_model.cfg.input_size, device=self.device)
        fwd_hooks = [
            (name, partial_counter_hook) for name in self.intervene_model_hooks
        ]

        self.low_level_model.run_with_hooks(x, fwd_hooks=fwd_hooks)

        return sizes

    def get_distributed_interchange_intervention_hooks(
        self,
        base_input: torch.Tensor,
        source_inputs: torch.Tensor,
    ) -> list[tuple[str, callable]]:
        """Get hooks for distributed interchange intervention

        Runs the model on the base input and each source input, and on input
        `i` records the value of the projection of the rotated activation
        space onto the `i`th subspace. It then builds hooks to patch in these
        values into the rotated activation space.

        Parameters
        ----------
        base_input : torch.Tensor of shape (input_size,) or (1, input_size)
            The base input to the model
        source_inputs : torch.Tensor of shape (num_dag_nodes, input_size)
            The source inputs to the model

        Returns
        -------
        list[tuple[str, callable]]
            The hooks to use for distributed interchange intervention, for
            each of `self.intervene_model_hooks`
        """

        if source_inputs.shape[0] != len(self.dag_nodes):
            raise ValueError(
                f"Expected {len(self.dag_nodes)} source inputs, got {source_inputs.shape[0]}"
            )

        if base_input.ndim == 1:
            base_input = base_input.unsqueeze(0)

        base_input = base_input.to(self.device)
        source_inputs = source_inputs.to(self.device)

        # Combine the base input and the source inputs
        combined_inputs = torch.cat((base_input, source_inputs), dim=0)

        def store_activation_value(
            value, hook: HookPoint, activation_values, space_index
        ):
            activation_values[:, space_index : space_index + value.shape[1]] = value

        activation_values = torch.empty(
            len(combined_inputs), self.total_space_size, device=self.device
        )

        # Hooks to store the activation values.
        fwd_hooks = []
        for i, name in enumerate(self.intervene_model_hooks):
            fwd_hooks.append(
                (
                    name,
                    partial(
                        store_activation_value,
                        activation_values=activation_values,
                        space_index=sum(self.layer_sizes[:i]),
                    ),
                )
            )
        # TODO: The last hook could raise an exception which gets caught here,
        # to avoid uselessly running the rest of the model

        if self.verbosity > 1:
            print("Running model to determine activation values on source inputs...")

        with torch.no_grad():
            self.low_level_model.run_with_hooks(combined_inputs, fwd_hooks=fwd_hooks)

        rotated_activation_values = self.rotation(activation_values)

        # Select the rotated activation values which are relevant, for patching in
        indices_list = []
        for i, subspace_size in enumerate(self.subspaces_sizes_with_y0):
            indices_list.append(torch.ones(subspace_size, device=self.device) * i)
        selection_indices = torch.cat(indices_list).long()
        new_rotated_activation_values = rotated_activation_values[
            selection_indices, torch.arange(self.total_space_size)
        ]

        # Unrotate the new rotated activation values, to get a vector
        # consisting of all the activation values to be patched in
        new_activation_values = self.rotation.inverse(new_rotated_activation_values)

        def intervention_hook(
            value,
            hook: HookPoint,
            space_index,
            new_activation_values,
        ):
            return new_activation_values[
                ..., space_index : space_index + value.shape[-1]
            ]

        fwd_hooks = []
        for i, name in enumerate(self.intervene_model_hooks):
            fwd_hooks.append(
                (
                    name,
                    partial(
                        intervention_hook,
                        space_index=sum(self.layer_sizes[:i]),
                        new_activation_values=new_activation_values,
                    ),
                )
            )

        return fwd_hooks

    def run_distributed_interchange_intervention(
        self, base_input: torch.Tensor, source_inputs: torch.Tensor
    ):
        """Compute the output of the low-level model on base and source inputs

        Does distributed interchange intervention, patching in the activation
        values from the source inputs, then running on the base input

        **Warning**: This method will disable gradients for the model parameters,
        setting `requires_grad` to `False`.

        Parameters
        ----------
        base_input : torch.Tensor of shape (input_size,)
            The base input to the model
        source_inputs : torch.Tensor of shape (num_dag_nodes, input_size)
            The source inputs to the model
        """

        base_input = base_input.to(self.device)
        source_inputs = source_inputs.to(self.device)

        # Get the patches to apply
        fwd_hooks = self.get_distributed_interchange_intervention_hooks(
            base_input, source_inputs
        )

        # Run the model with the patches applied, disabling gradients for the
        # model parameters, but not the rotation matrix used in the hooks
        self.low_level_model.requires_grad_(False)
        output = self.low_level_model.run_with_hooks(base_input, fwd_hooks=fwd_hooks)

        return output

    def run_interchange_intervention(
        self,
        base_input: torch.Tensor,
        source_inputs: torch.Tensor,
        output_type: str = "output_integer",
    ) -> torch.Tensor:
        """Run interchange intervention on the DAG

        Parameters
        ----------
        base_input : torch.Tensor of shape (input_size,)
            The base input to the model
        source_inputs : torch.Tensor of shape (num_dag_nodes, input_size)
            The source inputs to the model
        output_type : str, optional
            The type of output to return, by default "output_integer"

        Returns
        -------
        dag_output : torch.Tensor of shape (num_output_nodes,)
            The output of the dag, as integers ranging over the possible
            output values
        """

        # Convert the inputs into things the DAG can handle
        base_input_dag = self.input_alignment(base_input)
        source_inputs_dag = self.input_alignment(source_inputs)
        dag_node_singletons = [[dag_node] for dag_node in self.dag_nodes]

        # Run the intervention to create a patched model
        self.dag.do_interchange_intervention(dag_node_singletons, source_inputs_dag)

        # Run the patched model on the base input
        output = self.dag.run(
            base_input_dag,
            reset=False,
            output_type=output_type,
            output_format="torch",
        )

        return output

    def create_interchange_intervention_dataset(
        self, inputs: torch.Tensor, num_samples=10000
    ) -> TensorDataset:
        """Create a dataset of interchange intervention

        Samples `num_samples` instances of base and source inputs from
        `inputs`. Each sample consists of a random subset of the DAG nodes and
        random base and source inputs. It always returns the same number of
        source inputs, filling in for missing nodes with the base input.

        Parameters
        ----------
        inputs : torch.Tensor of shape (num_inputs, input_size)
            The inputs to sample from
        num_samples : int, default=10000
            The number of samples to take

        Returns
        -------
        dataset : TensorDataset
            A dataset of interchange intervention samples, consisting of the
            base inputs, source inputs, and gold (DAG) outputs
        """

        num_nodes = len(self.dag_nodes)
        base_input_indices = torch.randint(0, inputs.shape[0], (num_samples,))
        source_input_indices = torch.randint(
            0, inputs.shape[0], (num_samples, num_nodes)
        )

        base_inputs = inputs[base_input_indices]
        source_inputs = inputs[source_input_indices]

        gold_outputs = torch.empty((num_samples,), dtype=torch.long)

        for i in range(num_samples):
            dag_output = self.run_interchange_intervention(
                base_inputs[i], source_inputs[i], output_type="output_integer"
            )
            dag_output = dag_output[0]  # Assume there is only one output node. TODO
            gold_outputs[i] = dag_output

        # node_selected_mask = torch.randint(
        #     0, 2, (num_samples, num_nodes), dtype=torch.bool
        # )

        # source_inputs[node_selected_mask] = base_inputs[node_selected_mask]

        return TensorDataset(base_inputs, source_inputs, gold_outputs)

    def dii_training_objective_and_agreement(
        self,
        base_input: torch.Tensor,
        source_inputs: torch.Tensor,
        gold_output: Optional[torch.Tensor] = None,
        loss_fn: callable = F.cross_entropy,
    ):
        """Compute the training objective and accuracy of DII

        Returns both the loss and a boolean for agreement between performing
        distributed interchange intervention on the low-level model and
        interchange intervention on the DAG.

        Parameters
        ----------
        base_input : torch.Tensor of shape (input_size,)
            The base input to the model
        source_inputs : torch.Tensor of shape (num_dag_nodes, input_size)
            The source inputs to the model
        gold_output : torch.Tensor of shape (1,), optional
            The gold output. If not provided, it will be computed by running
            interchange intervention on the DAG.
        loss_fn : callable, default=F.cross_entropy
            The loss function to use

        Returns
        -------
        loss : torch.Tensor of shape (output_size,)
            The training objective
        agreement : bool
            Whether the low-level model and the DAG agree on the output after
            doing interventions
        """

        base_input = base_input.to(self.device)
        source_inputs = source_inputs.to(self.device)

        # Run distributed interchange intervention on the high-level model
        output_low_level = self.run_distributed_interchange_intervention(
            base_input, source_inputs
        )

        # Run interchange intervention on the DAG
        if gold_output is None:
            gold_output = self.run_interchange_intervention(
                base_input, source_inputs, output_type="output_integer"
            )
            # Assume there is only one output node. TODO
            gold_output = gold_output[0].to(self.device)

        # Compute the loss
        loss = loss_fn(output_low_level, gold_output)

        agreement = torch.equal(output_low_level.argmax(), gold_output)

        return loss, agreement

    def train_rotation_matrix(
        self,
        inputs: Optional[torch.Tensor] = None,
        ii_dataset: Optional[TensorDataset] = None,
        num_epochs: int = 10,
        lr: float = 0.01,
        num_samples: int = 10000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Train the rotation matrix to align the two models

        Exactly one of `inputs` or `ii_dataset` must be provided. If `inputs`
        is provided, it will be used to create an interchange intervention
        dataset. If `ii_dataset` is provided, it will be used directly.

        Parameters
        ----------
        inputs : torch.Tensor of shape (num_inputs, input_size), optional
            The input on which to train the rotation matrix. Base and source
            inputs will be samples from here
        ii_dataset : TensorDataset, optional
            The interchange intervention dataset to use
        num_epochs : int, default=0
            The number of epochs to train the rotation matrix for
        lr : float, default=0.01
            The learning rate to use
        num_samples : int, default=10000
            The number of samples to take from `inputs` to create the dataset,
            if `inputs` is provided.

        Returns
        -------
        losses : np.ndarray of shape (num_steps,)
            The average loss at each epoch
        accuracies : np.ndarray of shape (num_steps,)
            At each epoch the accuracy of the alignment between the low-level
            model and the DAG
        """

        if inputs is None and ii_dataset is None:
            raise ValueError("Either `inputs` or `ii_dataset` must be provided")

        if inputs is not None and ii_dataset is not None:
            raise ValueError("Only one of `inputs` or `ii_dataset` can be provided")

        if ii_dataset is None:
            if self.verbosity >= 1:
                print("Creating interchange intervention dataset...")
            ii_dataset = self.create_interchange_intervention_dataset(
                inputs, num_samples=num_samples
            )
        ii_dataloader = DataLoader(ii_dataset, batch_size=1)

        optimizer = torch.optim.SGD(self.rotation.parameters(), lr=lr)

        losses = np.empty(num_epochs)
        accuracies = np.empty(num_epochs)

        for epoch in range(num_epochs):
            total_loss = 0.0
            total_agreement = 0.0

            iterator = ii_dataloader
            if self.progress_bar:
                iterator = tqdm(
                    iterator,
                    total=len(ii_dataset),
                    desc=f"Epoch [{epoch+1}/{num_epochs}]",
                )
            for base_input, source_inputs, gold_output in iterator:
                base_input = base_input.squeeze(0)
                source_inputs = source_inputs.squeeze(0)
                gold_output = gold_output.squeeze(0)
                base_input = base_input.to(self.device)
                source_inputs = source_inputs.to(self.device)

                loss, agreement = self.dii_training_objective_and_agreement(
                    base_input, source_inputs, gold_output=gold_output
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_agreement += agreement

            losses[epoch] = total_loss / len(ii_dataset)
            accuracies[epoch] = total_agreement / len(ii_dataset)

            if self.verbosity >= 1:
                print(
                    f"Loss: {losses[epoch]:0.5f}, Accuracy: {accuracies[epoch]:0.5f}"
                )

        return losses, accuracies
    
    @torch.no_grad()
    def test_rotation_matrix(
        self,
        inputs: Optional[torch.Tensor] = None,
        ii_dataset: Optional[TensorDataset] = None,
        num_samples: int = 1000,
    ) -> tuple[float, float]:
        """Test the rotation matrix for alignment between the two models

        Exactly one of `inputs` or `ii_dataset` must be provided. If `inputs`
        is provided, it will be used to create an interchange intervention
        dataset. If `ii_dataset` is provided, it will be used directly.

        Parameters
        ----------
        inputs : torch.Tensor of shape (num_inputs, input_size), optional
            The input on which to train the rotation matrix. Base and source
            inputs will be samples from here
        ii_dataset : TensorDataset, optional
            The interchange intervention dataset to use
        num_samples : int, default=1000
            The number of samples to take from `inputs` to create the dataset,
            if `inputs` is provided.

        Returns
        -------
        loss : float
            The average loss
        accuracy : float
            The accuracy of the alignment between the low-level model and the
            DAG
        """

        if inputs is None and ii_dataset is None:
            raise ValueError("Either `inputs` or `ii_dataset` must be provided")

        if inputs is not None and ii_dataset is not None:
            raise ValueError("Only one of `inputs` or `ii_dataset` can be provided")

        if ii_dataset is None:
            if self.verbosity >= 1:
                print("Creating interchange intervention dataset...")
            ii_dataset = self.create_interchange_intervention_dataset(
                inputs, num_samples=num_samples
            )

        ii_dataloader = DataLoader(ii_dataset, batch_size=1)

        total_loss = 0.0
        total_agreement = 0.0

        iterator = ii_dataloader
        if self.progress_bar:
            iterator = tqdm(
                iterator,
                total=len(ii_dataset),
                desc=f"Testing",
            )
        for base_input, source_inputs, gold_output in iterator:
            base_input = base_input.squeeze(0)
            source_inputs = source_inputs.squeeze(0)
            gold_output = gold_output.squeeze(0)
            base_input = base_input.to(self.device)
            source_inputs = source_inputs.to(self.device)

            loss, agreement = self.dii_training_objective_and_agreement(
                base_input, source_inputs, gold_output=gold_output
            )

            total_loss += loss.item()
            total_agreement += agreement

        loss = total_loss / len(ii_dataset)
        accuracy = total_agreement / len(ii_dataset)

        return loss, accuracy