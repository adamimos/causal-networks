from functools import partial
from typing import Callable, Iterable, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import IterableDataset, DataLoader, TensorDataset

import numpy as np

from tqdm import tqdm

from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens import HookedTransformer

from .dag import DeterministicDAG, RecurrentDeterministicDAG


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
        return torch.matmul(x, self.get_orthogonal_matrix())

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.get_orthogonal_matrix().T)

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
                0, 2, (self.num_nodes,), dtype=torch.bool
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
        The device to put everything on. If None, use CUDA if available,
        otherwise CPU
    verbosity : int, default=1
        The verbosity level
    progress_bar : bool, default=True
        Whether to show a progress bar during and dataset generation
    """

    def __init__(
        self,
        dag: DeterministicDAG,
        low_level_model: HookedRootModule,
        dag_nodes: list[str],
        input_alignment: callable,
        output_alignment: callable,
        intervene_model_hooks: list[str],
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
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        self.num_nodes = len(dag_nodes)

        self.layer_sizes = self._determine_layer_sizes()
        self.total_space_size = sum(self.layer_sizes)

        self.y_0_size = self.total_space_size - sum(self.subspaces_sizes)
        self.subspaces_sizes_with_y0 = [self.y_0_size] + self.subspaces_sizes

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
        base_input : torch.Tensor of shape (batch_size, input_size) or (input_size,)
            The base input to the model
        source_inputs : torch.Tensor of shape (batch_size, num_dag_nodes, input_size) or (num_dag_nodes, input_size)
            The source inputs to the model

        Returns
        -------
        list[tuple[str, callable]]
            The hooks to use for distributed interchange intervention, for
            each of `self.intervene_model_hooks`
        """

        if source_inputs.shape[-2] != self.num_nodes:
            raise ValueError(
                f"Expected {self.num_nodes} source inputs, got {source_inputs.shape[0]}"
            )

        if base_input.ndim == 1:
            base_input = base_input.unsqueeze(0)

        if source_inputs.ndim == 2:
            source_inputs = source_inputs.unsqueeze(0)

        batch_size = base_input.shape[0]

        base_input = base_input.to(self.device)
        source_inputs = source_inputs.to(self.device)

        # Combine the base input and the source inputs
        # (batch_size, num_dag_nodes + 1, input_size)
        combined_inputs = torch.cat((base_input.unsqueeze(1), source_inputs), dim=1)

        # (batch_size * (num_dag_nodes + 1), input_size)
        combined_inputs_flattened = combined_inputs.flatten(0, 1)

        def store_activation_value(
            value, hook: HookPoint, activation_values, space_index
        ):
            activation_values[:, space_index : space_index + value.shape[1]] = value

        activation_values = torch.empty(
            combined_inputs_flattened.shape[0],
            self.total_space_size,
            device=self.device,
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
            self.low_level_model.run_with_hooks(
                combined_inputs_flattened, fwd_hooks=fwd_hooks
            )

        # (batch_size * (num_dag_nodes + 1), total_space_size)
        rotated_activation_values = self.rotation(activation_values)

        # Select the rotated activation values which are relevant, for patching in
        indices_list = []
        for i, subspace_size in enumerate(self.subspaces_sizes_with_y0):
            indices_list.append(torch.ones(subspace_size, device=self.device) * i)
        selection_indices = torch.cat(indices_list).long()
        selection_indices = selection_indices.tile((batch_size, 1))
        selection_indices = selection_indices + torch.arange(
            batch_size, device=self.device
        ).unsqueeze(1) * (self.num_nodes + 1)
        # (batch_size, total_space_size)
        new_rotated_activation_values = rotated_activation_values[
            selection_indices, torch.arange(self.total_space_size).tile((batch_size, 1))
        ]

        # Unrotate the new rotated activation values, to get a vector
        # consisting of all the activation values to be patched in
        # (batch_size, total_space_size)
        new_activation_values = self.rotation.inverse(new_rotated_activation_values)

        def intervention_hook(
            value,
            hook: HookPoint,
            space_index,
            new_activation_values,
        ):
            value[:] = new_activation_values[
                ..., space_index : space_index + value.shape[-1]
            ]
            return value

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
    ) -> torch.Tensor:
        """Compute the output of the low-level model on base and source inputs

        Does distributed interchange intervention, patching in the activation values
        from the source inputs, then running on the base input

        **Warning**: This method will disable gradients for the model parameters,
        setting `requires_grad` to `False`.

        Parameters
        ----------
        base_input : torch.Tensor of shape (batch_size, input_size) or (input_size,)
            The base input to the model
        source_inputs : torch.Tensor of shape (batch_size, num_dag_nodes, input_size) or
        (num_dag_nodes, input_size)
            The source inputs to the model

        Returns
        -------
        output : torch.Tensor of shape (batch_size, output_size) or (output_size,)
            The output of the low-level model on the base input and source inputs
        """

        if base_input.ndim == 1:
            squeeze_output = True
            base_input = base_input.unsqueeze(0)
        else:
            squeeze_output = False

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

        if squeeze_output:
            output = output.squeeze(0)

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

        if self.verbosity >= 1:
            print("Creating interchange intervention dataset...")

        num_inputs = inputs.shape[0]
        input_size = inputs.shape[1]
        num_nodes = self.num_nodes
        base_input_indices = torch.randint(0, num_inputs, (num_samples,))
        source_input_indices = torch.randint(0, num_inputs, (num_samples, num_nodes))

        base_inputs = inputs[base_input_indices]
        source_inputs = inputs[source_input_indices]

        node_selected_mask = torch.randint(
            0, 2, (num_samples, num_nodes), dtype=torch.bool
        )

        source_inputs = torch.where(
            node_selected_mask.unsqueeze(2).tile(1, 1, input_size),
            source_inputs,
            base_inputs.unsqueeze(1).tile(1, num_nodes, 1),
        )

        gold_outputs = torch.empty((num_samples,), dtype=torch.long)

        for i in range(num_samples):
            dag_output = self.run_interchange_intervention(
                base_inputs[i], source_inputs[i], output_type="output_integer"
            )
            dag_output = dag_output[0]  # Assume there is only one output node. TODO
            gold_outputs[i] = dag_output

        return TensorDataset(base_inputs, source_inputs, gold_outputs)

    def dii_training_objective_and_agreement(
        self,
        base_input: torch.Tensor,
        source_inputs: torch.Tensor,
        gold_outputs: Optional[torch.Tensor] = None,
        loss_fn: callable = F.cross_entropy,
    ):
        """Compute the training objective and accuracy of DII

        Returns both the loss and a boolean for agreement between performing
        distributed interchange intervention on the low-level model and
        interchange intervention on the DAG.

        Parameters
        ----------
        base_input : torch.Tensor of shape (batch_size, input_size) or
        (input_size,)
            The base input to the model
        source_inputs : torch.Tensor of shape (batch_size, num_dag_nodes,
        input_size) or (num_dag_nodes, input_size)
            The source inputs to the model
        gold_outputs : torch.Tensor of shape (batch_size,), optional
            The gold output. If not provided, it will be computed by running
            interchange intervention on the DAG.
        loss_fn : callable, default=F.cross_entropy
            The loss function to use

        Returns
        -------
        loss : torch.Tensor of shape (1,)
            The training objective output
        agreements : torch.Tensor of shape (batch_size,)
            Whether the low-level model and the DAG agree on the output after
            doing interventions
        """

        if base_input.ndim == 1:
            base_input = base_input.unsqueeze(0)
        if source_inputs.ndim == 2:
            source_inputs = source_inputs.unsqueeze(0)

        batch_size = base_input.shape[0]

        base_input = base_input.to(self.device)
        source_inputs = source_inputs.to(self.device)

        # Run distributed interchange intervention on the high-level model
        output_low_level = self.run_distributed_interchange_intervention(
            base_input, source_inputs
        )

        # Run interchange intervention on the DAG
        if gold_outputs is None:
            gold_outputs = []
            for i in range(batch_size):
                gold_output = self.run_interchange_intervention(
                    base_input[i], source_inputs[i], output_type="output_integer"
                )
                # Assume there is only one output node. TODO
                gold_output = gold_output[0].to(self.device)
                gold_outputs.append(gold_output)
            gold_outputs = torch.stack(gold_outputs)

        # Compute the loss
        loss = loss_fn(output_low_level, gold_outputs)

        agreements = output_low_level.argmax(dim=-1) == gold_outputs

        return loss, agreements

    def train_rotation_matrix(
        self,
        inputs: Optional[torch.Tensor] = None,
        ii_dataset: Optional[TensorDataset] = None,
        num_epochs: int = 10,
        lr: float = 0.01,
        batch_size: int = 32,
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
        batch_size : int, default=32
            The batch size to use
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
            ii_dataset = self.create_interchange_intervention_dataset(
                inputs, num_samples=num_samples
            )
        ii_dataloader = DataLoader(ii_dataset, batch_size=batch_size)

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
                    total=len(ii_dataloader),
                    desc=f"Epoch [{epoch+1}/{num_epochs}]",
                )
            for base_input, source_inputs, gold_outputs in iterator:
                base_input = base_input.to(self.device)
                source_inputs = source_inputs.to(self.device)
                gold_outputs = gold_outputs.to(self.device)

                loss, agreements = self.dii_training_objective_and_agreement(
                    base_input, source_inputs, gold_outputs=gold_outputs
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_agreement += agreements.float().mean().item()

            losses[epoch] = total_loss / len(ii_dataloader)
            accuracies[epoch] = total_agreement / len(ii_dataloader)

            if self.verbosity >= 1:
                print(f"Loss: {losses[epoch]:0.5f}, Accuracy: {accuracies[epoch]:0.5f}")

        return losses, accuracies

    @torch.no_grad()
    def test_rotation_matrix(
        self,
        inputs: Optional[torch.Tensor] = None,
        ii_dataset: Optional[TensorDataset] = None,
        batch_size: int = 32,
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
        batch_size : int, default=32
            The batch size to use
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

        ii_dataloader = DataLoader(ii_dataset, batch_size=batch_size)

        total_loss = 0.0
        total_agreement = 0.0

        iterator = ii_dataloader
        if self.progress_bar:
            iterator = tqdm(
                iterator,
                total=len(ii_dataloader),
                desc=f"Testing",
            )
        for base_input, source_inputs, gold_outputs in iterator:
            base_input = base_input.to(self.device)
            source_inputs = source_inputs.to(self.device)
            gold_outputs = gold_outputs.to(self.device)

            loss, agreement = self.dii_training_objective_and_agreement(
                base_input, source_inputs, gold_outputs=gold_outputs
            )

            total_loss += loss.item()
            total_agreement += agreement.float().mean().item()

        loss = total_loss / len(ii_dataloader)
        accuracy = total_agreement / len(ii_dataloader)

        return loss, accuracy


class VariableAlignmentTransformer(VariableAlignment):
    """A variable alignment between a recurrent DAG and a transformer

    Parameters
    ----------
    dag : RecurrentDeterministicDAG
        The DAG to align to the model
    low_level_model : HookedTransformer
        The hooked low-level model to align to the DAG
    dag_nodes : list[str]
        The names of the nodes (variables) of the DAG to align
    input_alignment : callable
        A function mapping (batches of) model input tensors to DAG input dictionaries
    output_alignment : callable
        A function mapping (batches of) model output tensors to DAG output values
    intervene_model_hooks : list[str]
        The names of the model hooks together give the whole activation space for
        alignment the DAG nodes
    subspaces_sizes : list[int]
        The sizes of the subspaces to use for each DAG nodes
    output_modifier : callable, optional
        A function to modify the output of the low-level model before comparing it with
        the DAG output
    device : Union[str, torch.device], default=None
        The device to put everything on. If None, use CUDA if available, otherwise CPU
    verbosity : int, default=1
        The verbosity level
    progress_bar : bool, default=True
        Whether to show a progress bar during and dataset generation
    """

    dag: RecurrentDeterministicDAG

    def __init__(
        self,
        dag: RecurrentDeterministicDAG,
        low_level_model: HookedTransformer,
        dag_nodes: list[str],
        input_alignment: callable,
        output_alignment: callable,
        intervene_model_hooks: list[str],
        subspaces_sizes: list[int],
        output_modifier: Optional[callable] = None,
        device: Optional[Union[str, torch.device]] = None,
        verbosity: int = 1,
        progress_bar: bool = True,
    ):
        super().__init__(
            dag=dag,
            low_level_model=low_level_model,
            dag_nodes=dag_nodes,
            input_alignment=input_alignment,
            output_alignment=output_alignment,
            intervene_model_hooks=intervene_model_hooks,
            subspaces_sizes=subspaces_sizes,
            device=device,
            verbosity=verbosity,
            progress_bar=progress_bar,
        )

        self.output_modifier = output_modifier

        self.d_vocab = self.low_level_model.cfg.d_vocab

    def _determine_layer_sizes(self) -> list[int]:
        """Run the model to determine the sizes of the activation spaces"""

        if self.verbosity > 0:
            print("Running model to determine activation space size...")

        sizes = []

        def counter_hook(value, hook: HookPoint, sizes):
            sizes.append(value.shape[-1])

        partial_counter_hook = partial(counter_hook, sizes=sizes)

        x = torch.zeros(1, 1, device=self.device, dtype=torch.long)
        fwd_hooks = [
            (name, partial_counter_hook) for name in self.intervene_model_hooks
        ]

        self.low_level_model.run_with_hooks(x, fwd_hooks=fwd_hooks)

        return sizes

    def get_distributed_interchange_intervention_hooks(
        self,
        base_input: torch.Tensor,
        source_inputs: torch.Tensor,
        minibatch_size: int = 32,
    ) -> list[tuple[str, callable]]:
        """Get hooks for distributed interchange intervention

        Runs the model on the base input and each source input, and on input `i` records
        the value of the projection of the rotated activation space onto the `i`th
        subspace. It then builds hooks to patch in these values into the rotated
        activation space.

        Parameters
        ----------
        base_input : torch.Tensor of shape (batch_size, seq_len) or (seq_len)
            The base input to the model
        source_inputs : torch.Tensor of shape (batch_size, num_dag_nodes, seq_len,
        seq_len) or (num_dag_nodes, seq_len, seq_len)
            The source inputs to the model, one for each batch element, DAG node and
            token position
        minibatch_size : int, default=32
            The batch size to use for running the model. The base and source inputs are
            split into minibatches of this size, and the model is run on each minibatch.

        Returns
        -------
        list[tuple[str, callable]]
            The hooks to use for distributed interchange intervention, for each of
            `self.intervene_model_hooks`
        """

        if source_inputs.shape[-3] != self.num_nodes:
            raise ValueError(
                f"Expected {self.num_nodes} source inputs, got {source_inputs.shape[-3]}"
            )

        # (batch_size, seq_len)
        if base_input.ndim == 1:
            base_input = base_input.unsqueeze(0)

        # (batch_size, num_dag_nodes, seq_len, seq_len)
        if source_inputs.ndim == 3:
            source_inputs = source_inputs.unsqueeze(0)

        if base_input.shape[0] != source_inputs.shape[0]:
            raise ValueError(
                f"Expected batch size of base input and source inputs to be the same,"
                f" got {base_input.shape[0]} and {source_inputs.shape[0]}"
            )

        batch_size = base_input.shape[0]
        seq_len = base_input.shape[1]

        base_input = base_input.to(self.device)
        source_inputs = source_inputs.to(self.device)

        # Combine the base input and the source inputs
        # (batch_size, 1 + num_dag_nodes * seq_len, seq_len)
        combined_inputs = torch.cat(
            (
                base_input.view(batch_size, 1, seq_len),
                source_inputs.view(batch_size, -1, seq_len),
            ),
            dim=1,
        )

        # (batch_size * (1 + num_dag_nodes * seq_len), seq_len)
        combined_inputs_flattened = combined_inputs.view(-1, seq_len)

        super_batch_size = combined_inputs_flattened.shape[0]

        # (batch_size * (1 + num_dag_nodes * seq_len), seq_len, total_space_size)
        activation_values = torch.empty(
            (super_batch_size, seq_len, self.total_space_size),
            device=self.device,
        )

        def store_activation_value(
            value, hook: HookPoint, activation_values, space_index, minibatch_start
        ):
            activation_values[
                minibatch_start : minibatch_start + value.shape[0],
                ...,
                space_index : space_index + value.shape[-1],
            ] = value

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

        if self.verbosity >= 2:
            print("Running model to determine activation values on source inputs...")

        with torch.no_grad():
            for minibatch_start in range(0, super_batch_size, minibatch_size):
                batch_fwd_hooks = []
                for name, hook in fwd_hooks:
                    batch_fwd_hooks.append(
                        (name, partial(hook, minibatch_start=minibatch_start))
                    )
                self.low_level_model.run_with_hooks(
                    combined_inputs_flattened[
                        minibatch_start : minibatch_start + minibatch_size, ...
                    ],
                    fwd_hooks=batch_fwd_hooks,
                )

        # (batch_size * (1 + num_dag_nodes * seq_len), seq_len, total_space_size)
        rotated_activation_values = self.rotation(activation_values)

        # Select the rotated activation values which are relevant, for patching in. We
        # select the indices of Y_0 for the base input across all positions at once, and
        # select indices of each Y_i for each position, since we patch in for each tuple
        # (node, stream) separately (these are dimensions 1 and 2 of `source_inputs`).
        super_batch_indices = torch.empty(
            (batch_size, seq_len, self.total_space_size), device=self.device, dtype=int
        )
        pos_indices = torch.arange(seq_len, device=self.device, dtype=int)
        pos_indices = pos_indices.view(1, -1, 1)
        space_indices = torch.arange(
            self.total_space_size, device=self.device, dtype=int
        )
        space_indices = space_indices.view(1, 1, -1)

        # Compute the indices for the super batch dimension (i.e. then batch_size * (1 +
        # num_dag_nodes * seq_len) one). Select Y_0 across all positions, and each Y_i
        # for each position.
        sub_batch_size = 1 + self.num_nodes * seq_len
        super_batch_indices[:, :, : self.y_0_size] = (
            torch.arange(batch_size).view(batch_size, 1, 1) * sub_batch_size
        )
        space_index = self.y_0_size
        for i, subspace_size in enumerate(self.subspaces_sizes, start=1):
            super_batch_indices[:, :, space_index : space_index + subspace_size] = (
                torch.arange(batch_size).view(batch_size, 1, 1) * sub_batch_size
                + (i - 1) * seq_len
                + torch.arange(seq_len).view(1, seq_len, 1)
            )
            space_index += subspace_size

        # (batch_size, seq_len, total_space_size)
        new_rotated_activation_values = rotated_activation_values[
            super_batch_indices, pos_indices, space_indices
        ]

        # Unrotate the new rotated activation values, to get a vector
        # consisting of all the activation values to be patched in
        # (batch_size, seq_len, total_space_size)
        new_activation_values = self.rotation.inverse(new_rotated_activation_values)

        def intervention_hook(
            value,
            hook: HookPoint,
            space_index,
            new_activation_values,
        ):
            value[:] = new_activation_values[
                ..., space_index : space_index + value.shape[-1]
            ]
            return value

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
        self,
        base_input: torch.Tensor,
        source_inputs: torch.Tensor,
        minibatch_size: int = 32,
    ) -> torch.Tensor:
        """Compute the output of the low-level model on base and source inputs

        Does distributed interchange intervention, patching in the activation values
        from the source inputs, then running on the base input

        **Warning**: This method will disable gradients for the model parameters,
        setting `requires_grad` to `False`.

        Parameters
        ----------
        base_input : torch.Tensor of shape (batch_size, seq_len) or (seq_len,)
            The base input to the model
        source_inputs : torch.Tensor of shape (batch_size, num_dag_nodes, seq_len,
        seq_len) or  (num_dag_nodes, seq_len, seq_len)
            The source inputs to the model
        minibatch_size : int, default=32
            The batch size to use for running the model. The base and source inputs are
            split into minibatches of this size, and the model is run on each minibatch.

        Returns
        -------
        output : torch.Tensor of shape (batch_size, seq_len, ...) or (seq_len, ...)
            The output of the low-level model (plus the output modifier, if set) on the
            base input and source inputs
        """

        if base_input.ndim == 1:
            squeeze_output = True
            base_input = base_input.unsqueeze(0)
        else:
            squeeze_output = False

        base_input = base_input.to(self.device)
        source_inputs = source_inputs.to(self.device)

        # Get the patches to apply
        fwd_hooks = self.get_distributed_interchange_intervention_hooks(
            base_input, source_inputs, minibatch_size=minibatch_size
        )

        # Run the model with the patches applied, disabling gradients for the
        # model parameters, but not the rotation matrix used in the hooks
        self.low_level_model.requires_grad_(False)
        output = self.low_level_model.run_with_hooks(base_input, fwd_hooks=fwd_hooks)

        if squeeze_output:
            output = output.squeeze(0)

        if self.output_modifier is not None:
            output = self.output_modifier(output)

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
        base_input : torch.Tensor of shape (seq_len, )
            The base input to the model
        source_inputs : torch.Tensor of shape (num_dag_nodes, seq_len, seq_len)
            The source inputs to the model
        output_type : str, optional
            The type of output to return, by default "output_integer"

        Returns
        -------
        dag_output : torch.Tensor of shape (num_output_nodes, seq_len)
            The output of the dag, as integers ranging over the possible
            output values
        """

        seq_len = base_input.shape[0]

        # (num_dag_nodes * seq_len, seq_len)
        source_inputs = source_inputs.view(-1, seq_len)

        # Convert the inputs into things the DAG can handle
        base_input_dag = self.input_alignment(base_input)
        source_inputs_dag = self.input_alignment(source_inputs)
        dag_node_singletons = [
            [(dag_node, stream)]
            for dag_node in self.dag_nodes
            for stream in range(seq_len)
        ]

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

    def dii_training_objective_and_agreement(
        self,
        base_input: torch.Tensor,
        source_inputs: torch.Tensor,
        gold_outputs: Optional[torch.Tensor] = None,
        loss_fn: callable = F.cross_entropy,
        minibatch_size: int = 32,
    ):
        """Compute the training objective and accuracy of DII

        Returns both the loss and a boolean for agreement between performing distributed
        interchange intervention on the low-level model and interchange intervention on
        the DAG.

        Parameters
        ----------
        base_input : torch.Tensor of shape (batch_size, seq_len) or (seq_len,)
            The base input to the model
        source_inputs : torch.Tensor of shape (batch_size, num_dag_nodes, seq_len,
        seq_len) or (num_dag_nodes, seq_len, seq_len)
            The source inputs to the model
        gold_outputs : torch.Tensor of shape (batch_size, seq_len), optional
            The gold outputs. If not provided, it will be computed by running
            interchange intervention on the DAG.
        loss_fn : callable, default=F.cross_entropy
            The loss function to use
        minibatch_size : int, default=32
            The batch size to use for running the model. The base and source inputs are
            split into minibatches of this size, and the model is run on each minibatch.

        Returns
        -------
        loss : torch.Tensor of shape (1,)
            The training objective output
        agreements : torch.Tensor of shape (batch_size, seq_len)
            Whether the low-level model and the DAG agree on the output after doing
            interventions
        """
        if base_input.ndim == 1:
            base_input = base_input.unsqueeze(0)
        if source_inputs.ndim == 2:
            source_inputs = source_inputs.unsqueeze(0)

        batch_size = base_input.shape[0]

        base_input = base_input.to(self.device)
        source_inputs = source_inputs.to(self.device)

        # Run distributed interchange intervention on the high-level model
        output_low_level = self.run_distributed_interchange_intervention(
            base_input, source_inputs, minibatch_size=minibatch_size
        )

        # Run interchange intervention on the DAG
        if gold_outputs is None:
            gold_outputs = []
            for i in range(batch_size):
                gold_output = self.run_interchange_intervention(
                    base_input[i], source_inputs[i], output_type="output_integer"
                )
                # Assume there is only one output node. TODO
                gold_output = gold_output[0].to(self.device)
                gold_outputs.append(gold_output)
            gold_outputs = torch.stack(gold_outputs)

        # Compute the loss
        loss = loss_fn(output_low_level.flatten(0, 1), gold_outputs.flatten(0, 1))

        agreements = output_low_level.argmax(dim=-1) == gold_outputs

        return loss, agreements

    def create_interchange_intervention_dataset(
        self, inputs: torch.Tensor, num_samples=10000
    ) -> TensorDataset:
        """Create a dataset of interchange intervention

        Samples `num_samples` instances of base and source inputs from `inputs`. Each
        sample consists of a random subset of the DAG nodes and streams and random base
        and source inputs. It always returns the same number of source inputs, filling
        in for missing nodes with the base input.

        Parameters
        ----------
        inputs : torch.Tensor of shape (num_inputs, seq_len)
            The inputs to sample from
        num_samples : int, default=10000
            The number of samples to take

        Returns
        -------
        dataset : TensorDataset
            A dataset of interchange intervention samples, consisting of the base
            inputs, source inputs, and gold (DAG) outputs
        """

        if self.verbosity >= 1:
            print("Creating interchange intervention dataset...")

        num_inputs = inputs.shape[0]
        seq_len = inputs.shape[1]
        num_nodes = self.num_nodes

        # Set the number of DAG streams to be the same as the number of tokens (i.e.
        # `seq_len`)
        self.dag.num_streams = seq_len

        if self.verbosity >= 2:
            print("[1/4] Sampling base and source inputs...")

        base_input_indices = torch.randint(0, num_inputs, (num_samples,))
        source_input_indices = torch.randint(
            0, num_inputs, (num_samples, num_nodes, seq_len)
        )

        # (num_samples, seq_len)
        base_inputs = inputs[base_input_indices]

        # (num_samples, num_nodes, seq_len, seq_len)
        source_inputs = inputs[source_input_indices]

        if self.verbosity >= 2:
            print("[2/4] Sampling node and stream mask...")

        # (num_samples, num_nodes, seq_len)
        node_stream_selected_mask = torch.randint(
            0,
            2,
            (num_samples, num_nodes, seq_len),
            dtype=torch.bool,
            device=self.device,
        )

        # (num_samples, num_nodes, seq_len, seq_len)
        base_inputs_expanded = base_inputs.view(num_samples, 1, 1, seq_len)
        base_inputs_expanded = base_inputs_expanded.tile(1, num_nodes, seq_len, 1)

        # (num_samples, num_nodes, seq_len, seq_len)
        mask_expanded = node_stream_selected_mask.view(
            num_samples, num_nodes, seq_len, 1
        )
        mask_expanded = mask_expanded.tile(1, 1, 1, seq_len)

        if self.verbosity >= 2:
            print("[3/4] Selecting from source inputs using mask...")

        # (num_samples, num_nodes, seq_len, seq_len)
        source_inputs = torch.where(
            mask_expanded,
            source_inputs,
            base_inputs_expanded,
        )

        if self.verbosity >= 2:
            print("[4/4] Computing gold outputs...")

        gold_outputs = torch.empty((num_samples, seq_len), dtype=torch.long)

        iterator = range(num_samples)
        if self.verbosity >= 1 and self.progress_bar:
            iterator = tqdm(iterator, desc="Computing gold outputs")
        for i in iterator:
            dag_output = self.run_interchange_intervention(
                base_inputs[i], source_inputs[i], output_type="output_integer"
            )
            dag_output = dag_output[0]  # Assume there is only one output node. TODO
            gold_outputs[i] = dag_output

        return TensorDataset(base_inputs, source_inputs, gold_outputs)

    def train_rotation_matrix(
        self,
        inputs: Optional[torch.Tensor] = None,
        ii_dataset: Optional[TensorDataset] = None,
        num_epochs: int = 10,
        lr: float = 0.01,
        batch_size: int = 32,
        minibatch_size: int = 32,
        num_samples: int = 10000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Train the rotation matrix to align the two models

        Exactly one of `inputs` or `ii_dataset` must be provided. If `inputs`
        is provided, it will be used to create an interchange intervention
        dataset. If `ii_dataset` is provided, it will be used directly.

        Parameters
        ----------
        inputs : torch.Tensor of shape (num_inputs, seq_len), optional
            The input on which to train the rotation matrix. Base and source
            inputs will be samples from here
        ii_dataset : TensorDataset, optional
            The interchange intervention dataset to use
        num_epochs : int, default=0
            The number of epochs to train the rotation matrix for
        lr : float, default=0.01
            The learning rate to use
        batch_size : int, default=32
            The batch size to use
        minibatch_size : int, default=32
            The batch size to use for running the model. The base and source inputs are
            split into minibatches of this size, and the model is run on each minibatch.
        num_samples : int, default=10000
            The number of samples to take from `inputs` to create the dataset,
            if `inputs` is provided.

        Returns
        -------
        losses : np.ndarray of shape (num_epochs,)
            The average loss at each epoch
        accuracies : np.ndarray of shape (num_epochs,)
            At each epoch the accuracy of the alignment between the low-level
            model and the DAG
        """

        if inputs is None and ii_dataset is None:
            raise ValueError("Either `inputs` or `ii_dataset` must be provided")

        if inputs is not None and ii_dataset is not None:
            raise ValueError("Only one of `inputs` or `ii_dataset` can be provided")

        if ii_dataset is None:
            ii_dataset = self.create_interchange_intervention_dataset(
                inputs, num_samples=num_samples
            )
        ii_dataloader = DataLoader(ii_dataset, batch_size=batch_size)

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
                    total=len(ii_dataloader),
                    desc=f"Epoch [{epoch+1}/{num_epochs}]",
                )
            for base_input, source_inputs, gold_outputs in iterator:
                base_input = base_input.to(self.device)
                source_inputs = source_inputs.to(self.device)
                gold_outputs = gold_outputs.to(self.device)

                loss, agreements = self.dii_training_objective_and_agreement(
                    base_input,
                    source_inputs,
                    gold_outputs=gold_outputs,
                    minibatch_size=minibatch_size,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_agreement += agreements.float().mean().item()

            losses[epoch] = total_loss / len(ii_dataloader)
            accuracies[epoch] = total_agreement / len(ii_dataloader)

            if self.verbosity >= 1:
                print(f"Loss: {losses[epoch]:0.5f}, Accuracy: {accuracies[epoch]:0.5f}")

        return losses, accuracies
