from functools import partial
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import IterableDataset, DataLoader

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
    verbosity : int, default=1
        The verbosity level. 0 means no output, 1 means output setup messages,
        2 means output message during intervention and training
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
        verbosity: int = 1,
        progress_bar: bool = True,
    ):
        if len(dag_nodes) != len(subspaces_sizes):
            raise ValueError(
                f"Expected {len(dag_nodes)} subspaces sizes, got {len(subspaces_sizes)}"
            )

        self.dag = dag
        self.low_level_model = low_level_model
        self.dag_nodes = dag_nodes
        self.input_alignment = input_alignment
        self.output_alignment = output_alignment
        self.intervene_model_hooks = intervene_model_hooks
        self.subspaces_sizes = subspaces_sizes
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

    def _determine_layer_sizes(self) -> list[int]:
        """Run the model to determine the sizes of the activation spaces"""

        if self.verbosity > 0:
            print("Running model to determine activation space size...")

        sizes = []

        def counter_hook(value, hook: HookPoint, sizes):
            sizes.append(value.shape[1])

        partial_counter_hook = partial(counter_hook, sizes=sizes)

        x = torch.zeros(1, self.low_level_model.cfg.input_size)
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

        # Combine the base input and the source inputs
        combined_inputs = torch.cat((base_input, source_inputs), dim=0)

        def store_activation_value(
            value, hook: HookPoint, activation_values, space_index
        ):
            activation_values[:, space_index : space_index + value.shape[1]] = value

        activation_values = torch.empty(len(combined_inputs), self.total_space_size)

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
            indices_list.append(torch.ones(subspace_size) * i)
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
    ) -> torch.Tensor:
        """Run interchange intervention on the DAG

        Parameters
        ----------
        base_input : torch.Tensor of shape (input_size,)
            The base input to the model
        source_inputs : torch.Tensor of shape (num_dag_nodes, input_size)
            The source inputs to the model

        Returns
        -------
        dag_output_distribution : torch.Tensor of shape (num_output_nodes,
        num_outclasses)
            A delta distribution over the output nodes of the DAG, after
            running with source inputs and base input
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
            output_type="output_distribution",
            output_format="torch",
        )

        return output

    def dii_training_objective_and_agreement(
        self,
        base_input: torch.Tensor,
        source_inputs: torch.Tensor,
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

        # Run distributed interchange intervention on the high-level model
        output_low_level = self.run_distributed_interchange_intervention(
            base_input, source_inputs
        )

        # Run interchange intervention on the DAG
        output_dag = self.run_interchange_intervention(
            base_input, source_inputs
        ).float()

        # Assume there is only one output node. TODO
        output_dag = output_dag[0]

        # Compute the loss
        loss = loss_fn(output_low_level, output_dag)

        agreement = torch.equal(output_low_level.argmax(), output_dag.argmax())

        return loss, agreement

    def train_rotation_matrix(
        self, inputs: torch.Tensor, num_steps: int = 100000, lr: float = 0.01
    ):
        """Train the rotation matrix to align the two models

        Parameters
        ----------
        inputs : torch.Tensor of shape (num_inputs, input_size)
            The input on which to train the rotation matrix. Base and source
            inputs will be samples from here
        num_steps : int, default=100000
            The number of steps to train the rotation matrix for

        Returns
        -------
        losses : np.ndarray of shape (num_steps,)
            The loss at each step
        agreements : np.ndarray of shape (num_steps,)
            At each step, whether the low-level model and the DAG agree on the
            inputs
        """

        ii_dataset = InterchangeInterventionDataset(inputs, self.dag_nodes, num_steps)
        # ii_dataloader = DataLoader(ii_dataset, batch_size=1)

        optimizer = torch.optim.SGD(self.rotation.parameters(), lr=lr)

        losses = np.empty(num_steps)
        agreements = np.empty(num_steps)

        iterator = enumerate(ii_dataset)
        if self.progress_bar:
            iterator = tqdm(iterator, total=num_steps)
        for step, (nodes, base_input, source_inputs) in iterator:

            loss, agreement = self.dii_training_objective_and_agreement(
                base_input, source_inputs
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses[step] = loss.item()
            agreements[step] = agreement

        return losses, agreements
