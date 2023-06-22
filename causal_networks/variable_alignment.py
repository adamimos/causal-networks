from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from tqdm import tqdm

from transformer_lens.hook_points import HookedRootModule, HookPoint

from .dag import DeterministicDAG


class ParametrisedRotation(nn.Module):
    """A parametrised rotation matrix"""

    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.theta = nn.Linear(size, size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.theta(x)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return self.theta(x).T


class VariableAlignment:
    """A variable alignment between a DAG and a hooked model

    Parameters
    ----------
    dag : DeterministicDAG
        The DAG to align to the model
    hooked_model : HookedRootModule
        The hooked model to align to the DAG
    dag_nodes : list[str]
        The names of the nodes (variables) of the DAG to align
    input_alignment : callable
        A function mapping model input tensors to DAG input dictionaries
    output_alignment : callable
        A function mapping model output tensors to DAG output values
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
        hooked_model: HookedRootModule,
        dag_nodes: list[str],
        input_alignment: callable,
        output_alignment: callable,
        intervene_model_hooks: list[str],
        subspaces_sizes: list[int],
        verbosity: int = 1,
    ):
        if len(dag_nodes) != len(subspaces_sizes):
            raise ValueError(
                f"Expected {len(dag_nodes)} subspaces sizes, got {len(subspaces_sizes)}"
            )

        self.dag = dag
        self.hooked_model = hooked_model
        self.dag_nodes = dag_nodes
        self.input_alignment = input_alignment
        self.output_alignment = output_alignment
        self.intervene_model_hooks = intervene_model_hooks
        self.subspaces_sizes = subspaces_sizes
        self.verbosity = verbosity

        self.space_sizes = self._determine_space_sizes()
        self.total_space_size = sum(self.space_sizes)

        if sum(self.subspaces_sizes) > self.total_space_size:
            raise ValueError(
                f"Sum of subspace sizes ({sum(self.subspaces_sizes)}) "
                f"exceeds activation space size ({self.total_space_size})"
            )

        self.rotation = ParametrisedRotation(self.total_space_size)

    def _determine_space_sizes(self) -> list[int]:
        """Run the model to determine the sizes of the activation spaces"""

        if self.verbosity > 0:
            print("Running model to determine activation space size...")

        sizes = []

        def counter_hook(value, hook: HookPoint, sizes):
            sizes.append(value.shape[1])

        partial_counter_hook = partial(counter_hook, sizes=sizes)

        x = torch.zeros(1, self.hooked_model.cfg.input_size)
        fwd_hooks = [
            (name, partial_counter_hook) for name in self.intervene_model_hooks
        ]

        self.hooked_model.run_with_hooks(x, fwd_hooks=fwd_hooks)

        return sizes

    def get_distributed_interchange_intervention_hooks(
        self, source_inputs: torch.Tensor
    ) -> list[tuple[str, callable]]:
        """Get hooks for distributed interchange intervention

        Runs the model on each source input, and on input `i` records the
        value of the projection of the rotated activation space onto the `i`th
        subspace. It then builds hooks to patch in these values into the
        rotated activation space.

        Parameters
        ----------
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

        def store_activation_value(value, hook: HookPoint, activation_values, index):
            activation_values[:, index : index + value.shape[1]] = value

        activation_values = torch.empty(len(source_inputs), self.total_space_size)

        # Hooks to store the activation values
        fwd_hooks = []
        for i, name in enumerate(self.intervene_model_hooks):
            fwd_hooks.append(
                (
                    name,
                    partial(
                        store_activation_value,
                        activation_values=activation_values,
                        index=sum(self.space_sizes[:i]),
                    ),
                )
            )

        if self.verbosity > 1:
            print("Running model to determine activation values on source inputs...")

        with torch.no_grad():
            self.hooked_model.run_with_hooks(source_inputs, fwd_hooks=fwd_hooks)

        rotated_activation_values = self.rotation(activation_values)

        # Select the activation values which are relevant, for patching in
        indices_list = []
        indices_list.append(
            torch.zeros(self.total_space_size - sum(self.subspaces_sizes))
        )
        for i, subspace_size in enumerate(self.subspaces_sizes):
            indices_list.append(torch.ones(subspace_size) * i)
        selection_indices = torch.cat(indices_list).long()
        selected_rotated_activation_values = rotated_activation_values[
            selection_indices, torch.arange(self.total_space_size)
        ]

        def intervention_hook(
            value,
            hook: HookPoint,
            variable_alignment,
            selected_rotated_activation_values,
            space_index,
            selection_indices,
        ):
            # Pad the value with zeros, to put in the right place
            pad_left = space_index
            pad_right = variable_alignment.total_space_size - value.shape[1] - pad_left
            value = F.pad(value, (pad_left, pad_right))

            value.requires_grad = False

            # Rotate the value
            rotated_base_activation_values = variable_alignment.rotation(value)

            # Patch in the values in rotated activation space
            rotated_base_activation_values[
                selection_indices, torch.arange(self.total_space_size)
            ] = selected_rotated_activation_values

            # Rotate back
            base_activation_values = variable_alignment.rotation.inverse(
                rotated_base_activation_values
            )

        fwd_hooks = []
        for i, name in enumerate(self.intervene_model_hooks):
            fwd_hooks.append(
                (
                    name,
                    partial(
                        intervention_hook,
                        variable_alignment=self,
                        selected_rotated_activation_values=selected_rotated_activation_values,
                        space_index=sum(self.space_sizes[:i]),
                        selection_indices=selection_indices,
                    ),
                )
            )

        return fwd_hooks