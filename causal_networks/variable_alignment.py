from functools import partial
from dataclasses import dataclass

import torch
from torch import nn

from transformer_lens.hook_points import HookedRootModule, HookPoint

from .dag import DeterministicDAG


class ParametrisedRotation(nn.Module):
    """A parametrised rotation matrix"""

    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.theta = nn.Linear(size, size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.theta


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

        self.space_size = self._determine_space_size()

        if sum(self.subspaces_sizes) > self.space_size:
            raise ValueError(
                f"Sum of subspace sizes ({sum(self.subspaces_sizes)}) "
                f"exceeds activation space size ({self.space_size})"
            )

        self.rotation = ParametrisedRotation(self.space_size)

    def _determine_space_size(self) -> int:
        """Run the model to determine the size of the activation space"""

        print("Running model to determine activation space size...")

        # A singleton class to hold the size, so that it can be by the hooks
        @dataclass
        class Size:
            size = 0
        size = Size()

        def counter_hook(value, hook: HookPoint, size):
            size.size += value.shape[1]

        partial_counter_hook = partial(counter_hook, size=size)

        x = torch.zeros(1, self.hooked_model.cfg.input_size)
        fwd_hooks = [
            (name, partial_counter_hook) for name in self.intervene_model_hooks
        ]

        self.hooked_model.run_with_hooks(x, fwd_hooks=fwd_hooks)

        return size.size
