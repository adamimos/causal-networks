from typing import Any, Optional, Union
from functools import partial

import torch
from torch import Tensor

from jaxtyping import Float, Bool

import networkx as nx

import matplotlib.pyplot as plt

from rich.table import Table
from rich.console import Console

from matplotlib import colormaps

from transformer_lens.hook_points import HookedRootModule, HookPoint

from .module import DAGModule


class DAGModel(HookedRootModule):
    """A deterministic computational DAG"""

    def __init__(self):
        super().__init__()
        self.graph = nx.DiGraph()
        self.visualization_params = dict(
            layout=nx.kamada_kawai_layout, canvas_size=None, cmap_name="Pastel1"
        )
        self.device = torch.device("cpu")

    def add_node(
        self,
        name: str,
        module: DAGModule,
        **node_attributes: Any,
    ):
        """Add a node to the DAG

        Runs the `setup` method each time after adding the node, to update the list of
        hooks.

        Parameters
        ----------
        name : str
            The name of the node
        module : DAGModule
            The module to be used as the node, which computes its value based on the
            values of its parents
        **node_attributes : Any
            Any additional attributes to be added to the node
        """
        hook_point = HookPoint()
        self.graph.add_node(
            name, module=module, hook_point=hook_point, **node_attributes
        )
        self.add_module(name, module)
        self.add_module(f"hook_{name}", hook_point)
        self.setup()

    def add_edge(self, node1: str, node2: str):
        """Add an edge to the DAG

        Parameters
        ----------
        node1 : str
            The name of the first node
        node2 : str
            The name of the second node
        """
        if node1 not in self.graph or node2 not in self.graph:
            raise ValueError(
                f"Nodes must exist before an edge can be created between them."
            )
        self.graph.add_edge(node1, node2)

    def get_node_value(self, node: str) -> Tensor:
        """Get the value of a node

        Parameters
        ----------
        node : str
            The name of the node

        Returns
        -------
        value : Tensor
            The value of the node
        """
        return self.graph.nodes[node]["module"].value

    def get_node_display_value(self, node: str) -> str:
        """Get a human-readable representation of the value of a node

        Parameters
        ----------
        node : str
            The name of the node

        Returns
        -------
        display_value : str
            A human-readable representation of the value of the node
        """
        return self.graph.nodes[node]["module"].get_display_value()

    def get_input_nodes(self) -> list[str]:
        """Get the input nodes

        Returns
        -------
        input_nodes : list[str]
            The names of the input nodes
        """
        return [node for node in self.graph.nodes if self.graph.in_degree(node) == 0]

    def get_output_nodes(self) -> list[str]:
        """Get the output nodes

        Returns
        -------
        output_nodes : list[str]
            The names of the output nodes
        """
        return [node for node in self.graph.nodes if self.graph.out_degree(node) == 0]

    def forward(
        self,
        inputs: dict[str, Tensor],
        output_type="output_nodes",
        return_integers=False,
    ) -> dict[str, Tensor]:
        """Forward pass through the DAG

        Parameters
        ----------
        inputs : dict[str, Tensor]
            The input values for the input nodes. The values can be batched.
        output_type : str, default="output_nodes"
            The type of output to return. Can be one of:
                - "output_nodes": only the output nodes
                - "all_nodes": all node values
        return_integers : bool, default=False
            Whether to return the values as integers. If True, the values are converted
            to integers where possible. Where this is not possible a RuntimeError is
            raised.

        Returns
        -------
        outputs : dict[str, Tensor]
            The values for the output nodes. Batched in the same way as the inputs.
        """

        # Compute the values of the nodes
        for node in nx.topological_sort(self.graph):
            # For input nodes, take the value from `inputs`
            if self.graph.in_degree(node) == 0:
                if node not in inputs:
                    raise ValueError(f"Missing input for node {node}")
                self.graph.nodes[node]["module"].value = inputs[node]

            # For other nodes, compute the value based on the value of the parents
            else:
                parents = self.graph.predecessors(node)
                parent_values = {
                    parent: self.graph.nodes[parent]["module"].value
                    for parent in parents
                }
                self.graph.nodes[node]["module"].forward(
                    parent_values, hook_point=self.graph.nodes[node]["hook_point"]
                )

        # Determine the requested output nodes
        if output_type == "output_nodes":
            requested_nodes = self.get_output_nodes()
        elif output_type == "all_nodes":
            requested_nodes = self.graph.nodes
        else:
            raise ValueError(f"Unknown output type {output_type}")

        # Return the requested output node values, converting to integers if requested
        if return_integers:
            requested_values = {}
            for node in requested_nodes:
                module = self.graph.nodes[node]["module"]
                try:
                    requested_values[node] = module.get_value_as_integer()
                except NotImplementedError:
                    raise RuntimeError(
                        f"Cannot convert value of node {node} to integer"
                    )
            return requested_values
        else:
            return {
                node: self.graph.nodes[node]["module"].value for node in requested_nodes
            }

    def run_interchange_intervention(
        self,
        base_input: dict[str, Float[Tensor, "batch_size ..."]],
        source_inputs: dict[str, Float[Tensor, "batch_size num_source_inputs ..."]],
        intervention_mask: dict[str, Bool[Tensor, "batch_size num_source_inputs ..."]],
        output_type="output_nodes",
        return_integers=False,
    ) -> dict[str, Tensor]:
        """Run interchange intervention with base and source inputs

        Runs the forward pass with the source inputs, recording the values of the nodes.
        Then runs the forward pass with the base inputs, replacing the values of the
        nodes with the recorded values according to the intervention mask.

        All inputs must be batched. The source inputs and intervention mask must have a
        second batch dimension for the number of source inputs.

        See below for how the intervention mask is broadcasted to match the values. This
        can allow for more fine-grained intervention than just per node.

        Parameters
        ----------
        base_input : dict[str, Tensor of shape (batch_size, ...)]
            The base input values for the input nodes
        source_inputs : dict[str, Tensor of shape (batch_size, num_source_inputs,
        ...)]
            The source input values for the input nodes
        intervention_mask : dict[str, torch.BoolTensor of shape (batch_size,
        num_source_inputs, ...)]
            The intervention mask for the nodes. Any nodes not appearing as keys will
            not be intervened on. For each node and source input `i`, the mask value at
            `[:, i, ...]` will have dimensions added to match the value, and will then
            be broadcasted to match the value (i.e. we do left-to-right broadcasting).
            This means that a 2D mask will allow intervening per node, while
            higher-dimensional masks will allow more fine-grained intervention.
        output_type : str, default="output_nodes"
            The type of output to return. Can be one of:
                - "output_nodes": only the output nodes
                - "all_nodes": all node values
        return_integers : bool, default=False
            Whether to return the values as integers. If True, the values are converted
            to integers where possible. Where this is not possible a RuntimeError is
            raised.

        Returns
        -------
        outputs : dict[str, Tensor of shape (batch_size, ...)]
            The values for the nodes. Batched in the same way as the inputs.
        """

        source_values = self.forward(source_inputs, output_type="all_nodes")

        # An intervention hook which replaces the value with the source value according
        # to the intervention mask
        def intervention_hook(
            value: Tensor,
            hook: HookPoint,
            node: str,
            source_values: dict[str, Tensor],
            intervention_mask: dict[str, Tensor],
        ):
            # If the node is not in the intervention mask, don't intervene
            if node not in intervention_mask:
                return value

            # Add trailing dimensions to the intervention mask to match the value plus
            # the source input dimension
            node_intervention_mask = intervention_mask[node]
            node_intervention_mask = node_intervention_mask.view(
                *node_intervention_mask.shape,
                *[1] * (1 + value.ndim - node_intervention_mask.ndim),
            )

            # For each source input, replace the value with the source value according
            # to the broadcasted intervention mask
            for i in range(source_values[node].shape[1]):
                value = torch.where(
                    intervention_mask[node][:, i, ...],
                    source_values[node][:, i, ...],
                    value,
                )

            return value

        # Register the intervention hooks
        fwd_hooks = []
        for node in self.graph:
            fwd_hooks.append(
                (
                    f"hook_{node}",
                    partial(
                        intervention_hook,
                        node=node,
                        source_values=source_values,
                        intervention_mask=intervention_mask,
                    ),
                )
            )

        # Run the forward pass with the base input and the intervention hooks
        output = self.run_with_hooks(
            base_input,
            fwd_hooks=fwd_hooks,
            output_type=output_type,
            return_integers=return_integers,
        )

        return output

    def set_visualization_params(
        self,
        layout: Optional[callable] = None,
        canvas_size: Optional[tuple[int, int]] = None,
        cmap_name: Optional[str] = None,
    ):
        """Set parameters for visualizing the DAG

        Parameters
        ----------
        layout : callable, optional
            The layout function to use for drawing the graph. It is recommended to use a
            function from `networkx.drawing.layout` (possibly wrapped in a partial). By
            default uses the Kamada-Kawai path-length cost-function
        canvas_size : tuple[int, int], optional
            The size of the canvas to draw the graph on. By default uses the default
            networkx canvas size.
        cmap_name : str, optional
            The name of the colormap to use for the node colors. By default uses
            "pastel1".
        """
        if layout is not None:
            self.visualization_params["layout"] = layout
        if canvas_size is not None:
            self.visualization_params["canvas_size"] = canvas_size
        if cmap_name is not None:
            self.visualization_params["cmap_name"] = cmap_name

    def visualize(
        self,
        display_node_table=True,
        display_node_values=False,
        use_human_readable_values=True,
        visualization_params: Optional[dict] = None,
    ):
        """Draw a visualization of the DAG

        Uses

        Parameters
        ----------
        display_node_table : bool, default=True
            Whether to display information about the nodes in a table
        display_node_values : bool, default=False
            Whether to display the values of the nodes in the table
        use_human_readable_values : bool, default=True
            Whether to use human-readable values for the nodes. If False, uses the raw
            values.
        visualization_params : dict, optional
            Parameters used to control the visualization. If None, uses the values of
            `self.visualization_params`. See `self.set_visualization_params` for
            details.
        """

        if visualization_params is None:
            visualization_params = self.visualization_params
        else:
            for param, value in self.visualization_params.items():
                if param not in visualization_params:
                    visualization_params[param] = value

        cmap = colormaps[visualization_params["cmap_name"]]

        plt.figure(figsize=visualization_params["canvas_size"])
        ax = plt.gca()
        nx.draw(
            self.graph,
            pos=visualization_params["layout"](self.graph),
            ax=ax,
            with_labels=True,
            node_color=[cmap(1)] * len(self.graph.nodes),
        )
        plt.show()

        if display_node_table:
            table = Table(
                title="Node Information", show_header=True, header_style="bold"
            )
            table.add_column("Node")
            table.add_column("Module")
            if display_node_values:
                table.add_column("Value")
            for node in self.graph.nodes:
                column_values = [node, str(self.graph.nodes[node]["module"])]
                if display_node_values:
                    if self.get_node_value(node) is not None:
                        if use_human_readable_values:
                            column_values.append(self.get_node_display_value(node))
                        else:
                            column_values.append(str(self.get_node_value(node)))
                    else:
                        column_values.append("-")
                table.add_row(*column_values)
            console = Console()
            console.print(table)

    def to(self, device: Union[str, torch.device]):
        """Move the model to a device

        Parameters
        ----------
        device : str or torch.device
            The device to move the model to
        """
        self.device = device
        for node in self.graph.nodes:
            self.graph.nodes[node]["module"].to(device)
        return self