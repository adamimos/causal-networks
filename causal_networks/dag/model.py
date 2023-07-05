from typing import Any, Optional

import torch

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

    def add_node(
        self,
        name: str,
        module: DAGModule,
    ):
        """Add a node to the DAG

        Parameters
        ----------
        name : str
            The name of the node
        module : DAGModule
            The module to be used as the node, which computes its value based on the
            values of its parents
        """
        self.graph.add_node(name, module=module)
        self.add_module(name, module)

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

    def get_node_value(self, node: str) -> torch.Tensor:
        """Get the value of a node

        Parameters
        ----------
        node : str
            The name of the node
        """
        return self.graph.nodes[node]["module"].value

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

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass through the DAG"""

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
                self.graph.nodes[node]["module"].forward(parent_values)

        # Return the values of the output nodes
        ouput_nodes = self.get_output_nodes()
        return {node: self.graph.nodes[node]["module"].value for node in ouput_nodes}

    def visualize(
        self, display_node_info=True, display_node_values=False, cmap_name="Pastel1"
    ):
        """Draw a visualization of the DAG

        Parameters
        ----------
        display_node_info : bool, default=True
            Whether to display information about the nodes in a table
        display_node_values : bool, default=False
            Whether to display the values of the nodes in the table
        cmap_name : str, default="Pastel1"
            The name of the colormap to use
        """

        cmap = colormaps[cmap_name]

        nx.draw_kamada_kawai(self.graph, with_labels=True, node_color=cmap(1))
        plt.show()

        if display_node_info:
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
                        column_values.append(str(self.get_node_value(node)))
                    else:
                        column_values.append("-")
                table.add_row(*column_values)
            console = Console()
            console.print(table)
