from typing import Any, Optional
from collections import defaultdict

import networkx as nx
import matplotlib.pyplot as plt
from rich.table import Table
from rich.console import Console

from matplotlib import colormaps

import numpy as np


class DeterministicDAG:
    """A class for representing a deterministic DAG."""

    def __init__(self):
        self.G = nx.DiGraph()
        self.funcs = {}
        self.validators = {}
        self.samplers = {}

    def add_node(
        self,
        node: str,
        sampler: callable,
        validator: callable,
        func: Optional[callable] = None,
    ):
        self.G.add_node(node)
        self.G.nodes[node]["value"] = None
        self.G.nodes[node]["intervened"] = False
        if func is not None:
            self.funcs[node] = func
        self.samplers[node] = sampler
        self.validators[node] = validator

    def add_edge(self, node1: str, node2: str):
        if node1 not in self.G or node2 not in self.G:
            raise ValueError(
                f"Nodes must exist before an edge can be created between them."
            )
        self.G.add_edge(node1, node2)

    def set_value(self, node: str, value) -> Any:
        if not self.validators[node](value):
            raise ValueError(
                f"Invalid value {value} for node {node}. Value does not pass the validator."
            )
        self.G.nodes[node]["value"] = value
        self.G.nodes[node]["intervened"] = False
        return value

    def get_value(self, node: str) -> Any:
        return self.G.nodes[node]["value"]
    
    def reset_value(self, node: str):
        self.G.nodes[node]["value"] = None

    def reset_all_values(self):
        for node in self.G.nodes:
            self.reset_value(node)

    def reset_interventions(self):
        for node in self.G.nodes:
            self.G.nodes[node]["intervened"] = False

    def reset(self):
        self.reset_all_values()
        self.reset_interventions()

    def compute_node(self, node: str) -> Any:
        """Compute the value of a single node"""

        # Do not recalculate the value of a node that has been intervened on
        if self.G.nodes[node]["intervened"]:
            return self.get_value(node)

        # If the node doesn't have a function, just return its value
        if node not in self.funcs:
            return self.get_value(node)

        # If the node has a function, compute its value from its parents
        if node in self.funcs:
            parents = list(self.G.predecessors(node))
            parent_values = [self.get_value(p) for p in parents] if parents else None
            if parent_values is None:
                return self.set_value(node, None)
            return self.set_value(node, self.funcs[node](*parent_values))

    def compute_all_nodes(self) -> dict[str, Any]:
        """Compute the value of all nodes recursively"""
        node_values = {}
        for node in nx.topological_sort(self.G):
            node_values[node] = self.compute_node(node)
        return node_values

    def set_inputs(self, inputs: dict):
        """Set the values of non-intervened input nodes"""
        input_nodes = self.get_roots()
        for node, value in inputs.items():
            if node not in input_nodes:
                raise ValueError(f"Node {node} is not an input node.")
            if not self.G.nodes[node]["intervened"]:
                self.set_value(node, value)

    def run(self, inputs: dict, reset=True) -> dict[str, Any]:
        """Reset the model and run with inputs"""
        if reset:
            self.reset()
        self.set_inputs(inputs)
        return self.compute_all_nodes()

    def intervene(
        self,
        intervention_nodes: str | list[str],
        intervention_values: Any | list[Any],
    ):
        """Intervene on a list of nodes"""

        if not isinstance(intervention_nodes, list):
            intervention_nodes = [intervention_nodes]
        if not isinstance(intervention_values, list):
            intervention_values = [intervention_values] * len(intervention_nodes)

        if len(intervention_nodes) != len(intervention_values):
            raise ValueError(
                f"Number of intervention nodes and values must be equal. "
                f"Got {len(intervention_nodes)} nodes and {len(intervention_values)} values."
            )
        
        # Intervene on each node, marking it as intervened and setting its value
        for node, value in zip(intervention_nodes, intervention_values):
            if not self.validators[node](value):
                raise ValueError(f"Invalid value {value} for node {node}")
            self.G.nodes[node]["value"] = value
            self.G.nodes[node]["intervened"] = True

    def intervene_and_run(
        self,
        intervention_nodes: str | list[str],
        intervention_values: Any | list[Any],
        inputs: dict,
    ) -> dict[str, Any]:
        """Reset the model, intervene and run with inputs"""

        self.reset()
        self.intervene(intervention_nodes, intervention_values)
        self.set_inputs(inputs)
        return self.compute_all_nodes()

    def do_interchange_intervention(
        self,
        node_lists: list[list[str]],
        input_list: list[dict[str, Any]],
    ):
        """Do interchange intervention on a list of nodes

        Computes the value of each node in `node_lists` on each input in
        `input_list`, and intervenes on the model, setting the value of those
        nodes to the computed value.

        Note that this function first resets the model.
        """

        # Make sure `intervention_node_lists` is a list of disjoint lists
        for i in range(len(node_lists)):
            for j in range(len(node_lists)):
                if i == j:
                    continue
                for node in node_lists[i]:
                    if node in node_lists[j]:
                        raise ValueError(
                            f"Expected `intervention_node_lists` to be a list of disjoint"
                            f" lists of nodes, but list item {i} intersects with list item {j}"
                        )

        # Get the values of each set of nodes on each input
        intervention_nodes = []
        intervention_values = []
        for node_list, inputs in zip(node_lists, input_list):
            intervention_nodes.extend(node_list)
            node_values = self.run(inputs, reset=True)
            for node in node_list:
                intervention_values.append(node_values[node])
                    
        self.reset()

        # Intervene on the model, setting the value of each node to the
        # computed value under its respective input
        self.intervene(intervention_nodes, intervention_values)

    def get_roots(self):
        return [node for node, degree in self.G.in_degree() if degree == 0]
    
    def get_leaves(self):
        return [node for node, degree in self.G.out_degree() if degree == 0]

    def visualize(self, display_node_info=True, cmap_name="Pastel1"):
        cmap = colormaps[cmap_name]

        def get_node_colour(node):
            if self.G.nodes[node]["intervened"]:
                return cmap(0)
            else:
                return cmap(1)

        colours = list(map(get_node_colour, self.G.nodes))
        nx.draw_kamada_kawai(self.G, with_labels=True, node_color=colours)
        plt.show()
        if display_node_info:
            table = Table(
                title="Node Information", show_header=True, header_style="bold"
            )
            table.add_column("Node")
            table.add_column("Function")
            table.add_column("Sampler")
            table.add_column("Validator")
            table.add_column("Value")
            for node in self.G.nodes:
                func_name = self.funcs[node].__name__ if node in self.funcs else "-"
                if self.G.nodes[node]["value"] is not None:
                    value_str = str(self.G.nodes[node]["value"])
                else:
                    value_str = "-"
                table.add_row(
                    node,
                    func_name,
                    self.samplers[node].__name__,
                    self.validators[node].__name__,
                    value_str,
                )
            console = Console()
            console.print(table)

    def create_interchange_intervention_dataset(
            self,
            num_samples: int = 10,
            max_source_inputs: int = 1,
            ):
        
        """
        Create a dataset for interchange intervention

        The base_input is the setting of the inputs for the base run
        The source_inputs are the settings of the inputs for the interventions
        The gold_output is the output of the model for the interchange interventions run with the base_input
        """

        root_nodes = self.get_roots() # these are the input nodes
        leaf_nodes = self.get_leaves() # this is the output node(s)

        # the possible source nodes are those that are neither root nor leaf nodes
        possible_source_nodes = [node for node in self.G.nodes if node not in root_nodes and node not in leaf_nodes]
        # print(f"Possible source nodes: {possible_source_nodes}\nthere are {len(possible_source_nodes)} of them")

        dataset = []
        for i in range(num_samples):
                        
            # sample from the root nodes, in order to get the base input
            base_input = {node: self.samplers[node]() for node in root_nodes}
            num_source_inputs = np.random.randint(1, max_source_inputs+1)

            # randomly choose the source input variables, these will be variables to intervene on
            # QUESTION: should we allow for multiple interventions simultaneously? unclear from the paper
            source_variables = np.random.choice(possible_source_nodes, num_source_inputs, replace=False)

            # for each, choose a setting of the input
            source_inputs = {
                node: {
                    root_node: self.samplers[root_node]() for root_node in root_nodes
                }
                for node in source_variables
            }

            # do the interchange intervention
            self.do_interchange_intervention(node_lists=[[s] for s in source_variables],
                                                          input_list=[source_inputs[s] for s in source_variables])
            gold_output = self.run(inputs=base_input, reset=False)

            # TODO: if there's more than one output this needs to change
            gold_output = gold_output[leaf_nodes[0]] # we only care about the first leaf node

            dataset.append({'base': base_input, 'source': source_inputs, 'gold': gold_output})

        return dataset




            