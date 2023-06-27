from typing import Any, Optional
from collections import OrderedDict

import torch

import networkx as nx

import matplotlib.pyplot as plt

from rich.table import Table
from rich.console import Console

from matplotlib import colormaps


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
        sampler: Optional[callable] = None,
        validator: Optional[callable] = None,
        func: Optional[callable] = None,
        possible_values: Optional[list] = None,
    ):
        self.G.add_node(node)
        self.G.nodes[node]["value"] = None
        self.G.nodes[node]["intervened"] = False
        if sampler is not None:
            self.samplers[node] = sampler
        if validator is not None:
            self.validators[node] = validator
        if func is not None:
            self.funcs[node] = func
        if possible_values is not None:
            self.G.nodes[node]["possible_values"] = possible_values

    def add_edge(self, node1: str, node2: str):
        if node1 not in self.G or node2 not in self.G:
            raise ValueError(
                f"Nodes must exist before an edge can be created between them."
            )
        self.G.add_edge(node1, node2)

    def set_value(self, node: str, value) -> Any:
        if node in self.validators and not self.validators[node](value):
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

    def get_roots(self):
        return [node for node, degree in self.G.in_degree() if degree == 0]

    def get_leaves(self):
        return [node for node, degree in self.G.out_degree() if degree == 0]

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

    def compute_all_nodes(
        self, output_type="all_nodes", output_format="ordereddict"
    ) -> dict[str, Any]:
        """Compute the value of all nodes recursively

        Parameters
        ----------
        output_type : str
            The type of output to return. Can be one of:
                - "all_nodes" (default): all node values
                - "output_nodes": only the output nodes
                - "output_integer": the output as an integer, ranging over the
                  possible output values
                - "output_distribution": a delta distribution over the output
                  nodes
        output_format : str
            The format of the output. Can be one of:
                - "ordereddict" (default): an OrderedDict, ordered according
                  to `self.G.nodes`
                - "torch": a PyTorch tensor ordered according to
                  `self.G.nodes`
        """

        node_values = {}
        for node in nx.topological_sort(self.G):
            node_values[node] = self.compute_node(node)

        if output_type == "all_nodes":
            node_values = OrderedDict(
                [(node, node_values[node]) for node in self.G.nodes]
            )
        elif output_type in ["output_nodes", "output_integer", "output_distribution"]:
            node_values = OrderedDict(
                [(node, node_values[node]) for node in self.get_leaves()]
            )
            if output_type == "output_integer":
                for node, value in node_values.items():
                    if self.G.nodes[node]["possible_values"] is None:
                        raise ValueError(
                            f"Cannot compute output integer for node {node} "
                            f"because it has no set of possible values."
                        )
                    node_values[node] = self.G.nodes[node]["possible_values"].index(
                        value
                    )
            elif output_type == "output_distribution":
                for node, value in node_values.items():
                    if self.G.nodes[node]["possible_values"] is None:
                        raise ValueError(
                            f"Cannot compute output distribution for node {node} "
                            f"because it has no set of possible values."
                        )
                    dist = [0 for _ in self.G.nodes[node]["possible_values"]]
                    dist[self.G.nodes[node]["possible_values"].index(value)] = 1
                    node_values[node] = dist
        else:
            raise ValueError(f"Invalid output type {output_type}")

        if output_format == "ordereddict":
            return node_values
        elif output_format == "torch":
            return torch.tensor(list(node_values.values()))

        return node_values

    def set_inputs(self, inputs: dict):
        """Set the values of non-intervened input nodes"""
        input_nodes = self.get_roots()
        for node, value in inputs.items():
            if node not in input_nodes:
                raise ValueError(f"Node {node} is not an input node.")
            if not self.G.nodes[node]["intervened"]:
                self.set_value(node, value)

    def run(
        self,
        inputs: dict,
        reset=True,
        output_type="all_nodes",
        output_format="ordereddict",
    ) -> dict[str, Any]:
        """Reset the model and run with inputs

        Parameters
        ----------
        inputs : dict
            A dictionary mapping input nodes to their values
        reset : bool
            Whether to reset the model before running
        output_type : str
            The type of output to return. Can be one of:
                - "all_nodes" (default): all node values
                - "output_nodes": only the output nodes
                - "output_distribution": a delta distribution over the output
                  nodes
        output_format : str
            The format of the output. Can be one of:
                - "ordereddict" (default): an OrderedDict, ordered according
                  to `self.G.nodes`
                - "torch": a PyTorch tensor ordered according
                  to `self.G.nodes`
        """
        if reset:
            self.reset()
        self.set_inputs(inputs)
        return self.compute_all_nodes(
            output_type=output_type, output_format=output_format
        )

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
        output_type="all_nodes",
        output_format="ordereddict",
    ) -> dict[str, Any]:
        """Reset the model, intervene and run with inputs

        Parameters
        ----------
        intervention_nodes : str | list[str]
            The node(s) to intervene on
        intervention_values : Any | list[Any]
            The value(s) to set the intervened node(s) to
        inputs : dict
            A dictionary mapping input nodes to their values
        output_type : str
            The type of output to return. Can be one of:
                - "all_nodes" (default): all node values
                - "output_nodes": only the output nodes
                - "output_distribution": a delta distribution over the output
                  nodes
        output_format : str
            The format of the output. Can be one of:
                - "ordereddict" (default): an OrderedDict, ordered according
                  to `self.G.nodes`
                - "torch": a PyTorch tensor ordered according
                  to `self.G.nodes`
        """

        self.reset()
        self.intervene(intervention_nodes, intervention_values)
        self.set_inputs(inputs)
        return self.compute_all_nodes(
            output_type=output_type, output_format=output_format
        )

    def do_interchange_intervention(
        self,
        node_lists: list[list[str]],
        source_input_list: list[dict[str, Any]],
    ):
        """Do interchange intervention on a list of nodes

        Computes the value of each node in `node_lists` on each input in
        `source_input_list`, and intervenes on the model, setting the value of
        those nodes to the computed value.

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
        for node_list, inputs in zip(node_lists, source_input_list):
            intervention_nodes.extend(node_list)
            node_values = self.run(inputs, reset=True)
            for node in node_list:
                intervention_values.append(node_values[node])

        self.reset()

        # Intervene on the model, setting the value of each node to the
        # computed value under its respective input
        self.intervene(intervention_nodes, intervention_values)

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
                sampler_name = (
                    self.samplers[node].__name__ if node in self.samplers else "-"
                )
                validator_name = (
                    self.validators[node].__name__ if node in self.validators else "-"
                )
                if self.G.nodes[node]["value"] is not None:
                    value_str = str(self.G.nodes[node]["value"])
                else:
                    value_str = "-"
                table.add_row(
                    node,
                    func_name,
                    sampler_name,
                    validator_name,
                    value_str,
                )
            console = Console()
            console.print(table)


class RecurrentDeterministicDAG(DeterministicDAG):
    """A deterministic DAG with recurrent connections"""

    def __init__(self, num_streams: int = 1):
        self._num_streams = num_streams

        self.G = nx.DiGraph()
        self.funcs = {}
        self.validators = {}
        self.samplers = {}

    @property
    def num_streams(self):
        return self._num_streams

    @num_streams.setter
    def num_streams(self, new_num_streams):
        if new_num_streams < 1:
            raise ValueError("num_streams must be at least 1")
        diff = new_num_streams - self._num_streams
        for node in self.G.nodes:
            if diff > 0:
                self.G.nodes[node]["value"].extend([None for _ in range(diff)])
                self.G.nodes[node]["intervened"].extend([False for _ in range(diff)])
            elif diff < 0:
                self.G.nodes[node]["value"] = self.G.nodes[node]["value"][
                    :new_num_streams
                ]
                self.G.nodes[node]["intervened"] = self.G.nodes[node]["intervened"][
                    :new_num_streams
                ]
        self._num_streams = new_num_streams

    def add_node(
        self,
        node: str,
        sampler: Optional[callable] = None,
        validator: Optional[callable] = None,
        func: Optional[callable] = None,
        possible_values: Optional[list] = None,
    ):
        super().add_node(node, sampler, validator, func, possible_values)
        self.G.nodes[node]["value"] = [None for _ in range(self._num_streams)]
        self.G.nodes[node]["intervened"] = [False for _ in range(self._num_streams)]

    def add_edge(self, node1: str, node2: str, edge_type: str = "current_stream"):
        """Add an edge between two nodes

        Parameters
        ----------
        node1 : str
            The first node
        node2 : str
            The second node
        edge_type : str, default "current_stream"
            The type of edge to add. Can be one of:
                - "current_stream" (default): the edge exists only locally,
                  per stream
                - "all_streams": the edge exists globally, connecting the
                  first node to the instances of the second node in all
                  streams
                - "current_and_previous_streams": the edge exists globally,
                  connecting the first node to the instances of the second
                  node in the current and previous streams
        """
        if node1 not in self.G or node2 not in self.G:
            raise ValueError(
                f"Nodes must exist before an edge can be created between them."
            )
        if edge_type not in [
            "current_stream",
            "all_streams",
            "current_and_previous_streams",
        ]:
            raise ValueError(
                f"Edge type must be one of 'current_stream', 'all_streams', "
                f"or 'current_and_previous_streams', got '{edge_type}'"
            )
        self.G.add_edge(node1, node2, edge_type=edge_type)

    def set_value(self, node: str, value, stream: Optional[int] = None) -> Any:
        """Set the value of a node

        Parameters
        ----------
        node : str
            The node to set the value of
        value : Any
            The value to set the node to
        stream : Optional[int]
            The stream to set the value of. If None, the value is set for all
            streams.
        """
        if node in self.validators and not self.validators[node](value):
            raise ValueError(
                f"Invalid value {value} for node {node}. Value does not pass the validator."
            )
        if stream is None:
            self.G.nodes[node]["value"] = [value for _ in range(self._num_streams)]
        else:
            self.G.nodes[node]["value"][stream] = value
        self.G.nodes[node]["intervened"] = [False for _ in range(self._num_streams)]
        return value

    def set_value_across_streams(
        self, node: str, values: list, exclude_intervened=True
    ) -> Any:
        """Set the value of a node according to a list of stream values

        Parameters
        ----------
        node : str
            The node to set the value of
        value : list
            The values to set the node to, one for each stream
        exclude_intervened : bool, default=True
            If True, only set the value for streams that have not been intervened on
        """
        for stream, value in enumerate(values):
            if exclude_intervened and self.G.nodes[node]["intervened"][stream]:
                continue
            if node in self.validators and not self.validators[node](value):
                raise ValueError(
                    f"Invalid value {value} for node {node}. Value does not pass the validator."
                )
            self.G.nodes[node]["value"][stream] = value
            self.G.nodes[node]["intervened"][stream] = False
        return self.G.nodes[node]["value"]

    def get_value(self, node: str, stream: Optional[int] = None) -> Any:
        if stream is None:
            return self.G.nodes[node]["value"]
        else:
            return self.G.nodes[node]["value"][stream]

    def reset_value(self, node: str, stream: Optional[int] = None):
        if stream is None:
            self.G.nodes[node]["value"] = [None for _ in range(self._num_streams)]
        else:
            self.G.nodes[node]["value"][stream] = None

    def reset_interventions(self):
        for node in self.G.nodes:
            self.G.nodes[node]["intervened"] = [False for _ in range(self._num_streams)]

    def compute_node(self, node: str, stream: Optional[int] = None) -> Any:
        """Compute the value of a single node"""

        if stream is None:
            streams = range(self._num_streams)
        else:
            streams = [stream]

        for curr_stream in streams:
            # Do not recalculate the value of a node that has been intervened on
            if self.G.nodes[node]["intervened"][curr_stream]:
                continue

            # If the node doesn't have a function, just return its value
            if node not in self.funcs:
                continue

            # If the node has a function, compute its value from its parents
            parents = list(self.G.predecessors(node))
            if len(parents) > 0:
                args = {}
                for parent in parents:
                    if self.G[parent][node]["edge_type"] == "current_stream":
                        args[parent] = self.G.nodes[parent]["value"][curr_stream]
                    elif self.G[parent][node]["edge_type"] == "all_streams":
                        args[parent] = self.G.nodes[parent]["value"]
                    elif (
                        self.G[parent][node]["edge_type"]
                        == "current_and_previous_streams"
                    ):
                        args[parent] = self.G.nodes[parent]["value"][: curr_stream + 1]
                self.G.nodes[node]["value"][curr_stream] = self.funcs[node](**args)
            else:
                self.G.nodes[node]["value"][curr_stream] = self.funcs[node]()

        if stream is None:
            return self.G.nodes[node]["value"]
        else:
            return self.G.nodes[node]["value"][stream]

    def compute_all_nodes(
        self, output_type="all_nodes", output_format="ordereddict"
    ) -> dict[str, Any]:
        """Compute the value of all nodes recursively

        Parameters
        ----------
        output_type : str
            The type of output to return. Can be one of:
                - "all_nodes" (default): all node values
                - "output_nodes": only the output nodes
                - "output_integer": the output as an integer, ranging over the
                  possible output values
                - "output_distribution": a delta distribution over the output
                  nodes
        output_format : str
            The format of the output. Can be one of:
                - "ordereddict" (default): an OrderedDict, ordered according
                  to `self.G.nodes`
                - "torch": a PyTorch tensor ordered according to
                  `self.G.nodes`
        """

        node_values = super().compute_all_nodes(
            output_type="all_nodes", output_format="ordereddict"
        )

        if output_type in ["output_nodes", "output_integer", "output_distribution"]:
            node_values = OrderedDict(
                [(node, node_values[node]) for node in self.get_leaves()]
            )
            if output_type == "output_integer":
                for node, values in node_values.items():
                    if self.G.nodes[node]["possible_values"] is None:
                        raise ValueError(
                            f"Cannot compute output integer for node {node} "
                            f"because it has no set of possible values."
                        )
                    node_values[node] = []
                    for value in values:
                        node_values[node].append(
                            self.G.nodes[node]["possible_values"].index(value)
                        )
            elif output_type == "output_distribution":
                for node, values in node_values.items():
                    if self.G.nodes[node]["possible_values"] is None:
                        raise ValueError(
                            f"Cannot compute output distribution for node {node} "
                            f"because it has no set of possible values."
                        )
                    node_values[node] = []
                    for value in values:
                        dist = [0 for _ in self.G.nodes[node]["possible_values"]]
                        dist[self.G.nodes[node]["possible_values"].index(value)] = 1
                        node_values[node].append(dist)
        elif output_type != "all_nodes":
            raise ValueError(f"Invalid output type {output_type}")

        if output_format == "ordereddict":
            return node_values
        elif output_format == "torch":
            return torch.tensor(list(node_values.values()))

        return node_values

    def set_inputs(self, inputs: dict[str, list]):
        """Set the values of non-intervened input nodes across streams"""
        input_nodes = self.get_roots()
        for node, values in inputs.items():
            if node not in input_nodes:
                raise ValueError(f"Node {node} is not an input node.")
            self.set_value_across_streams(node, values)

    def run(
        self,
        inputs: dict[str, list[Any]],
        reset=True,
        output_type="all_nodes",
        output_format="ordereddict",
    ) -> dict[str, Any]:
        """Reset the model and run with inputs

        Parameters
        ----------
        inputs : dict[str, list[Any]]
            A dictionary mapping input nodes to their list of values (one for
            each stream).
        reset : bool
            Whether to reset the model before running
        output_type : str
            The type of output to return. Can be one of:
                - "all_nodes" (default): all node values
                - "output_nodes": only the output nodes
                - "output_distribution": a delta distribution over the output
                  nodes
        output_format : str
            The format of the output. Can be one of:
                - "ordereddict" (default): an OrderedDict, ordered according
                  to `self.G.nodes`
                - "torch": a PyTorch tensor ordered according to
                  `self.G.nodes`
        """
        for node, values in inputs.items():
            if not isinstance(values, list) or len(values) != self._num_streams:
                raise ValueError(
                    f"Input values for node {node} must be a list of length {self._num_streams}"
                )
        return super().run(
            inputs, reset=reset, output_type=output_type, output_format=output_format
        )

    def intervene(
        self,
        intervention_nodes_streams: str | tuple[str, int] | list[str | tuple[str, int]],
        intervention_values: Any | list,
    ):
        """Intervene on a list of nodes at specific streams"""

        if not isinstance(intervention_nodes_streams, list):
            intervention_nodes_streams = [intervention_nodes_streams]
        if not isinstance(intervention_values, list):
            intervention_values = [intervention_values] * len(
                intervention_nodes_streams
            )

        if len(intervention_nodes_streams) != len(intervention_values):
            raise ValueError(
                f"Number of intervention nodes and values must be equal. "
                f"Got {len(intervention_nodes_streams)} nodes and {len(intervention_values)} values."
            )

        # Intervene on each node, marking it as intervened and setting its value
        for item, value in zip(intervention_nodes_streams, intervention_values):
            if isinstance(item, str):
                for val in value:
                    if node in self.validators and not self.validators[node](val):
                        raise ValueError(f"Invalid value {val} for node {node}")
                if len(value) != self._num_streams:
                    raise ValueError(
                        f"Intervention value for node {item} must be a "
                        f"list of length {self._num_streams}"
                    )
                self.G.nodes[item]["value"] = value
                self.G.nodes[item]["intervened"] = [
                    True for _ in range(self._num_streams)
                ]
            elif isinstance(item, tuple):
                node, stream = item
                if node in self.validators and not self.validators[node](value):
                    raise ValueError(f"Invalid value {value} for node {node}")
                self.G.nodes[node]["value"][stream] = value
                self.G.nodes[node]["intervened"][stream] = True
            else:
                raise ValueError(
                    f"Invalid intervention node {item}. Must be a string or a tuple."
                )

    def intervene_and_run(
        self,
        intervention_nodes_streams: str | tuple[str, int] | list[str | tuple[str, int]],
        intervention_values: Any | list[Any],
        inputs: dict,
        output_type="all_nodes",
        output_format="ordereddict",
    ) -> dict[str, Any]:
        """Reset the model, intervene and run with inputs

        Parameters
        ----------
        intervention_nodes : str | tuple[str, int] | list[str | tuple[str,
        int]]
            The nodes and stream positions to intervene on. Nodes can be
            specified either as strings or as tuples of the form (node,
            stream). In the former case, the corresponding value must be a
            list of values, one for each stream.
        intervention_values : Any | list[Any]
            The values to set the intervened nodes to
        inputs : dict
            A dictionary mapping input nodes to their values
        output_type : str
            The type of output to return. Can be one of:
                - "all_nodes" (default): all node values
                - "output_nodes": only the output nodes
                - "output_distribution": a delta distribution over the output
                  nodes
        output_format : str
            The format of the output. Can be one of:
                - "ordereddict" (default): an OrderedDict, ordered according
                  to `self.G.nodes`
                - "torch": a PyTorch tensor ordered according to
                  `self.G.nodes`
        """
        super().intervene_and_run(
            intervention_nodes=intervention_nodes_streams,
            intervention_values=intervention_values,
            inputs=inputs,
            output_type=output_type,
            output_format=output_format,
        )

    def do_interchange_intervention(
        self,
        node_lists: list[list[str | tuple[str, int]]],
        source_input_list: list[dict[str, Any]],
    ):
        """Do interchange intervention on a list of nodes

        Computes the value of each node and stream in `node_lists` on each input in
        `source_input_list`, and intervenes on the model, setting the value of
        those nodes to the computed value.

        Note that this function first resets the model.
        """

        # Replace all single nodes with tuples of the form (node, stream), one
        # for each stream
        new_node_lists = []
        for i in range(len(node_lists)):
            new_node_list = []
            for j in range(len(node_lists[i])):
                if isinstance(node_lists[i][j], str):
                    new_node_list.extend(
                        [(node_lists[i][j], stream) for stream in range(self._num_streams)]
                    )
                elif isinstance(node_lists[i][j], tuple):
                    new_node_list.append(node_lists[i][j])
                else:
                    raise ValueError(
                        f"Invalid node {node_lists[i][j]}. Must be a string or a tuple."
                    )
            new_node_lists.append(new_node_list)
        node_lists = new_node_lists

        # Make sure `node_lists` is a list of disjoint lists
        for i in range(len(node_lists)):
            for j in range(len(node_lists)):
                if i == j:
                    continue
                for node in node_lists[i]:
                    if node in node_lists[j]:
                        raise ValueError(
                            f"Expected `intervention_node_lists` to be a list of disjoint lists"
                        )

        # Get the values of each set of nodes on each input
        intervention_nodes = []
        intervention_values = []
        for node_list, inputs in zip(node_lists, source_input_list):
            node_values = self.run(inputs, reset=True)
            for node, stream in node_list:
                intervention_nodes.append((node, stream))
                intervention_values.append(node_values[node][stream])

        self.reset()

        # Intervene on the model, setting the value of each node to the
        # computed value under its respective input
        self.intervene(intervention_nodes, intervention_values)

    def create_full_graph(self):
        full_graph = nx.DiGraph()
        for node in self.G.nodes:
            for stream in range(self._num_streams):
                full_graph.add_node(
                    f"{node}:{stream}",
                    original_node=node,
                    stream=stream,
                    value=self.G.nodes[node]["value"][stream],
                    intervened=self.G.nodes[node]["intervened"][stream],
                )
        for node1, node2 in self.G.edges:
            if self.G[node1][node2]["edge_type"] == "current_stream":
                for stream in range(self._num_streams):
                    full_graph.add_edge(f"{node1}:{stream}", f"{node2}:{stream}")
            elif self.G[node1][node2]["edge_type"] == "all_streams":
                for stream1 in range(self._num_streams):
                    for stream2 in range(self._num_streams):
                        full_graph.add_edge(f"{node1}:{stream1}", f"{node2}:{stream2}")
            elif self.G[node1][node2]["edge_type"] == "current_and_previous_streams":
                for stream2 in range(self._num_streams):
                    for stream1 in range(stream2 + 1):
                        full_graph.add_edge(f"{node1}:{stream1}", f"{node2}:{stream2}")
        return full_graph

    def visualize(self, display_node_info=True, cmap_name="Pastel1"):
        cmap = colormaps[cmap_name]

        full_graph = self.create_full_graph()

        def get_node_colour(node, full_graph):
            if full_graph.nodes[node]["intervened"]:
                return cmap(0)
            else:
                return cmap(1)

        colours = [get_node_colour(node, full_graph) for node in full_graph.nodes]

        nx.draw(
            full_graph,
            with_labels=True,
            node_color=colours,
            pos=nx.multipartite_layout(full_graph, subset_key="stream"),
        )

        plt.show()
        if display_node_info:
            table = Table(
                title="Node Information", show_header=True, header_style="bold"
            )
            table.add_column("Node")
            table.add_column("Function")
            for stream in range(self._num_streams):
                table.add_column(f"Value {stream}")
            for node in self.G.nodes:
                func_name = self.funcs[node].__name__ if node in self.funcs else "-"
                sampler_name = (
                    self.samplers[node].__name__ if node in self.samplers else "-"
                )
                validator_name = (
                    self.validators[node].__name__ if node in self.validators else "-"
                )
                value_strs = []
                for stream in range(self._num_streams):
                    if self.G.nodes[node]["value"][stream] is not None:
                        value_strs.append(str(self.G.nodes[node]["value"][stream]))
                    else:
                        value_strs.append("-")
                table.add_row(
                    node,
                    func_name,
                    *value_strs,
                )
            console = Console()
            console.print(table)
