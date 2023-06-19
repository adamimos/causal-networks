import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

class DeterministicDAG:
    def __init__(self):
        self.G = nx.DiGraph()
        self.funcs = defaultdict(lambda: lambda x: x)
        self.validators = defaultdict(lambda: lambda x: True)
        self.samplers = defaultdict(lambda: None)

    def add_node(self, node, func=None, sampler=None, validator=None):
        if sampler is None or validator is None:
            raise ValueError("Both a validator and a sampler must be provided when adding a node.")
        self.G.add_node(node)
        self.G.nodes[node]['intervened'] = False
        if func is not None:
            self.funcs[node] = func
        self.samplers[node] = sampler
        self.validators[node] = validator

    def add_edge(self, node1, node2):
        if node1 not in self.G or node2 not in self.G:
            raise ValueError(f"Nodes must exist before an edge can be created between them.")
        self.G.add_edge(node1, node2)

    def set_value(self, node, value):
        if not self.validators[node](value):
            raise ValueError(f"Invalid value {value} for node {node}. Value does not pass the validator.")
        self.G.nodes[node]['value'] = value
        self.G.nodes[node]['intervened'] = False

    def get_value(self, node):
        return self.G.nodes[node]['value']

    def compute_node(self, node):
        if self.G.nodes[node]['intervened']:
            return  # Do not recalculate the value of a node that has been intervened on
        if node in self.funcs:
            parents = list(self.G.predecessors(node))
            parent_values = [self.get_value(p) for p in parents] if parents else None
            if parent_values is None:
                self.set_value(node, None)
            elif len(parent_values) == 1:
                self.set_value(node, self.funcs[node](parent_values[0]))
            else:
                self.set_value(node, self.funcs[node](*parent_values))

    def run_inputs(self):
        for node in nx.topological_sort(self.G):
            if 'value' not in self.G.nodes[node]:
                self.compute_node(node)

    def intervene(self, node, value):
        if not self.validators[node](value):
            raise ValueError(f"Invalid value {value} for node {node}")
        self.G.nodes[node]['value'] = value
        self.G.nodes[node]['intervened'] = True  # Indicate that this node has been intervened on
        children = list(nx.descendants(self.G, node))
        for child in children:
            if 'value' in self.G.nodes[child]:
                del self.G.nodes[child]['value']
            self.G.nodes[child]['intervened'] = False  # Reset the 'intervened' attribute of the children nodes

    def run_with_intervention(self, intervention_node, intervention_value, **inputs):
        self.set_inputs(**inputs)
        self.intervene(intervention_node, intervention_value)
        self.run_inputs()

    def set_inputs(self, **inputs):
        input_nodes = self.get_roots()
        for node, value in inputs.items():
            if node not in input_nodes:
                raise ValueError(f"Node {node} is not an input node.")
            self.set_value(node, value)
        # Reset the 'intervened' attribute of all nodes
        for node in self.G.nodes:
            self.G.nodes[node]['intervened'] = False

    def reset_interventions(self):
        for node in self.G.nodes:
            self.G.nodes[node]['intervened'] = False

    def get_roots(self):
        return [node for node, degree in self.G.in_degree() if degree == 0]

    def visualize(self):
        nx.draw_networkx(self.G, with_labels=True)
        plt.show()