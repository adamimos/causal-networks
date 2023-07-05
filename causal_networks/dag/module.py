from abc import ABC, abstractmethod

import torch
from torch import nn


class DAGModule(nn.Module, ABC):
    """A module that can be used as a node in a DAG model"""
    
    def __init__(self):
        super().__init__()
        self.value = None

    @abstractmethod
    def forward(self, parent_values: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the value of the node based on the values of its parents

        Records the value of the node in the `value` attribute, and returns it.
        
        Parameters
        ----------
        parent_values : dict[str, torch.Tensor]
            The values of the parents of the node
        
        Returns
        -------
        value : torch.Tensor
            The value of the node
        """
        pass

class InputNode(DAGModule):
    """A node that takes an input value"""
    def __init__(self):
        super().__init__()
        self.value = None

    def forward(self, parent_values: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the value of the node
        
        Parameters
        ----------
        parent_values : dict[str, torch.Tensor]
            The values of the parents of the node (ignored)
        
        Returns
        -------
        value : torch.Tensor
            The value of the node
        """
        return self.value
    
class EqualityNode(DAGModule):
    """A node that checks if two inputs are equal"""

    def forward(self, parent_values: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the value of the node
        
        Parameters
        ----------
        parent_values : dict[str, torch.Tensor]
            The values of the parents of the node. There must be two parents.
        
        Returns
        -------
        value : torch.Tensor (boolean)
            A boolean tensor indicating where the two inputs are equal
        """
        if len(parent_values) != 2:
            raise ValueError("EqualityNode must have exactly two parents")
        values = list(parent_values.values())
        return values[0] == values[1]