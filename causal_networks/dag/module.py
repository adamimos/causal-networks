from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn

from transformer_lens.hook_points import HookPoint


class DAGModule(nn.Module, ABC):
    """A module that can be used as a node in a DAG model"""

    def __init__(self, **kwargs):
        super().__init__()
        self._init_kwargs = kwargs
        self.value = None

    @abstractmethod
    def compute_value(self, parent_values: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the value of the node based on the values of its parents

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

    def forward(
        self, parent_values: dict[str, torch.Tensor], hook_point: Optional[HookPoint]
    ) -> torch.Tensor:
        """Do a forward pass through the module

        Uses the `compute_value` method to compute the value of the node based on the
        values of its parents. Optionally passes this through a hook point. Records this
        value in the `value` attribute, and returns it.

        Parameters
        ----------
        parent_values : dict[str, torch.Tensor]
            The values of the parents of the node
        hook_point : HookPoint, optional
            A hook point to pass the value through, by default None

        Returns
        -------
        value : torch.Tensor
            The value of the node
        """
        self.value = self.compute_value(parent_values)
        if hook_point is not None:
            self.value = hook_point(self.value)
        return self.value
    
    def get_value_as_integer(self) -> torch.LongTensor:
        """Return the value of the node as integers, if possible"""
        raise NotImplementedError

    def __repr__(self):
        init_kwargs_str = ", ".join(f"{k}={v!r}" for k, v in self._init_kwargs.items())
        return f"{self.__class__.__name__}({init_kwargs_str})"


class InputNode(DAGModule):
    """A node that takes an input value"""

    def compute_value(self, parent_values: dict[str, torch.Tensor]) -> torch.Tensor:
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
    """A node that checks if two inputs are equal
    
    Parameters
    ----------
    dim : int, optional
        If set, check that all elements along this dimension are equal. This dimension
        is reduced to size 1 and squeezed out if `keepdim` is False.
    keepdim : bool, default=True
        If True, keep the dimension along which the equality is checked. If False, the
        dimension is squeezed out.
    """

    def __init__(self, dim: Optional[int] = None, keepdim: bool = True):
        super().__init__(dim=dim, keepdim=keepdim)
        self.dim = dim
        self.keepdim = keepdim

    def compute_value(self, parent_values: dict[str, torch.Tensor]) -> torch.BoolTensor:
        """Return a boolean tensor indicating where the two parent values are equal

        Parameters
        ----------
        parent_values : dict[str, torch.Tensor]
            The values of the parents of the node. There must be two parents.

        Returns
        -------
        value : torch.BoolTensor
            A boolean tensor indicating where the two inputs are equal
        """
        if len(parent_values) != 2:
            raise ValueError("EqualityNode must have exactly two parents")
        values = list(parent_values.values())
        eqaulity_values = values[0] == values[1]
        if self.dim is not None:
            eqaulity_values = eqaulity_values.all(dim=self.dim, keepdim=self.keepdim)
        return eqaulity_values
    
    def get_value_as_integer(self) -> torch.LongTensor:
        """Return the value of the node as integers, if possible"""
        return self.value.long()


class CumSumNode(DAGModule):
    """A node that computes the cumulative sum of its parents
    
    Parameters
    ----------
    dim : int, default=-1
        The dimension to compute the cumulative sum over
    """

    def __init__(self, dim: int = -1):
        super().__init__(dim=dim)
        self.dim = dim

    def compute_value(self, parent_values: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the cumulative sum of the input

        Parameters
        ----------
        parent_values : dict[str, torch.Tensor]
            The values of the parents of the node. There must be one parent.

        Returns
        -------
        value : torch.Tensor
            The cumulative sum of the input
        """
        if len(parent_values) != 1:
            raise ValueError("CumSumNode must have exactly one parent")
        values = list(parent_values.values())
        return torch.cumsum(values[0], dim=self.dim)
    
    def get_value_as_integer(self) -> torch.LongTensor:
        """Return the value of the node as integers, if possible"""
        return self.value.long()
