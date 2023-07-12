from functools import partial

import torch
import torch.nn.functional as F

import numpy as np

from transformer_lens import HookedTransformer
from transformer_lens.utils import to_numpy

from ..dag import (
    DAGModel,
    InputNode,
    GreaterThanZeroNode,
    InSetOutSetNode,
    CumSumNode,
)
from ..variable_alignment import TransformerVariableAlignment

OPEN_PAREN_STR_TOKENS = ["("]
CLOSE_PAREN_STR_TOKENS = [")"]
SUPPRESSING_STR_TOKENS = ["(", "_", ",", "+", "."]


def input_display_value_converter(x: torch.Tensor, model: HookedTransformer) -> str:
    if x.ndim == 1:
        return str(model.to_str_tokens(x, prepend_bos=False))
    elif x.ndim == 2:
        return str(
            [model.to_str_tokens(x[i], prepend_bos=False) for i in range(x.shape[0])]
        )
    else:
        raise ValueError("Unsupported input dimensionality")


def _get_important_tokens(model: HookedTransformer):
    open_paren_tokens = model.to_tokens(
        OPEN_PAREN_STR_TOKENS, prepend_bos=False, move_to_device=False, truncate=False
    )
    open_paren_tokens = [
        open_paren_tokens[i, 0].item() for i in range(open_paren_tokens.shape[0])
    ]

    closed_paren_tokens = model.to_tokens(
        CLOSE_PAREN_STR_TOKENS, prepend_bos=False, move_to_device=False, truncate=False
    )
    closed_paren_tokens = [
        closed_paren_tokens[i, 0].item() for i in range(closed_paren_tokens.shape[0])
    ]

    suppressing_tokens = model.to_tokens(
        SUPPRESSING_STR_TOKENS, prepend_bos=False, move_to_device=False, truncate=False
    )
    suppressing_tokens = [
        suppressing_tokens[i, 0].item() for i in range(suppressing_tokens.shape[0])
    ]
    return open_paren_tokens, closed_paren_tokens, suppressing_tokens


def make_basic_pb_dag(model: HookedTransformer, device: str | torch.device = "cpu"):
    """Four-node cumsum DAG for parenthesis balancing"""

    open_paren_tokens, closed_paren_tokens, _ = _get_important_tokens(model)

    def vis_layout(*args, **kwargs):
        return dict(x=(0, 0), v=(1, 0), s=(2, 0), c=(3, 0))

    dag = DAGModel()

    dag.add_node(
        "x",
        InputNode(
            display_value_converter=partial(input_display_value_converter, model=model)
        ),
    )
    dag.add_node("v", InSetOutSetNode(open_paren_tokens, closed_paren_tokens))
    dag.add_node("s", CumSumNode())
    dag.add_node("c", GreaterThanZeroNode())

    dag.add_edge("x", "v")
    dag.add_edge("v", "s")
    dag.add_edge("s", "c")

    dag.set_visualization_params(layout=vis_layout, canvas_size=(3, 1))

    dag.to(device)

    return dag
