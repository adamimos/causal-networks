from functools import partial

import torch
import torch.nn.functional as F

import numpy as np

from transformer_lens import HookedTransformer
from transformer_lens.utils import to_numpy

from ..dag import (
    DAGModel,
    InputNode,
    NotNode,
    AndNode,
    GreaterThanZeroNode,
    InSetNode,
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


def make_suppressing_pb_dag(
    model: HookedTransformer, device: str | torch.device = "cpu"
):
    """Cumsum DAG with suppressing tokens for parenthesis balancing"""

    open_paren_tokens, closed_paren_tokens, suppressing_tokens = _get_important_tokens(
        model
    )

    def vis_layout(*args, **kwargs):
        return {
            "input": (1, 4),
            "bracket": (0, 3),
            "sum": (0, 2),
            "gt0": (0, 1),
            "supp": (2, 3),
            "not": (2, 1),
            "and": (1, 0),
        }

    dag = DAGModel()

    dag.add_node(
        "input",
        InputNode(
            display_value_converter=partial(input_display_value_converter, model=model)
        ),
    )
    dag.add_node("bracket", InSetOutSetNode(open_paren_tokens, closed_paren_tokens))
    dag.add_node("sum", CumSumNode())
    dag.add_node("gt0", GreaterThanZeroNode())
    dag.add_node("supp", InSetNode(suppressing_tokens))
    dag.add_node("not", NotNode())
    dag.add_node("and", AndNode())

    dag.add_edge("input", "bracket")
    dag.add_edge("bracket", "sum")
    dag.add_edge("sum", "gt0")
    dag.add_edge("gt0", "and")
    dag.add_edge("input", "supp")
    dag.add_edge("supp", "not")
    dag.add_edge("not", "and")

    dag.set_visualization_params(
        layout=vis_layout,
        canvas_size=(5, 4),
        draw_kwargs=dict(node_size=1000, node_shape="o"),
    )

    dag.to(device)

    return dag
