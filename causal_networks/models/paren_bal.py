from collections import OrderedDict

import torch
import torch.nn.functional as F

import numpy as np

from transformer_lens import HookedTransformer
from transformer_lens.utils import to_numpy

from ..dag import RecurrentDeterministicDAG
from ..variable_alignment import VariableAlignmentTransformer

OPEN_PAREN_STR_TOKENS = ["("]
CLOSE_PAREN_STR_TOKENS = [")"]
SUPPRESSING_STR_TOKENS = ["(", "_", ",", "+", "."]


def make_paren_bal_dag_and_variable_alignment(
    model_name: str,
    intervene_model_hooks: list[str],
    intervene_nodes: list[str],
    subspace_sizes: list[int],
    device,
) -> tuple[RecurrentDeterministicDAG, VariableAlignmentTransformer, HookedTransformer]:
    """Make a DAG and variable alignment for the parenthesis balancing task"""

    ## Load the model and tokens
    model = HookedTransformer.from_pretrained(model_name, device=device)

    d_vocab = model.config.vocab_size

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

    str_tokens = model.to_str_tokens(torch.arange(d_vocab), prepend_bos=False)

    ## Make the DAG
    dag = RecurrentDeterministicDAG()

    def input_token_validator(symbol):
        return symbol in str_tokens

    def interp_symbol_func(x):
        if x in OPEN_PAREN_STR_TOKENS:
            return 1
        elif x in CLOSE_PAREN_STR_TOKENS:
            return -1
        else:
            return 0

    def cumsum_func(v):
        return sum(v)

    def close_paren_func(s):
        if s > 0:
            return True
        else:
            return False

    dag.add_node("x", validator=input_token_validator)
    dag.add_node("v", func=interp_symbol_func)
    dag.add_node("s", func=cumsum_func)
    dag.add_node("c", func=close_paren_func, possible_values=[False, True])

    dag.add_edge("x", "v")
    dag.add_edge("v", "s", edge_type="current_and_previous_streams")
    dag.add_edge("s", "c")

    def vectorised_runner(inputs, output_type="all_nodes", output_format="ordereddict"):
        x = inputs["x"]
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        v = (x == open_paren_tokens[0]).astype(int) - (
            x == closed_paren_tokens[0]
        ).astype(int)
        s = np.cumsum(v, axis=1)
        c = s > 0

        if output_type == "all_nodes":
            output = OrderedDict(x=x, v=v, s=s, c=c)
        elif output_type == "output_nodes":
            output = OrderedDict(c=c)
        elif output_type == "output_integer":
            output = OrderedDict(c=c.astype(int))

        if output_format == "torch":
            output = torch.from_numpy(np.array(list(output.values())))

        return output

    dag.vectorised_runner = vectorised_runner

    ## Make the variable alignment

    def input_alignment(x: torch.tensor, vectorised=False):
        if vectorised:
            return dict(x=to_numpy(x))
        if x.ndim == 1:
            return dict(x=model.to_str_tokens(x))
        elif x.ndim == 2:
            return [
                dict(
                    x=model.to_str_tokens(x[i]),
                )
                for i in range(x.shape[0])
            ]
        else:
            raise ValueError("Invalid input shape")

    def output_alignment(y: torch.tensor):
        return dict(y=list(to_numpy(torch.argmax(y, dim=-1))))

    def output_modifier(y: torch.tensor):
        binary_output = torch.empty(*y.shape[:-1], 2, device=y.device)
        binary_output[..., 0] = y[..., open_paren_tokens].sum(dim=-1)
        binary_output[..., 1] = y[..., closed_paren_tokens].sum(dim=-1)

        return F.softmax(binary_output, dim=-1)

    variable_alignment = VariableAlignmentTransformer(
        dag=dag,
        low_level_model=model,
        dag_nodes=intervene_nodes,
        input_alignment=input_alignment,
        output_alignment=output_alignment,
        intervene_model_hooks=intervene_model_hooks,
        subspaces_sizes=subspace_sizes,
        output_modifier=output_modifier,
        device=device,
        verbosity=1,
    )

    return dag, variable_alignment, model
