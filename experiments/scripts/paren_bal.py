from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
from datetime import datetime
import json

import torch

import numpy as np

from sklearn.model_selection import ParameterGrid

import pandas as pd

from causal_networks.models import make_paren_bal_dag_and_variable_alignment

SCRIPT_PATH = os.path.realpath(__file__)
PROJECT_DIR = os.path.normpath(SCRIPT_PATH + "/../../..")
DATASET_DIR = os.path.join(PROJECT_DIR, "data")
EXPERIMENT_RESULTS_DIR = os.path.join(PROJECT_DIR, "experiments", "results")

# Set up the arg parser
parser = ArgumentParser(
    description="Run the parenthesis balancing experiments",
    formatter_class=ArgumentDefaultsHelpFormatter,
)

# Add various arguments
parser.add_argument(
    "--combo-groups",
    type=int,
    default=1,
    help="Into how many groups to split the experiment combinations",
)
parser.add_argument(
    "--combo-num",
    type=int,
    default=0,
    help="Which combo group to run this time",
)
parser.add_argument(
    "--num-skip",
    type=int,
    default=0,
    help="The number of initial combos to skip. Useful to resume a group",
)
parser.add_argument(
    "--gpu-num", type=int, default=0, help="The (0-indexed) GPU number to use"
)

# Get the arguments
cmd_args = parser.parse_args()

# The different hyperparameters to test
param_grid = {
    "seed": [2384],
    "model_name": ["gelu-1l"],
    "dataset_file": ["clean_data.csv"],
    "dataset_max_len": [10000],
    "test_proportion": [0.1],
    "train_ii_dataset_size": [10000],
    "test_ii_dataset_size": [1000],
    "batch_size": [128],
    "intervene_node": ["v", "s"],
    "subspace_size": [64, 256],
    "intervene_hook": [
        "blocks.0.hook_resid_pre",
        "blocks.0.hook_resid_mid",
        "blocks.0.hook_resid_post",
    ],
    "train_lr": [1, 100, 10000],
    "num_epochs": [30],
}

# An interator over the configurations of hyperparameters
param_iter = ParameterGrid(param_grid)

# Enumerate these to keep track of them
combinations = enumerate(param_iter)

# Filter to combos
combinations = filter(
    lambda x: x[0] % cmd_args.combo_groups == cmd_args.combo_num, combinations
)
combinations = list(combinations)[cmd_args.num_skip :]

# Keep track of the results of the runs
run_results = []
for combo_num in range(len(combinations)):
    run_results.append("SKIPPED")

try:
    # Run the experiment for each sampled combination of parameters
    for i, (combo_index, combo) in enumerate(combinations):
        # Set the status of the current run to failed until proven otherwise
        run_results[i] = "FAILED"

        time_now = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")

        # Create a unique run_id for this trial
        run_id = f"paren_bal_{combo_index}_{time_now}"

        # Print the run_id and the Parameters
        print()
        print()
        print("=" * 79)
        title = f"| PAREN BALANCING EXPERIMENT | Run ID: {run_id}"
        title += (" " * (78 - len(title))) + "|"
        print(title)
        print("=" * 79)
        print()
        print()

        print("Setting the seed...")
        torch.manual_seed(combo["seed"])
        np.random.seed(combo["seed"])

        print("Making DAG and Variable Alignment...")
        dag, variable_alignment, model = make_paren_bal_dag_and_variable_alignment(
            model_name=combo["model_name"],
            intervene_model_hooks=[combo["intervene_hook"]],
            intervene_nodes=[combo["intervene_node"]],
            subspace_sizes=[combo["subspace_size"]],
            device=f"cuda:{cmd_args.gpu_num}",
        )

        print("Loading dataset...")
        data_df = pd.read_csv(os.path.join(DATASET_DIR, combo["dataset_file"]))
        data_df = data_df.iloc[: combo["dataset_max_len"]]

        print("Tokenizing dataset...")
        data_tokens = model.to_tokens(data_df["text"].values, move_to_device=False)

        print("Creating interchanging intervention datasets...")
        train_tokens = data_tokens[
            int(combo["test_proportion"] * data_tokens.shape[0]) :
        ]
        test_tokens = data_tokens[
            : int(combo["test_proportion"] * data_tokens.shape[0])
        ]
        train_ii_dataset = variable_alignment.create_interchange_intervention_dataset(
            train_tokens, num_samples=combo["train_ii_dataset_size"]
        )
        test_ii_dataset = variable_alignment.create_interchange_intervention_dataset(
            test_tokens, num_samples=combo["test_ii_dataset_size"]
        )

        print("Training rotation matrix...")
        train_losses, train_accuracies = variable_alignment.train_rotation_matrix(
            ii_dataset=train_ii_dataset,
            num_epochs=combo["num_epochs"],
            batch_size=combo["batch_size"],
        )

        print("Evaluating rotation matrix...")
        test_loss, test_accuracy = variable_alignment.test_rotation_matrix(
            ii_dataset=test_ii_dataset, batch_size=combo["batch_size"]
        )

        print("Saving results...")
        results_dict = {
            "run_id": run_id,
            "combo_index": combo_index,
            "parameters": combo,
            "train_losses": list(train_losses),
            "train_accuracies": list(train_accuracies),
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        }
        with open(
            os.path.join(EXPERIMENT_RESULTS_DIR, f"{run_id}.json"),
            "w",
        ) as f:
            json.dump(results_dict, f)

        run_results[i] = "SUCCEEDED"

finally:
    # Print a summary of the experiment results
    print()
    print()
    print("=" * 79)
    title = f"| SUMMARY | GROUP {cmd_args.combo_num}/{cmd_args.combo_groups}"
    title += (" " * (78 - len(title))) + "|"
    print(title)
    print("=" * 79)
    for result, (combo_num, combo) in zip(run_results, combinations):
        print()
        print(f"COMBO {combo_num}")
        print(combo)
        print(result)
