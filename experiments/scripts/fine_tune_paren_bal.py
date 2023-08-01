from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
import json
from datetime import datetime

import torch

from sklearn.model_selection import ParameterGrid

from components import fine_tune_paren_bal

BASE_MODEL_NAME = "gelu-1l"

SCRIPT_PATH = os.path.realpath(__file__)
PROJECT_DIR = os.path.normpath(SCRIPT_PATH + "/../../..")
DATASET_DIR = os.path.join(PROJECT_DIR, "data")
EXPERIMENT_RESULTS_DIR = os.path.join(PROJECT_DIR, "experiments", "results")
TEXT_DATASET_NAME = "single_line.csv"
TEXT_DATASET_FILE = os.path.join(DATASET_DIR, "paren-balancing", TEXT_DATASET_NAME)

device = torch.device("cuda")


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
    "batch_size": [512],
    "learning_rate": [0.3, 0.1, 0.03, 0.01, 0.003, 0.001],
    "num_epochs": [50],
    "optimizer": ["Adam"],
    "lr_scheduler_patience": [1000],
}

# An iterator over the configurations of hyperparameters
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
        run_id = f"fine_tune_paren_bal_{combo_index}_{time_now}"

        # Print the run_id and the Parameters
        print()
        print()
        print("=" * 79)
        title = f"| FINE-TUNING PAREN BAL | Run ID: {run_id}"
        title += (" " * (78 - len(title))) + "|"
        print(title)
        print("=" * 79)
        print()
        print()

        results = fine_tune_paren_bal(
            base_model_name=BASE_MODEL_NAME,
            text_dataset_file=TEXT_DATASET_FILE,
            device=device,
            batch_size=combo["batch_size"],
            learning_rate=combo["learning_rate"],
            num_epochs=combo["num_epochs"],
            lr_scheduler_patience=combo["lr_scheduler_patience"],
            seed=combo["seed"],
        )

        print("Saving results...")
        results["train_losses"] = list(results["train_losses"])
        results["train_accuracies"] = list(results["train_accuracies"])
        results["base_model_name"] = BASE_MODEL_NAME
        results["text_dataset_name"] = TEXT_DATASET_NAME
        results["run_id"] = run_id
        results["combo_index"] = combo_index
        results["parameters"] = combo
        with open(
            os.path.join(EXPERIMENT_RESULTS_DIR, f"{run_id}.json"),
            "w",
        ) as f:
            json.dump(results, f)

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
