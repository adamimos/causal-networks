from dataclasses import dataclass
import os
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens.utils import to_numpy

from ..dag import DeterministicDAG
from ..variable_alignment import VariableAlignment


@dataclass
class Config:
    input_size: int = 4
    hidden_size: int = 16
    output_size: int = 2


# Create a three layer hooked MLP
class MLP(HookedRootModule):
    def __init__(self, cfg: Config, device=None):
        super().__init__()

        if device is None:
            device = torch.device("cpu")

        self.device = device

        self.cfg = cfg

        self.hook_pre1 = HookPoint()
        self.layer1 = nn.Linear(
            cfg.input_size, cfg.hidden_size, bias=True, device=device
        )
        self.hook_mid1 = HookPoint()
        self.hook_pre2 = HookPoint()
        self.layer2 = nn.Linear(
            cfg.hidden_size, cfg.hidden_size, bias=True, device=device
        )
        self.hook_mid2 = HookPoint()
        self.hook_pre3 = HookPoint()
        self.layer3 = nn.Linear(
            cfg.hidden_size, cfg.output_size, bias=True, device=device
        )
        self.hook_mid3 = HookPoint()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.hook_pre1(x)
        x = self.hook_mid1(self.layer1(x))
        x = self.relu(x)
        x = self.hook_pre2(x)
        x = self.hook_mid2(self.layer2(x))
        x = self.relu(x)
        x = self.hook_pre3(x)
        x = self.hook_mid3(self.layer3(x))
        return x


def generate_grouped_data(n, size_per_input):
    """Generates n data points with 4 balanced groups,
    (w==x, y==z), (w==x, y!=z), (w!=x, y==z), (w!=x, y!=z)
    For the hierarchical equality task"""

    data = np.empty((n, 4, size_per_input))
    labels = np.empty(n)

    quarter = n // 4

    # w==x, y!=z
    data[:quarter, 0] = data[:quarter, 1] = np.random.uniform(
        -0.5, 0.5, (quarter, size_per_input)
    )
    data[:quarter, 2] = np.random.uniform(-0.5, 0.5, (quarter, size_per_input))
    data[:quarter, 3] = np.random.uniform(-0.5, 0.5, (quarter, size_per_input))
    labels[:quarter] = 0

    # w==x, y==z
    data[quarter : 2 * quarter, 0] = data[quarter : 2 * quarter, 1] = np.random.uniform(
        -0.5, 0.5, (quarter, size_per_input)
    )
    data[quarter : 2 * quarter, 2] = data[quarter : 2 * quarter, 3] = np.random.uniform(
        -0.5, 0.5, (quarter, size_per_input)
    )
    labels[quarter : 2 * quarter] = 1

    # w!=x, y==z
    data[2 * quarter : 3 * quarter, 0] = np.random.uniform(
        -0.5, 0.5, (quarter, size_per_input)
    )
    data[2 * quarter : 3 * quarter, 1] = np.random.uniform(
        -0.5, 0.5, (quarter, size_per_input)
    )
    data[2 * quarter : 3 * quarter, 2] = data[
        2 * quarter : 3 * quarter, 3
    ] = np.random.uniform(-0.5, 0.5, (quarter, size_per_input))
    labels[2 * quarter : 3 * quarter] = 0

    # w!=x, y!=z
    data[3 * quarter :, 0] = np.random.uniform(
        -0.5, 0.5, (n - 3 * quarter, size_per_input)
    )
    data[3 * quarter :, 1] = np.random.uniform(
        -0.5, 0.5, (n - 3 * quarter, size_per_input)
    )
    data[3 * quarter :, 2] = np.random.uniform(
        -0.5, 0.5, (n - 3 * quarter, size_per_input)
    )
    data[3 * quarter :, 3] = np.random.uniform(
        -0.5, 0.5, (n - 3 * quarter, size_per_input)
    )
    labels[3 * quarter :] = 1

    data = data.reshape(n, 4 * size_per_input)

    permute = np.random.permutation(n)
    data = data[permute]
    labels = labels[permute]

    return data, labels


def make_hier_equal_dag_and_variable_alignment(
    hidden_size: int,
    size_per_input: int,
    intervene_model_hooks: list[str],
    intervene_nodes: list[str],
    subspace_sizes: list[int],
    device,
    task_train_size=100000,
    task_epochs=5000,
    task_lr=0.01,
    task_scheduler_patience=1000,
) -> tuple[DeterministicDAG, VariableAlignment, HookedRootModule]:
    """Make a DAG and variable alignment for the hierarchical equality task"""

    # Define the model
    model = MLP(
        Config(hidden_size=hidden_size, input_size=4 * size_per_input), device=device
    )
    model.setup()

    # define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=task_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=task_scheduler_patience, verbose=True
    )

    # Generate some data
    data, labels = generate_grouped_data(task_train_size, size_per_input)

    # convert the data and labels to torch tensors
    data = torch.from_numpy(data).float()
    labels = torch.from_numpy(labels).long()

    data = data.to(device)
    labels = labels.to(device)
    model = model.to(device)

    # train the model
    print("Training the model...")
    for epoch in range(task_epochs):
        outputs = model(data)
        loss = criterion(outputs, labels)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

    predicted = torch.argmax(outputs, dim=1)
    correct = (predicted == labels).sum().item()

    print(
        f"Final loss: {loss.item():0.5}. Final accuracy: {correct / task_train_size:0.5%}"
    )

    def array_float_validator(value, size_per_input):
        return (
            isinstance(value, np.ndarray)
            and value.shape == (size_per_input,)
            and ((-0.5 <= value) & (value <= 0.5)).all()
        )

    # Bit of a hack to make the table work
    array_float_validator_size = partial(
        array_float_validator, size_per_input=size_per_input
    )
    array_float_validator_size.__name__ = f"array_float_validator(-, {size_per_input})"

    def bool_validator(value):
        return value is True or value is False

    def compare_func(value1, value2):
        return np.array_equal(value1, value2)

    dag = DeterministicDAG()

    # Define the nodes
    dag.add_node("x1", validator=array_float_validator_size)
    dag.add_node("x2", validator=array_float_validator_size)
    dag.add_node("x3", validator=array_float_validator_size)
    dag.add_node("x4", validator=array_float_validator_size)
    dag.add_node("b1", func=compare_func, validator=bool_validator)
    dag.add_node("b2", func=compare_func, validator=bool_validator)
    dag.add_node(
        "y", func=compare_func, validator=bool_validator, possible_values=[False, True]
    )

    # Define the edges
    edges = [
        ("x1", "b1"),
        ("x2", "b1"),
        ("x3", "b2"),
        ("x4", "b2"),
        ("b1", "y"),
        ("b2", "y"),
    ]
    for edge in edges:
        dag.add_edge(*edge)

    def input_alignment(x: torch.tensor):
        x = to_numpy(x)
        if x.ndim == 1:
            return dict(
                x1=x[:size_per_input],
                x2=x[size_per_input : 2 * size_per_input],
                x3=x[2 * size_per_input : 3 * size_per_input],
                x4=x[3 * size_per_input :],
            )
        elif x.ndim == 2:
            return [
                dict(
                    x1=x[i, :size_per_input],
                    x2=x[i, size_per_input : 2 * size_per_input],
                    x3=x[i, 2 * size_per_input : 3 * size_per_input],
                    x4=x[i, 3 * size_per_input :],
                )
                for i in range(x.shape[0])
            ]
        else:
            raise ValueError("Invalid input shape")

    def output_alignment(y: torch.tensor):
        return dict(y=torch.argmax(y).item())

    variable_alignment = VariableAlignment(
        dag=dag,
        low_level_model=model,
        dag_nodes=intervene_nodes,
        input_alignment=input_alignment,
        output_alignment=output_alignment,
        intervene_model_hooks=intervene_model_hooks,
        subspaces_sizes=subspace_sizes,
        device=device,
        verbosity=1,
    )

    return dag, variable_alignment, model
