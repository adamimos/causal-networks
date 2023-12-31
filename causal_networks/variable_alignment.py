from functools import partial
from typing import Any, Optional, Union
import time

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.utils.data import (
    Dataset,
    DataLoader,
    TensorDataset,
    BatchSampler,
    SequentialSampler,
    RandomSampler,
)
from torch.nn.utils.parametrizations import orthogonal

import einops

from jaxtyping import Float, Bool

import numpy as np

from tqdm import tqdm

from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens import HookedTransformer

from .dag import DAGModel


def mean_cross_entropy(input, target):
    return F.cross_entropy(input, target, reduction="mean")


class ParametrisedOrthogonalMatrix(nn.Module):
    """A parametrised orthogonal matrix"""

    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.orthogonal_parameter = orthogonal(nn.Linear(size, size, bias=False))

    def get_orthogonal_matrix(self) -> Tensor:
        return self.orthogonal_parameter.weight

    def forward(self, x: Tensor) -> Tensor:
        return torch.matmul(x, self.get_orthogonal_matrix())

    def inverse(self, x: Tensor) -> Tensor:
        return torch.matmul(x, self.get_orthogonal_matrix().T)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.orthogonal_parameter = self.orthogonal_parameter.to(*args, **kwargs)
        return self


class InterchangeInterventionDataset(Dataset):
    """A dataset for doing interchange intervention

    At each step it selects a random subset of the DAG nodes and random base and source
    inputs. It always returns the same number of source inputs, filling in for missing
    nodes with the base input.

    Parameters
    ----------
    inputs : Float[Tensor, "num_inputs input_size"]
        The inputs from which the base and source inputs are sampled
    activation_values : Float[Tensor, "num_inputs total_space_size"]
        The activation values for the inputs. For each input this is a vector of size
        the whole activation space (possibly the concatenation of several layers)
    base_input_indices : Float[Tensor, "num_samples"]
        The indices of the base inputs to use
    source_input_indices : Float[Tensor, "num_samples num_nodes"]
        The indices of the source inputs to use
    gold_outputs : Float[Tensor, "num_samples"]
        The gold outputs for the base and source inputs
    """

    def __init__(
        self,
        inputs: Float[Tensor, "num_inputs input_size"],
        activation_values: Float[Tensor, "num_inputs total_space_size"],
        base_input_indices: Float[Tensor, "num_samples"],
        source_input_indices: Float[Tensor, "num_samples num_nodes"],
        gold_outputs: Float[Tensor, "num_samples"],
    ):
        super().__init__()
        self.inputs = inputs
        self.activation_values = activation_values
        self.base_input_indices = base_input_indices
        self.source_input_indices = source_input_indices
        self.gold_outputs = gold_outputs

    def __len__(self):
        return self.base_input_indices.shape[0]

    def __getitem__(
        self, idx: Any
    ) -> tuple[
        Float[Tensor, "num_samples input_size"],
        Float[Tensor, "num_samples num_nodes input_size"],
        Float[Tensor, "num_samples 1+num_nodes total_space_size"],
        Float[Tensor, "num_samples"],
    ]:
        # For the following shapes we assume that idx is 1 dimensional
        if torch.is_tensor(idx):
            idx_ndim = idx.ndim
        elif isinstance(idx, slice) or isinstance(idx, list):
            idx_ndim = 1
        else:
            idx_ndim = 0

        # (num_samples,)
        base_inputs = self.inputs[self.base_input_indices[idx]]

        # (num_samples, num_nodes)
        source_inputs = self.inputs[self.source_input_indices[idx]]

        # (num_samples)
        gold_outputs = self.gold_outputs[idx]

        # (num_samples, total_space_size)
        base_input_activations = self.activation_values[self.base_input_indices[idx]]

        # (num_samples, num_nodes, total_space_size)
        source_input_activations = self.activation_values[
            self.source_input_indices[idx]
        ]

        # (num_samples, 1 + num_nodes, total_space_size)
        combined_input_activations = torch.cat(
            (
                base_input_activations.unsqueeze(idx_ndim),
                source_input_activations,
            ),
            dim=idx_ndim,
        )

        return (
            base_inputs,
            source_inputs,
            combined_input_activations,
            gold_outputs,
        )


class TransformerInterchangeInterventionDataset(InterchangeInterventionDataset):
    """A dataset for doing interchange intervention on a transformer

    At each step it selects a random subset of the DAG nodes and random base and source
    inputs. It always returns the same number of source inputs, filling in for missing
    nodes with the base input.

    Parameters
    ----------
    inputs : Float[Tensor, "num_inputs seq_len"]
        The inputs from which the base and source inputs are sampled
    activation_values : Float[Tensor, "num_inputs seq_len total_space_size"]
        The activation values for the inputs. For each input this is a vector of size
        the whole activation space (possibly the concatenation of several layers)
    base_input_indices : Float[Tensor, "num_samples"]
        The indices of the base inputs to use
    source_input_indices : Float[Tensor, "num_samples num_nodes seq_len"]
        The indices of the source inputs to use
    gold_outputs : Float[Tensor, "num_samples seq_len"]
        The gold outputs for the base and source inputs
    loss_mask : Bool[Tensor, "num_samples seq_len"], optional
        A mask to apply to the inputs to indicate which positions should be used for
        computing the loss. If None, all positions are used.
    """

    def __init__(
        self,
        inputs: Float[Tensor, "num_inputs seq_len"],
        activation_values: Float[Tensor, "num_inputs seq_len total_space_size"],
        base_input_indices: Float[Tensor, "num_samples"],
        source_input_indices: Float[Tensor, "num_samples num_nodes seq_len"],
        gold_outputs: Float[Tensor, "num_samples seq_len"],
        loss_mask: Optional[Bool[Tensor, "num_samples seq_len"]] = None,
    ):
        super().__init__(
            inputs=inputs,
            activation_values=activation_values,
            base_input_indices=base_input_indices,
            source_input_indices=source_input_indices,
            gold_outputs=gold_outputs,
        )
        if loss_mask is None:
            loss_mask = torch.ones_like(gold_outputs, dtype=torch.bool)
        self.loss_mask = loss_mask

    def __getitem__(
        self, idx: Any
    ) -> tuple[
        Float[Tensor, "num_samples seq_len"],
        Float[Tensor, "num_samples num_nodes seq_len seq_len"],
        Float[Tensor, "num_samples 1+num_nodes*seq_len total_space_size"],
        Float[Tensor, "num_samples seq_len"],
        Bool[Tensor, "num_samples seq_len"],
    ]:
        # For the following shapes we assume that idx is 1 dimensional
        if torch.is_tensor(idx):
            idx_ndim = idx.ndim
        elif isinstance(idx, slice) or isinstance(idx, list):
            idx_ndim = 1
        else:
            idx_ndim = 0

        # (num_samples, seq_len)
        base_inputs = self.inputs[self.base_input_indices[idx]]

        # (num_samples, num_nodes, seq_len, seq_len)
        source_inputs = self.inputs[self.source_input_indices[idx]]

        # (num_samples, seq_len)
        gold_outputs = self.gold_outputs[idx]

        # (num_samples, seq_len)
        loss_mask = self.loss_mask[idx]

        # (num_samples, seq_len, total_space_size)
        base_input_activations = self.activation_values[self.base_input_indices[idx]]

        # (num_samples, num_nodes, seq_len, seq_len, total_space_size)
        source_input_activations = self.activation_values[
            self.source_input_indices[idx]
        ]

        # (num_samples, 1 + num_nodes * seq_len, seq_len, total_space_size)
        combined_input_activations = torch.cat(
            (
                base_input_activations.unsqueeze(idx_ndim),
                source_input_activations.flatten(idx_ndim, idx_ndim + 1),
            ),
            dim=idx_ndim,
        )

        return (
            base_inputs,
            source_inputs,
            combined_input_activations,
            gold_outputs,
            loss_mask,
        )


class VariableAlignment:
    """A variable alignment between a DAG and a hooked model

    Parameters
    ----------
    dag : DAGModel
        The DAG to align to the low-level model
    low_level_model : HookedRootModule
        The hooked low-level model to align to the DAG
    dag_nodes : list[str]
        The names of the nodes (variables) of the DAG to align
    input_alignment : callable
        A function mapping (batches of) model input tensors to DAG input
        dictionaries
    intervene_model_hooks : list[str]
        The names of the model hooks together give the whole activation space
        for alignment the DAG nodes
    subspaces_sizes : list[int]
        The sizes of the subspaces to use for each DAG nodes
    output_modifier : callable, default=None
        A function to apply to the model outputs before computing the DII loss
    device : Union[str, torch.device], default=None
        The device to put everything on. If None, use CUDA if available,
        otherwise CPU
    verbosity : int, default=1
        The verbosity level
    progress_bar : bool, default=True
        Whether to show a progress bar during and dataset generation
    """

    def __init__(
        self,
        dag: DAGModel,
        low_level_model: HookedRootModule,
        dag_nodes: list[str],
        input_alignment: callable,
        intervene_model_hooks: list[str],
        subspaces_sizes: list[int],
        output_modifier: Optional[callable] = None,
        device: Optional[Union[str, torch.device]] = None,
        verbosity: int = 1,
        progress_bar: bool = True,
        debug: bool = False,
    ):
        if len(dag_nodes) != len(subspaces_sizes):
            raise ValueError(
                f"Expected {len(dag_nodes)} subspaces sizes, got {len(subspaces_sizes)}"
            )

        if len(dag.get_output_nodes()) != 1:
            raise ValueError(
                f"Expected DAG to have exactly one output node, got {len(dag.get_output_nodes)}"
            )

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dag = dag
        self.low_level_model = low_level_model
        self.dag_nodes = dag_nodes
        self.input_alignment = input_alignment
        self.intervene_model_hooks = intervene_model_hooks
        self.subspaces_sizes = subspaces_sizes
        self.output_modifier = output_modifier
        self.device = device
        self.verbosity = verbosity
        self.progress_bar = progress_bar
        self.debug = debug

        self._setup()

    def _setup(self):
        self.num_nodes = len(self.dag_nodes)

        self.layer_sizes = self._determine_layer_sizes()
        self.total_space_size = sum(self.layer_sizes)

        self.y_0_size = self.total_space_size - sum(self.subspaces_sizes)
        self.subspaces_sizes_with_y0 = [self.y_0_size] + self.subspaces_sizes

        if sum(self.subspaces_sizes) > self.total_space_size:
            raise ValueError(
                f"Sum of subspace sizes ({sum(self.subspaces_sizes)}) "
                f"exceeds activation space size ({self.total_space_size})"
            )

        self.rotation = ParametrisedOrthogonalMatrix(self.total_space_size)
        self.rotation = self.rotation.to(self.device)

    def _determine_layer_sizes(self) -> list[int]:
        """Run the model to determine the sizes of the activation spaces"""

        if self.verbosity > 0:
            print("Running model to determine activation space size...")

        sizes = []

        def counter_hook(value, hook: HookPoint, sizes):
            sizes.append(value.shape[1])

        partial_counter_hook = partial(counter_hook, sizes=sizes)

        x = torch.zeros(1, self.low_level_model.cfg.input_size, device=self.device)
        fwd_hooks = [
            (name, partial_counter_hook) for name in self.intervene_model_hooks
        ]

        self.low_level_model.run_with_hooks(x, fwd_hooks=fwd_hooks)

        return sizes

    def get_distributed_interchange_intervention_hooks(
        self,
        base_input: Float[Tensor, "batch_size input_size"],
        source_inputs: Float[Tensor, "batch_size num_nodes input_size"],
        combined_activations: Float[Tensor, "batch_size 1+num_nodes total_space_size"],
    ) -> list[tuple[str, callable]]:
        """Get hooks for distributed interchange intervention

        Runs the model on the base input and each source input, and on input
        `i` records the value of the projection of the rotated activation
        space onto the `i`th subspace. It then builds hooks to patch in these
        values into the rotated activation space.

        Parameters
        ----------
        base_input : Float[Tensor, "batch_size input_size"]
            The base input to the model
        source_inputs : Float[Tensor, "batch_size num_nodes input_size"]
            The source inputs to the model
        combined_activations : Float[Tensor, "batch_size 1+num_nodes total_space_size"]
            The activations of the model on the base input and each source input

        Returns
        -------
        list[tuple[str, callable]]
            The hooks to use for distributed interchange intervention, for
            each of `self.intervene_model_hooks`
        """

        if source_inputs.shape[-2] != self.num_nodes:
            raise ValueError(
                f"Expected {self.num_nodes} source inputs, got {source_inputs.shape[0]}"
            )

        batch_size = base_input.shape[0]

        # (batch_size * (1 + num_nodes), total_space_size)
        activation_values = combined_activations.view(-1, self.total_space_size)

        # (batch_size * (1 + num_nodes), total_space_size)
        rotated_activation_values = self.rotation(activation_values)

        # Select the rotated activation values which are relevant, for patching in
        indices_list = []
        for i, subspace_size in enumerate(self.subspaces_sizes_with_y0):
            indices_list.append(torch.ones(subspace_size, device=self.device) * i)
        selection_indices = torch.cat(indices_list).long()
        selection_indices = selection_indices.tile((batch_size, 1))
        selection_indices = selection_indices + torch.arange(
            batch_size, device=self.device
        ).unsqueeze(1) * (self.num_nodes + 1)
        # (batch_size, total_space_size)
        new_rotated_activation_values = rotated_activation_values[
            selection_indices, torch.arange(self.total_space_size).tile((batch_size, 1))
        ]

        # Unrotate the new rotated activation values, to get a vector
        # consisting of all the activation values to be patched in
        # (batch_size, total_space_size)
        new_activation_values = self.rotation.inverse(new_rotated_activation_values)

        def intervention_hook(
            value,
            hook: HookPoint,
            space_index,
            new_activation_values,
        ):
            value[:] = new_activation_values[
                ..., space_index : space_index + value.shape[-1]
            ]
            return value

        fwd_hooks = []
        for i, name in enumerate(self.intervene_model_hooks):
            fwd_hooks.append(
                (
                    name,
                    partial(
                        intervention_hook,
                        space_index=sum(self.layer_sizes[:i]),
                        new_activation_values=new_activation_values,
                    ),
                )
            )

        return fwd_hooks

    def run_distributed_interchange_intervention(
        self,
        base_input: Float[Tensor, "batch_size input_size"],
        source_inputs: Float[Tensor, "batch_size num_nodes input_size"],
        combined_activations: Float[Tensor, "batch_size 1+num_nodes total_space_size"],
    ) -> Float[Tensor, "batch_size output_size"]:
        """Compute the output of the low-level model on base and source inputs

        Does distributed interchange intervention, patching in the activation values
        from the source inputs, then running on the base input

        **Warning**: This method will disable gradients for the model parameters,
        setting `requires_grad` to `False`.

        Parameters
        ----------
        base_input : Float[Tensor, "batch_size input_size"]
            The base input to the model
        source_inputs : Float[Tensor, "batch_size num_nodes input_size"]
            The source inputs to the model
        combined_activations : Float[Tensor, "batch_size 1+num_nodes total_space_size"]
            The activations of the model on the base input and each source input

        Returns
        -------
        output : Float[Tensor, "batch_size output_size"]
            The output of the low-level model on the base input and source inputs
        """

        # Get the patches to apply
        fwd_hooks = self.get_distributed_interchange_intervention_hooks(
            base_input, source_inputs, combined_activations
        )

        # Run the model with the patches applied, disabling gradients for the
        # model parameters, but not the rotation matrix used in the hooks
        self.low_level_model.requires_grad_(False)
        output = self.low_level_model.run_with_hooks(base_input, fwd_hooks=fwd_hooks)

        if self.output_modifier is not None:
            output = self.output_modifier(output)

        return output

    def run_interchange_intervention(
        self,
        base_input: Float[Tensor, "batch_size input_size"],
        source_inputs: Float[Tensor, "batch_size num_nodes input_size"],
        output_type: str = "output_nodes",
    ) -> Float[Tensor, "batch_size"]:
        """Run interchange intervention on the DAG

        Parameters
        ----------
        base_input : Float[Tensor, "batch_size input_size"]
            The base input to the model
        source_inputs : Float[Tensor, "batch_size num_nodes input_size"]
            The source inputs to the model
        output_type : str, optional
            The type of output to return, by default "output_nodes"

        Returns
        -------
        dag_output : Float[Tensor, "batch_size"]
            The output of the dag
        """

        # Convert the inputs into things the DAG can handle
        base_input_dag = self.input_alignment(base_input)
        source_inputs_dag = self.input_alignment(source_inputs)
        intervention_mask = {}
        for i, node in enumerate(self.dag_nodes):
            intervention_mask[node] = torch.zeros(
                (1, len(self.dag_nodes)), dtype=torch.bool
            )
            intervention_mask[node][0, i] = True

        # Run interchange intervention
        output = self.dag.run_interchange_intervention(
            base_input_dag,
            source_inputs_dag,
            intervention_mask,
            output_type=output_type,
            return_integers=True,
        )

        # We assume that the DAG only has one output node. Return its value
        output = list(output.values())[0].squeeze()
        return output

    def compute_activation_values(
        self,
        inputs: Float[Tensor, "num_inputs input_size"],
        batch_size=10000,
        activation_values_device: str | torch.device = "cpu",
    ) -> Float[Tensor, "num_inputs total_space_size"]:
        """Compute the activation vector for the given inputs

        Parameters
        ----------
        inputs : Float[Tensor, "num_inputs input_size"]
            The inputs to compute the activation vector for
        batch_size : int, default=10000
            The batch size to use for running the model.
        activation_values_device : torch.device or str, default="cpu"
            The device to store the activation values on

        Returns
        -------
        activation_values : Float[Tensor, "num_inputs total_space_size"]
            The activation vector for the given inputs
        """

        num_inputs = inputs.shape[0]

        def store_activation_value(
            value, hook: HookPoint, activation_values, space_index, batch_start
        ):
            activation_values[
                batch_start : batch_start + value.shape[0],
                space_index : space_index + value.shape[-1],
            ] = value.to(activation_values.device)

        activation_values = torch.empty(
            num_inputs,
            self.total_space_size,
            device=activation_values_device,
        )

        # Hooks to store the activation values.
        fwd_hooks = []
        for i, name in enumerate(self.intervene_model_hooks):
            fwd_hooks.append(
                (
                    name,
                    partial(
                        store_activation_value,
                        activation_values=activation_values,
                        space_index=sum(self.layer_sizes[:i]),
                    ),
                )
            )
        # TODO: The last hook could raise an exception which gets caught here,
        # to avoid uselessly running the rest of the model

        if self.verbosity > 1:
            print("Running model to determine activation values on source inputs...")

        # Compute them in batches
        iterator = range(0, num_inputs, batch_size)
        if self.verbosity >= 1 and self.progress_bar:
            iterator = tqdm(iterator, desc="Precomputing activation vectors")
        with torch.no_grad():
            for batch_start in iterator:
                batch_fwd_hooks = []
                for name, hook in fwd_hooks:
                    batch_fwd_hooks.append(
                        (name, partial(hook, batch_start=batch_start))
                    )
                batch_inputs = inputs[batch_start : batch_start + batch_size, ...]
                batch_inputs = batch_inputs.to(self.device)
                self.low_level_model.run_with_hooks(
                    batch_inputs,
                    fwd_hooks=batch_fwd_hooks,
                )

        return activation_values

    def create_interchange_intervention_dataset(
        self,
        inputs: Tensor,
        num_samples=10000,
        dag_batch_size=10000,
        dataset_device: str | torch.device = "cpu",
    ) -> InterchangeInterventionDataset:
        """Create a dataset of interchange intervention

        Samples `num_samples` instances of base and source inputs from
        `inputs`. Each sample consists of a random subset of the DAG nodes and
        random base and source inputs. It always returns the same number of
        source inputs, filling in for missing nodes with the base input.

        Parameters
        ----------
        inputs : Tensor of shape (num_inputs, input_size)
            The inputs to sample from
        num_samples : int, default=10000
            The number of samples to take
        dag_batch_size : int, default=10000
            The batch size to use when running interchange intervention on the DAG
        dataset_device : str or torch.device, default="cpu"
            The device to put the dataset on.

        Returns
        -------
        dataset : InterchangeInterventionDataset
            A dataset of interchange intervention samples, consisting of the
            base inputs, source inputs, and gold (DAG) outputs
        """

        if self.verbosity >= 1:
            print("Creating interchange intervention dataset...")

        num_inputs = inputs.shape[0]
        num_nodes = self.num_nodes

        # Make sure the inputs are on the right device
        inputs = inputs.to(dataset_device)

        # Compute the activation values for the inputs
        activation_values = self.compute_activation_values(
            inputs, activation_values_device=dataset_device
        )

        # Sample the base and source inputs
        base_input_indices = torch.randint(
            0, num_inputs, (num_samples,), device=dataset_device
        )
        source_input_indices = torch.randint(
            0, num_inputs, (num_samples, num_nodes), device=dataset_device
        )

        # Only intervene on some nodes with the source inputs; for the rest use the base
        # input
        node_selected_mask = torch.randint(
            0, 2, (num_samples, num_nodes), dtype=torch.bool, device=dataset_device
        )
        source_input_indices = torch.where(
            node_selected_mask,
            source_input_indices,
            base_input_indices.view(num_samples, 1),
        )

        base_inputs = inputs[base_input_indices]
        source_inputs = inputs[source_input_indices]

        gold_outputs = torch.empty(
            (num_samples,), dtype=torch.long, device=dataset_device
        )

        iterator = range(0, num_samples, dag_batch_size)
        if self.verbosity >= 1 and self.progress_bar:
            iterator = tqdm(iterator, desc="Computing gold outputs")
        for batch_start in iterator:
            dag_output = self.run_interchange_intervention(
                base_inputs[batch_start : batch_start + dag_batch_size].to(
                    self.dag.device
                ),
                source_inputs[batch_start : batch_start + dag_batch_size].to(
                    self.dag.device
                ),
                output_type="output_nodes",
            ).to(dataset_device)
            gold_outputs[batch_start : batch_start + dag_batch_size] = dag_output

        return InterchangeInterventionDataset(
            inputs=inputs,
            activation_values=activation_values,
            base_input_indices=base_input_indices,
            source_input_indices=source_input_indices,
            gold_outputs=gold_outputs,
        )

    def dii_training_objective_and_agreement(
        self,
        base_input: Float[Tensor, "batch_size input_size"],
        source_inputs: Float[Tensor, "batch_size num_nodes input_size"],
        combined_activations: Float[Tensor, "batch_size 1+num_nodes total_space_size"],
        gold_outputs: Float[Tensor, "batch_size"],
        loss_fn: callable = F.cross_entropy,
    ) -> tuple[Float[Tensor, ""], Float[Tensor, "batch_size"]]:
        """Compute the training objective and accuracy of DII

        Returns both the loss and a boolean for agreement between performing distributed
        interchange intervention on the low-level model and interchange intervention on
        the DAG.

        Parameters
        ----------
        base_input : Float[Tensor, "batch_size input_size"]
            The base input to the model
        source_inputs : Float[Tensor, "batch_size, num_nodes, input_size"]
            The source inputs to the model
        combined_activations : Float[Tensor, "batch_size 1+num_nodes total_space_size"]
            The combined activations of the base and source inputs
        gold_outputs : Float[Tensor, "batch_size"]
            The gold outputs
        loss_fn : callable, default=F.cross_entropy
            The loss function to use

        Returns
        -------
        loss : Float[Tensor, ""]
            The training objective output
        agreements : Float[Tensor, "batch_size"]
            Whether the low-level model and the DAG agree on the output after doing
            interventions
        """

        # Run distributed interchange intervention on the high-level model
        output_low_level = self.run_distributed_interchange_intervention(
            base_input, source_inputs, combined_activations
        )

        # Permute the output so that the class dimension is second
        output_low_level = einops.rearrange(output_low_level, "b ... c -> b c ...")

        # Compute the loss
        loss = loss_fn(output_low_level, gold_outputs)

        agreements = output_low_level.argmax(dim=1) == gold_outputs

        return loss, agreements

    def _run_epoch(
        self,
        ii_dataloader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        progress_bar_desc: Optional[str] = None,
    ) -> tuple[float, float]:
        """Run an epoch of training or evaluation

        Parameters
        ----------
        ii_dataloader : DataLoader
            The dataloader to use
        optimizer : torch.optim.Optimizer, optional
            The optimizer to use for training. If None, we will run evaluation instead
        progress_bar_desc : str, optional
            The description to use for the progress bar

        Returns
        -------
        loss : float
            The average loss over the epoch
        accuracy : float
            The average accuracy over the epoch
        """
        total_loss = 0.0
        total_agreement = 0.0

        iterator = ii_dataloader
        if self.progress_bar:
            iterator = tqdm(
                iterator,
                total=len(ii_dataloader),
                desc=progress_bar_desc,
            )
        for batch in iterator:
            base_input = batch[0].to(self.device)
            source_inputs = batch[1].to(self.device)
            combined_activations = batch[2].to(self.device)
            gold_outputs = batch[3].to(self.device)

            loss, agreements = self.dii_training_objective_and_agreement(
                base_input=base_input,
                source_inputs=source_inputs,
                combined_activations=combined_activations,
                gold_outputs=gold_outputs,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_agreement += agreements.float().mean().item()

        return total_loss / len(ii_dataloader), total_agreement / len(ii_dataloader)

    def train_rotation_matrix(
        self,
        inputs: Optional[Tensor] = None,
        ii_dataset: Optional[InterchangeInterventionDataset] = None,
        num_epochs: int = 10,
        lr: float = 0.01,
        batch_size: int = 32,
        num_samples: int = 10000,
        shuffle: bool = False,
    ) -> tuple[Float[np.ndarray, "num_epochs"], Float[np.ndarray, "num_epochs"]]:
        """Train the rotation matrix to align the two models

        Exactly one of `inputs` or `ii_dataset` must be provided. If `inputs` is
        provided, it will be used to create an interchange intervention dataset. If
        `ii_dataset` is provided, it will be used directly.

        Parameters
        ----------
        inputs : Tensor of shape (num_inputs, input_size), optional
            The input on which to train the rotation matrix. Base and source inputs will
            be samples from here
        ii_dataset : InterchangeInterventionDataset, optional
            The interchange intervention dataset to use
        num_epochs : int, default=0
            The number of epochs to train the rotation matrix for
        lr : float, default=0.01
            The learning rate to use
        batch_size : int, default=32
            The batch size to use
        num_samples : int, default=10000
            The number of samples to take from `inputs` to create the dataset, if
            `inputs` is provided.
        shuffle : bool, default=False
            Whether to shuffle the dataset when training

        Returns
        -------
        losses : np.ndarray of shape (num_steps,)
            The average loss at each epoch
        accuracies : np.ndarray of shape (num_steps,)
            At each epoch the accuracy of the alignment between the low-level model and
            the DAG
        """

        if inputs is None and ii_dataset is None:
            raise ValueError("Either `inputs` or `ii_dataset` must be provided")

        if inputs is not None and ii_dataset is not None:
            raise ValueError("Only one of `inputs` or `ii_dataset` can be provided")

        if ii_dataset is None:
            ii_dataset = self.create_interchange_intervention_dataset(
                inputs, num_samples=num_samples
            )

        if shuffle:
            sampler = RandomSampler(ii_dataset)
        else:
            sampler = SequentialSampler(ii_dataset)
        sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
        ii_dataloader = DataLoader(ii_dataset, sampler=sampler, batch_size=None)

        optimizer = torch.optim.SGD(self.rotation.parameters(), lr=lr)

        losses = np.empty(num_epochs)
        accuracies = np.empty(num_epochs)

        for epoch in range(num_epochs):
            losses[epoch], accuracies[epoch] = self._run_epoch(
                ii_dataloader, optimizer, f"Epoch [{epoch+1}/{num_epochs}]"
            )
            if self.verbosity >= 1:
                print(f"Loss: {losses[epoch]:0.5f}, Accuracy: {accuracies[epoch]:0.5f}")

        return losses, accuracies

    @torch.no_grad()
    def test_rotation_matrix(
        self,
        inputs: Optional[Tensor] = None,
        ii_dataset: Optional[TensorDataset] = None,
        batch_size: int = 32,
        num_samples: int = 1000,
    ) -> tuple[float, float]:
        """Test the rotation matrix for alignment between the two models

        Exactly one of `inputs` or `ii_dataset` must be provided. If `inputs`
        is provided, it will be used to create an interchange intervention
        dataset. If `ii_dataset` is provided, it will be used directly.

        Parameters
        ----------
        inputs : Tensor of shape (num_inputs, input_size), optional
            The input on which to train the rotation matrix. Base and source
            inputs will be samples from here
        ii_dataset : TensorDataset, optional
            The interchange intervention dataset to use
        batch_size : int, default=32
            The batch size to use
        num_samples : int, default=1000
            The number of samples to take from `inputs` to create the dataset,
            if `inputs` is provided.

        Returns
        -------
        loss : float
            The average loss
        accuracy : float
            The accuracy of the alignment between the low-level model and the
            DAG
        """

        if inputs is None and ii_dataset is None:
            raise ValueError("Either `inputs` or `ii_dataset` must be provided")

        if inputs is not None and ii_dataset is not None:
            raise ValueError("Only one of `inputs` or `ii_dataset` can be provided")

        if ii_dataset is None:
            if self.verbosity >= 1:
                print("Creating interchange intervention dataset...")
            ii_dataset = self.create_interchange_intervention_dataset(
                inputs, num_samples=num_samples
            )

        sampler = SequentialSampler(ii_dataset)
        sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
        ii_dataloader = DataLoader(ii_dataset, sampler=sampler, batch_size=None)

        loss, accuracy = self._run_epoch(ii_dataloader, None, "Testing")

        return loss, accuracy


class TransformerVariableAlignment(VariableAlignment):
    """A variable alignment between a recurrent DAG and a transformer

    Parameters
    ----------
    dag : RecurrentDeterministicDAG
        The DAG to align to the model
    low_level_model : HookedTransformer
        The hooked low-level model to align to the DAG
    dag_nodes : list[str]
        The names of the nodes (variables) of the DAG to align
    input_alignment : callable
        A function mapping (batches of) model input tensors to DAG input dictionaries
    output_alignment : callable
        A function mapping (batches of) model output tensors to DAG output values
    intervene_model_hooks : list[str]
        The names of the model hooks together give the whole activation space for
        alignment the DAG nodes
    subspaces_sizes : list[int]
        The sizes of the subspaces to use for each DAG nodes
    output_modifier : callable, optional
        A function to modify the output of the low-level model before comparing it with
        the DAG output
    device : Union[str, torch.device], default=None
        The device to put everything on. If None, use CUDA if available, otherwise CPU
    verbosity : int, default=1
        The verbosity level
    progress_bar : bool, default=True
        Whether to show a progress bar during and dataset generation
    """

    def _setup(self):
        super()._setup()
        self.d_vocab = self.low_level_model.cfg.d_vocab

    def _determine_layer_sizes(self) -> list[int]:
        """Run the model to determine the sizes of the activation spaces"""

        if self.verbosity > 0:
            print("Running model to determine activation space size...")

        sizes = []

        def counter_hook(value, hook: HookPoint, sizes):
            sizes.append(value.shape[-1])

        partial_counter_hook = partial(counter_hook, sizes=sizes)

        x = torch.zeros(1, 1, device=self.device, dtype=torch.long)
        fwd_hooks = [
            (name, partial_counter_hook) for name in self.intervene_model_hooks
        ]

        self.low_level_model.run_with_hooks(x, fwd_hooks=fwd_hooks)

        return sizes

    def get_distributed_interchange_intervention_hooks(
        self,
        base_input: Float[Tensor, "batch_size seq_len"],
        source_inputs: Float[Tensor, "batch_size num_nodes seq_len seq_len"],
        combined_activations: Float[
            Tensor, "batch_size 1+num_nodes*seq_len seq_len total_space_size"
        ],
    ) -> list[tuple[str, callable]]:
        """Get hooks for distributed interchange intervention

        Runs the model on the base input and each source input, and on input `i` records
        the value of the projection of the rotated activation space onto the `i`th
        subspace. It then builds hooks to patch in these values into the rotated
        activation space.

        Parameters
        ----------
        base_input : Float[Tensor, "batch_size seq_len"]
            The base input to the model
        source_inputs : Float[Tensor, "batch_size num_nodes seq_len seq_len"]
            The source inputs to the model, one for each batch element, DAG node and
            token position
        combined_activations : Float[Tensor, "batch_size 1+num_nodes*seq_len seq_len,
        total_space_size"]
            The activations of the low-level model on the base input and source inputs,
            combined

        Returns
        -------
        list[tuple[str, callable]]
            The hooks to use for distributed interchange intervention, for each of
            `self.intervene_model_hooks`
        """

        if source_inputs.shape[-3] != self.num_nodes:
            raise ValueError(
                f"Expected {self.num_nodes} source inputs, got {source_inputs.shape[-3]}"
            )

        batch_size = base_input.shape[0]
        seq_len = base_input.shape[1]

        base_input = base_input.to(self.device)
        source_inputs = source_inputs.to(self.device)

        # (batch_size * (1 + num_nodes * seq_len), seq_len, total_space_size)
        activation_values = combined_activations.view(
            -1, seq_len, self.total_space_size
        )

        # (batch_size * (1 + num_nodes * seq_len), seq_len, total_space_size)
        rotated_activation_values = self.rotation(activation_values)

        # Select the rotated activation values which are relevant, for patching in. We
        # select the indices of Y_0 for the base input across all positions at once, and
        # select indices of each Y_i for each position, since we patch in for each tuple
        # (node, stream) separately (these are dimensions 1 and 2 of `source_inputs`).
        super_batch_indices = torch.empty(
            (batch_size, seq_len, self.total_space_size), device=self.device, dtype=int
        )
        pos_indices = torch.arange(seq_len, device=self.device, dtype=int)
        pos_indices = pos_indices.view(1, -1, 1)
        space_indices = torch.arange(
            self.total_space_size, device=self.device, dtype=int
        )
        space_indices = space_indices.view(1, 1, -1)

        # Compute the indices for the super batch dimension (i.e. then batch_size * (1 +
        # num_nodes * seq_len) one). Select Y_0 across all positions, and each Y_i
        # for each position.
        sub_batch_size = 1 + self.num_nodes * seq_len
        super_batch_indices[:, :, : self.y_0_size] = (
            torch.arange(batch_size).view(batch_size, 1, 1) * sub_batch_size
        )
        space_index = self.y_0_size
        for i, subspace_size in enumerate(self.subspaces_sizes, start=1):
            super_batch_indices[:, :, space_index : space_index + subspace_size] = (
                torch.arange(batch_size).view(batch_size, 1, 1) * sub_batch_size
                + (i - 1) * seq_len
                + torch.arange(seq_len).view(1, seq_len, 1)
            )
            space_index += subspace_size

        # (batch_size, seq_len, total_space_size)
        new_rotated_activation_values = rotated_activation_values[
            super_batch_indices, pos_indices, space_indices
        ]

        # Unrotate the new rotated activation values, to get a vector
        # consisting of all the activation values to be patched in
        # (batch_size, seq_len, total_space_size)
        new_activation_values = self.rotation.inverse(new_rotated_activation_values)

        def intervention_hook(
            value,
            hook: HookPoint,
            space_index,
            new_activation_values,
        ):
            value[:] = new_activation_values[
                ..., space_index : space_index + value.shape[-1]
            ]
            return value

        fwd_hooks = []
        for i, name in enumerate(self.intervene_model_hooks):
            fwd_hooks.append(
                (
                    name,
                    partial(
                        intervention_hook,
                        space_index=sum(self.layer_sizes[:i]),
                        new_activation_values=new_activation_values,
                    ),
                )
            )

        return fwd_hooks

    def run_distributed_interchange_intervention(
        self,
        base_input: Float[Tensor, "batch_size seq_len"],
        source_inputs: Float[Tensor, "batch_size num_nodes seq_len seq_len"],
        combined_activations: Float[
            Tensor, "batch_size 1+num_nodes*seq_len seq_len total_space_size"
        ],
    ) -> Float[Tensor, "batch_size seq_len ..."]:
        """Compute the output of the low-level model on base and source inputs

        Does distributed interchange intervention, patching in the activation values
        from the source inputs, then running on the base input

        **Warning**: This method will disable gradients for the model parameters,
        setting `requires_grad` to `False`.

        Parameters
        ----------
        base_input : Float[Tensor, "batch_size seq_len"]
            The base input to the model
        source_inputs : Float[Tensor, "batch_size num_nodes seq_len seq_len"]
            The source inputs to the model
        combined_activations : Float[Tensor, "batch_size 1+num_nodes*seq_len seq_len
        total_space_size"]
            The activations of the low-level model on the base input and source inputs,
            combined

        Returns
        -------
        output : Float[Tensor, "batch_size seq_len ..."]
            The output of the low-level model (plus the output modifier, if set) on the
            base input and source inputs
        """

        return super().run_distributed_interchange_intervention(
            base_input, source_inputs, combined_activations
        )

    def run_interchange_intervention(
        self,
        base_input: Float[Tensor, "batch_size seq_len"],
        source_inputs: Float[Tensor, "batch_size num_nodes seq_len seq_len"],
        output_type: str = "output_nodes",
    ) -> Float[Tensor, "batch_size seq_len"]:
        """Run interchange intervention on the DAG

        Uses the vectorised runner if available.

        Parameters
        ----------
        base_input : Float[Tensor, "batch_size seq_len"]
            The base input to the model
        source_inputs : Float[Tensor, "batch_size num_nodes seq_len seq_len"]
            The source inputs to the model
        output_type : str, optional
            The type of output to return, by default "output_integer"

        Returns
        -------
        dag_output : Float[Tensor, "batch_size seq_len"]
            The output of the dag, as integers ranging over the possible
            output values
        """

        device = base_input.device

        batch_size = base_input.shape[0]
        seq_len = base_input.shape[1]

        # (batch_size, num_nodes * seq_len, seq_len)
        source_inputs = source_inputs.view(batch_size, -1, seq_len)

        # Convert the inputs into things the DAG can handle
        base_input_dag = self.input_alignment(base_input)
        source_inputs_dag = self.input_alignment(source_inputs)
        intervention_mask = {}
        for i, node in enumerate(self.dag_nodes):
            intervention_mask[node] = torch.zeros(
                (1, len(self.dag_nodes) * seq_len, seq_len),
                dtype=torch.bool,
                device=device,
            )
            intervention_mask[node][
                0, torch.arange(i * seq_len, (i + 1) * seq_len), torch.arange(seq_len)
            ] = True

        # Run interchange intervention
        output = self.dag.run_interchange_intervention(
            base_input_dag,
            source_inputs_dag,
            intervention_mask,
            output_type=output_type,
            return_integers=True,
        )

        # We assume that the DAG only has one output node. Return its value
        output = list(output.values())[0].squeeze()
        return output

    def dii_training_objective_and_agreement(
        self,
        base_input: Float[Tensor, "batch_size seq_len"],
        source_inputs: Float[Tensor, "batch_size num_nodes seq_len seq_len"],
        combined_activations: Float[
            Tensor, "batch_size 1+num_nodes*seq_len seq_len total_space_size"
        ],
        gold_outputs: Float[Tensor, "batch_size seq_len"],
        loss_fn: callable = F.cross_entropy,
        loss_mask: Optional[Bool[Tensor, "batch_size seq_len"]] = None,
    ) -> tuple[Float[Tensor, ""], Float[Tensor, "batch_size seq_len"]]:
        """Compute the training objective and accuracy of DII

        Returns both the loss and a boolean for agreement between performing distributed
        interchange intervention on the low-level model and interchange intervention on
        the DAG.

        Parameters
        ----------
        base_input : Float[Tensor, "batch_size seq_len"]
            The base input to the model
        source_inputs : Float[Tensor, "batch_size num_nodes seq_len seq_len"]
            The source inputs to the model
        combined_activations : Float[Tensor, "batch_size 1+num_nodes*seq_len seq_len
        total_space_size"]
            The activations of the low-level model on the base input and source inputs,
            combined
        gold_outputs : Float[Tensor, "batch_size seq_len"]
            The gold outputs
        loss_fn : callable, default=F.cross_entropy
            The loss function to use
        loss_mask : Bool[Tensor, "batch_size seq_len"], optional
            A mask to apply before computing the loss. By default, no mask is applied

        Returns
        -------
        loss : Float[Tensor, ""]
            The training objective output
        agreements : Float[Tensor, "batch_size seq_len"]
            Whether the low-level model and the DAG agree on the output after doing
            interventions
        """

        if loss_mask is None:
            loss_mask = torch.ones_like(gold_outputs, dtype=torch.bool)

        def masked_loss(input, target, loss_fn, loss_mask):
            loss_unreduced = loss_fn(input, target, reduction="none")
            loss_reduced = loss_unreduced[loss_mask].mean()
            return loss_reduced

        new_loss_fn = partial(masked_loss, loss_fn=loss_fn, loss_mask=loss_mask)

        return super().dii_training_objective_and_agreement(
            base_input=base_input,
            source_inputs=source_inputs,
            combined_activations=combined_activations,
            gold_outputs=gold_outputs,
            loss_fn=new_loss_fn,
        )

    def compute_activation_values(
        self,
        inputs: Float[Tensor, "num_inputs seq_len"],
        batch_size=32,
        activation_values_device: str | torch.device = "cpu",
    ) -> Float[Tensor, "num_inputs seq_len total_space_size"]:
        """Compute the activation vector for the given inputs

        Parameters
        ----------
        inputs : Tensor of shape (num_inputs, seq_len)
            The inputs to compute the activation vector for
        batch_size : int, default=32
            The batch size to use for running the model.
        activation_values_device : torch.device or str, default="cpu"
            The device to store the activation values on

        Returns
        -------
        activation_values : Tensor of shape (num_inputs, seq_len, total_space_size)
            The activation vector for the given inputs
        """

        num_inputs = inputs.shape[0]
        seq_len = inputs.shape[1]

        activation_values = torch.empty(
            (num_inputs, seq_len, self.total_space_size),
            dtype=torch.float32,
            device=activation_values_device,
        )

        def store_activation_value(
            value, hook: HookPoint, activation_values, space_index, batch_start
        ):
            activation_values[
                batch_start : batch_start + value.shape[0],
                :,
                space_index : space_index + value.shape[-1],
            ] = value.to(activation_values.device)

        # Hooks to store the activation values.
        fwd_hooks = []
        for i, name in enumerate(self.intervene_model_hooks):
            fwd_hooks.append(
                (
                    name,
                    partial(
                        store_activation_value,
                        activation_values=activation_values,
                        space_index=sum(self.layer_sizes[:i]),
                    ),
                )
            )

        # Compute them in batches
        iterator = range(0, num_inputs, batch_size)
        if self.verbosity >= 1 and self.progress_bar:
            iterator = tqdm(iterator, desc="Precomputing activation vectors")
        with torch.no_grad():
            for batch_start in iterator:
                batch_fwd_hooks = []
                for name, hook in fwd_hooks:
                    batch_fwd_hooks.append(
                        (name, partial(hook, batch_start=batch_start))
                    )
                batch_inputs = inputs[batch_start : batch_start + batch_size, ...]
                batch_inputs = batch_inputs.to(self.device)
                self.low_level_model.run_with_hooks(
                    batch_inputs,
                    fwd_hooks=batch_fwd_hooks,
                )

        return activation_values

    def create_interchange_intervention_dataset(
        self,
        inputs: Float[Tensor, "num_inputs seq_len"],
        num_samples=10000,
        batch_size=32,
        dag_batch_size=10000,
        dataset_device: str | torch.device = "cpu",
        pad_token_id: Optional[int] = None,
    ) -> TransformerInterchangeInterventionDataset:
        """Create a dataset of interchange intervention

        Samples `num_samples` instances of base and source inputs from `inputs`. Each
        sample consists of a random subset of the DAG nodes and streams and random base
        and source inputs. It always returns the same number of source inputs, filling
        in for missing nodes with the base input.

        Parameters
        ----------
        inputs : Float[Tensor, "num_inputs seq_len"]
            The inputs to sample from
        num_samples : int, default=10000
            The number of samples to take
        batch_size : int, default=32
            The batch size to use for running the low-level model.
        dag_batch_size : int, default=10000
            The batch size to use when running interchange intervention on the DAG
        dataset_device : str or torch.device, default="cpu"
            The device to put the dataset on.
        pad_token_id : int, optional
            The padding token ID to use. If not provided, use
            `low_level_model.tokenizer.pad_token_id`

        Returns
        -------
        dataset : TransformerInterchangeInterventionDataset
            A dataset of interchange intervention samples, which outputs the base
            inputs, source inputs, the activation vectors on the combined base and
            source inputs, and the gold (DAG) outputs
        """

        if self.verbosity >= 1:
            print("Creating interchange intervention dataset...")

        num_inputs = inputs.shape[0]
        seq_len = inputs.shape[1]
        num_nodes = self.num_nodes

        if pad_token_id is None:
            pad_token_id = self.low_level_model.tokenizer.pad_token_id

        # Make sure the inputs are on the right device
        inputs = inputs.to(dataset_device)

        activation_values = self.compute_activation_values(
            inputs, batch_size=batch_size, activation_values_device=dataset_device
        )

        # Set the number of DAG streams to be the same as the number of tokens (i.e.
        # `seq_len`)
        self.dag.num_streams = seq_len

        # (num_samples, )
        base_input_indices = torch.randint(
            0, num_inputs, (num_samples,), device=dataset_device
        )

        # (num_samples, num_nodes, seq_len)
        source_input_indices = torch.randint(
            0, num_inputs, (num_samples, num_nodes, seq_len), device=dataset_device
        )

        # (num_samples, num_nodes, seq_len)
        node_stream_selected_mask = torch.randint(
            0,
            2,
            (num_samples, num_nodes, seq_len),
            dtype=torch.bool,
            device=dataset_device,
        )

        source_input_indices = torch.where(
            node_stream_selected_mask,
            source_input_indices,
            base_input_indices.view(num_samples, 1, 1),
        )

        # (num_samples, seq_len)
        base_inputs = inputs[base_input_indices]

        # (num_samples, num_nodes, seq_len, seq_len)
        source_inputs = inputs[source_input_indices]

        gold_outputs = torch.empty(
            (num_samples, seq_len), dtype=torch.long, device=dataset_device
        )

        # Compute the gold outputs in batches
        iterator = range(0, num_samples, dag_batch_size)
        if self.verbosity >= 1 and self.progress_bar:
            iterator = tqdm(iterator, desc="Computing gold outputs")
        for batch_start in iterator:
            dag_output = self.run_interchange_intervention(
                base_inputs[batch_start : batch_start + dag_batch_size].to(
                    self.dag.device
                ),
                source_inputs[batch_start : batch_start + dag_batch_size].to(
                    self.dag.device
                ),
                output_type="output_nodes",
            ).to(dataset_device)
            gold_outputs[batch_start : batch_start + dag_batch_size, :] = dag_output

        # (num_samples, seq_len)
        loss_mask = base_inputs != pad_token_id
        # loss_mask = loss_mask & einops.reduce(
        #     source_inputs != pad_token_id,
        #     "num_samples num_nodes seq_len_1 seq_len_2 -> num_samples seq_len_2",
        #     reduction="min",
        # )

        return TransformerInterchangeInterventionDataset(
            inputs=inputs,
            activation_values=activation_values,
            base_input_indices=base_input_indices,
            source_input_indices=source_input_indices,
            gold_outputs=gold_outputs,
            loss_mask=loss_mask,
        )

    def _run_epoch(
        self,
        ii_dataloader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        progress_bar_desc: Optional[str] = None,
    ) -> tuple[float, float]:
        """Run an epoch of training or evaluation

        Parameters
        ----------
        ii_dataloader : DataLoader
            The dataloader to use
        optimizer : torch.optim.Optimizer, optional
            The optimizer to use for training. If None, we will run evaluation instead
        progress_bar_desc : str, optional
            The description to use for the progress bar

        Returns
        -------
        loss : float
            The average loss over the epoch
        accuracy : float
            The average accuracy over the epoch
        """
        total_loss = 0.0
        total_agreement = 0.0

        iterator = ii_dataloader
        if self.progress_bar:
            iterator = tqdm(
                iterator,
                total=len(ii_dataloader),
                desc=progress_bar_desc,
            )
        for batch in iterator:
            base_input = batch[0].to(self.device)
            source_inputs = batch[1].to(self.device)
            combined_activations = batch[2].to(self.device)
            gold_outputs = batch[3].to(self.device)
            loss_mask = batch[4].to(self.device)

            loss, agreements = self.dii_training_objective_and_agreement(
                base_input=base_input,
                source_inputs=source_inputs,
                combined_activations=combined_activations,
                gold_outputs=gold_outputs,
                loss_mask=loss_mask,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_agreement += agreements[loss_mask].float().mean().item()

        return total_loss / len(ii_dataloader), total_agreement / len(ii_dataloader)

    def train_rotation_matrix(
        self,
        inputs: Optional[Float[Tensor, "num_inputs seq_len"]] = None,
        ii_dataset: Optional[TransformerInterchangeInterventionDataset] = None,
        num_epochs: int = 10,
        lr: float = 0.01,
        batch_size: int = 32,
        num_samples: int = 10000,
        shuffle: bool = True,
    ) -> tuple[Float[np.ndarray, "num_epochs"], Float[np.ndarray, "num_epochs"]]:
        """Train the rotation matrix to align the two models

        Exactly one of `inputs` or `ii_dataset` must be provided. If `inputs`
        is provided, it will be used to create an interchange intervention
        dataset. If `ii_dataset` is provided, it will be used directly.

        Parameters
        ----------
        inputs : Tensor of shape (num_inputs, seq_len), optional
            The input on which to train the rotation matrix. Base and source
            inputs will be samples from here
        ii_dataset : TransformerInterchangeInterventionDataset, optional
            The interchange intervention dataset to use
        num_epochs : int, default=0
            The number of epochs to train the rotation matrix for
        lr : float, default=0.01
            The learning rate to use
        batch_size : int, default=32
            The batch size to use
        num_samples : int, default=10000
            The number of samples to take from `inputs` to create the dataset,
            if `inputs` is provided.
        shuffle : bool, default=False
            Whether to shuffle the dataset when training

        Returns
        -------
        losses : np.ndarray of shape (num_epochs,)
            The average loss at each epoch
        accuracies : np.ndarray of shape (num_epochs,)
            At each epoch the accuracy of the alignment between the low-level
            model and the DAG
        """

        return super().train_rotation_matrix(
            inputs=inputs,
            ii_dataset=ii_dataset,
            num_epochs=num_epochs,
            lr=lr,
            batch_size=batch_size,
            num_samples=num_samples,
            shuffle=shuffle,
        )

    @torch.no_grad()
    def test_rotation_matrix(
        self,
        inputs: Optional[Tensor] = None,
        ii_dataset: Optional[TensorDataset] = None,
        batch_size: int = 32,
        num_samples: int = 1000,
    ) -> tuple[float, float]:
        """Test the rotation matrix for alignment between the two models

        Exactly one of `inputs` or `ii_dataset` must be provided. If `inputs` is
        provided, it will be used to create an interchange intervention dataset. If
        `ii_dataset` is provided, it will be used directly.

        Parameters
        ----------
        inputs : Tensor of shape (num_inputs, seq_len), optional
            The input on which to train the rotation matrix. Base and source inputs will
            be samples from here
        ii_dataset : TensorDataset, optional
            The interchange intervention dataset to use
        batch_size : int, default=32
            The batch size to use
        num_samples : int, default=1000
            The number of samples to take from `inputs` to create the dataset, if
            `inputs` is provided.

        Returns
        -------
        loss : float
            The average loss
        accuracy : float
            The accuracy of the alignment between the low-level model and the DAG
        """

        return super().test_rotation_matrix(
            inputs=inputs,
            ii_dataset=ii_dataset,
            batch_size=batch_size,
            num_samples=num_samples,
        )
