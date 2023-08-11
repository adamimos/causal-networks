from copy import deepcopy
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import einops

import numpy as np

import pandas as pd


from tqdm import tqdm

from transformer_lens import HookedTransformer


def fine_tune_paren_bal(
    base_model_name: str,
    text_dataset_file: str,
    final_model: bool = False,
    device: torch.device | str = "cuda",
    open_paren_str_tokens: list = ["("],
    close_paren_str_tokens: list = [")"],
    test_dataset_size: float = 0.1,
    validation_dataset_size: float = 0.1,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    num_epochs: int = 10,
    lr_scheduler_patience: int = 1000,
    optimizer_name: str = "Adam",
    seed: int = 2384,
) -> tuple[dict[str, np.ndarray], HookedTransformer]:
    #####################
    # Set the random seed
    #####################

    torch.manual_seed(seed)
    np.random.seed(seed)

    #####################
    # Load the model
    #####################

    print("Loading model...")

    base_model = HookedTransformer.from_pretrained(base_model_name, device=device)

    model_config = deepcopy(base_model.cfg)
    model_config.d_vocab_out = 2

    model = HookedTransformer(cfg=model_config)

    model_state_dict = deepcopy(base_model.state_dict())
    model_state_dict["unembed.W_U"] = torch.empty(
        model_config.d_model, model_config.d_vocab_out
    ).to(device)
    model_state_dict["unembed.b_U"] = torch.empty(model_config.d_vocab_out).to(device)
    nn.init.uniform_(
        model_state_dict["unembed.W_U"],
        -1 / sqrt(model_config.d_model),
        1 / sqrt(model_config.d_model),
    )
    nn.init.uniform_(
        model_state_dict["unembed.b_U"],
        -1 / sqrt(model_config.d_model),
        1 / sqrt(model_config.d_model),
    )

    model.load_state_dict(model_state_dict)

    pad_token_id = base_model.tokenizer.pad_token_id

    #####################
    # Create the dataset
    #####################

    print("Creating dataset...")

    open_paren_tokens = model.to_tokens(
        open_paren_str_tokens, prepend_bos=False, move_to_device=False, truncate=False
    )
    open_paren_tokens = [
        open_paren_tokens[i, 0].item() for i in range(open_paren_tokens.shape[0])
    ]

    closed_paren_tokens = model.to_tokens(
        close_paren_str_tokens, prepend_bos=False, move_to_device=False, truncate=False
    )
    closed_paren_tokens = [
        closed_paren_tokens[i, 0].item() for i in range(closed_paren_tokens.shape[0])
    ]

    text_data = pd.read_csv(text_dataset_file)
    text_data_tokenised = base_model.to_tokens(
        text_data["text"].values, move_to_device=False
    )

    open_bracket = torch.isin(text_data_tokenised, torch.tensor(open_paren_tokens))
    closed_bracket = torch.isin(text_data_tokenised, torch.tensor(closed_paren_tokens))
    bracket_values = torch.zeros_like(text_data_tokenised, dtype=torch.long)
    bracket_values = bracket_values + open_bracket.long() - closed_bracket.long()
    cumsum = torch.cumsum(bracket_values, dim=-1)
    output_data = (cumsum > 0).to(dtype=torch.long)

    loss_mask = text_data_tokenised != pad_token_id

    shuffled_indices = torch.randperm(text_data_tokenised.shape[0])
    train_indices = shuffled_indices[
        : int(
            text_data_tokenised.shape[0]
            * (1 - test_dataset_size - validation_dataset_size)
        )
    ]
    validation_indices = shuffled_indices[
        int(
            text_data_tokenised.shape[0]
            * (1 - test_dataset_size - validation_dataset_size)
        ) : int(text_data_tokenised.shape[0] * (1 - test_dataset_size))
    ]
    test_indices = shuffled_indices[
        int(text_data_tokenised.shape[0] * (1 - test_dataset_size)) :
    ]

    train_dataset = TensorDataset(
        text_data_tokenised[train_indices],
        output_data[train_indices],
        loss_mask[train_indices],
    )
    validation_dataset = TensorDataset(
        text_data_tokenised[validation_indices],
        output_data[validation_indices],
        loss_mask[validation_indices],
    )
    test_dataset = TensorDataset(
        text_data_tokenised[test_indices],
        output_data[test_indices],
        loss_mask[test_indices],
    )

    #####################
    # Train the model
    #####################

    print("Training model...")

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    model.train()

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer name {optimizer_name}")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=lr_scheduler_patience, verbose=True
    )
    losses = np.empty(num_epochs)
    accuracies = np.empty(num_epochs)

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_agreement = 0.0

        iterator = tqdm(
            train_dataloader,
            total=len(train_dataloader),
            desc=f"Epoch {epoch + 1}/{num_epochs}",
        )
        for tokens, gold_output, loss_mask in iterator:
            tokens = tokens.to(device)
            gold_output = gold_output.to(device)
            loss_mask = loss_mask.to(device)

            optimizer.zero_grad()

            output = model(tokens)
            output_rearranged = einops.rearrange(
                output, "batch seq out -> batch out seq"
            )
            loss = F.cross_entropy(output_rearranged, gold_output, reduction="none")
            loss = loss[loss_mask].mean()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            total_loss += loss.item()
            with torch.no_grad():
                total_agreement += (
                    (torch.argmax(output, dim=-1) == gold_output)[loss_mask]
                    .float()
                    .mean()
                    .item()
                )

        losses[epoch] = total_loss / len(train_dataloader)
        accuracies[epoch] = total_agreement / len(train_dataloader)

        print(f"Loss: {losses[epoch]:.4f}, Accuracy: {accuracies[epoch]:.4f}")

    #####################
    # Evaluate the model
    #####################

    print("Evaluating model...")

    model.eval()

    total_agreement = 0.0

    if final_model:
        dataloader = test_dataloader
    else:
        dataloader = validation_dataloader

    iterator = tqdm(
        dataloader,
        total=len(dataloader),
        desc="Evaluating model"
    )
    with torch.no_grad():
        for tokens, gold_output, loss_mask in iterator:
            tokens = tokens.to(device)
            gold_output = gold_output.to(device)
            loss_mask = loss_mask.to(device)
            output = model(tokens)
            total_agreement += (
                (torch.argmax(output, dim=-1) == gold_output)[loss_mask]
                .float()
                .mean()
                .item()
            )
    test_accuracy = total_agreement / len(validation_dataloader)

    if final_model:
        print(f"Test accuracy: {test_accuracy:.4f}")
    else:
        print(f"Validation accuracy: {test_accuracy:.4f}")

    #####################
    # Return the results
    #####################

    results = {
        "train_losses": losses,
        "train_accuracies": accuracies,
    }
    if final_model:
        results["test_accuracy"] = test_accuracy
    else:
        results["val_accuracy"] = test_accuracy

    return results, model
