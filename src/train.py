import os
import re
from typing import Callable
import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_model(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    tokenizer: Tokenizer,
    epochs=5,
    checkpoint_path='checkpoints',
):
    """Train a transformer model for sequence-to-sequence tasks.
    This function handles the training loop for a transformer model, including both training
    and validation phases, with support for learning rate scheduling and model checkpointing.
    Args:
        model (torch.nn.Module): The transformer model to train.
        train_dataloader (DataLoader): DataLoader for training data.
        test_dataloader (DataLoader): DataLoader for validation data.
        loss_fn (Callable): Loss function to optimize.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        tokenizer (Tokenizer): Tokenizer object for handling padding tokens.
        epochs (int, optional): Number of training epochs. Defaults to 5.
        checkpoint_path (str, optional): Directory to save model checkpoints. Defaults to 'checkpoints'.
    The function performs the following steps for each epoch:
        1. Training phase with loss computation and backpropagation
        2. Validation phase to evaluate model performance
        3. Model checkpoint saving
        4. Learning rate adjustment based on validation loss
    Returns:
        None
    Notes:
        - Uses CUDA if available, falls back to CPU otherwise
        - Implements teacher forcing during training
        - Uses dynamic learning rate scheduling with ReduceLROnPlateau
        - Creates attention masks for both source and target sequences
        - Saves model checkpoint after each epoch
    """

    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (src_batch, tgt_batch, src_mask) in tqdm(
            enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch+1}'
        ):
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            src_mask = src_mask.to(device)

            optimizer.zero_grad()

            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]

            tgt_padding_mask = (tgt_input != tokenizer.token_to_id('<pad>')).unsqueeze(1).unsqueeze(2)
            size = tgt_input.size(1)
            tgt_subsequent_mask = torch.triu(torch.ones((1, size, size), device=device), diagonal=1).bool()
            tgt_mask = tgt_padding_mask & ~tgt_subsequent_mask

            outputs = model(src_batch, tgt_input, src_mask, tgt_mask)
            outputs = outputs.view(-1, outputs.size(-1))
            tgt_output = tgt_output.contiguous().view(-1)

            loss = loss_fn(outputs, tgt_output)
            loss.backward()
            optimizer.step()
            lr_scheduler.step(loss.item())

            total_loss += loss.item()
        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch+1}, Average Training Loss {avg_loss:.4f}')

        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for src_batch, tgt_batch, src_mask in test_dataloader:
                src_batch = src_batch.to(device)
                tgt_batch = tgt_batch.to(device)
                src_mask = src_mask.to(device)

                tgt_input = tgt_batch[:, :-1]
                tgt_output = tgt_batch[:, 1:]

                tgt_padding_mask = (tgt_input != tokenizer.token_to_id('<pad>')).unsqueeze(1).unsqueeze(2)
                size = tgt_input.size(1)
                tgt_subsequent_mask = torch.triu(torch.ones((1, size, size), device=device), diagonal=1).bool()
                tgt_mask = tgt_padding_mask & ~tgt_subsequent_mask

                outputs = model(src_batch, tgt_input, src_mask, tgt_mask)
                outputs = outputs.view(-1, outputs.size(-1))
                tgt_output = tgt_output.contiguous().view(-1)

                loss = loss_fn(outputs, tgt_output)
                total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(test_dataloader)
            print(f'Epoch {epoch+1}, Average Validation Loss {avg_val_loss:.4f}')

        save_model(model, f'{checkpoint_path}/epoch_{epoch+1}.pth')


def save_model(model: torch.nn.Module, path: str):
    """
    Save a PyTorch model's state dictionary to the specified path.

    This function saves the model's state dictionary to a file, creating the necessary
    directories if they don't exist.

    Args:
        model (torch.nn.Module): The PyTorch model to save
        path (str): The file path where the model should be saved

    Returns:
        None

    Examples:
        >>> model = MyModel()
        >>> save_model(model, 'models/my_model.pth')
        Model saved: models/my_model.pth
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f'Model saved: {path}')


def load_model(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """Loads a PyTorch model's state dictionary from a file.
    Args:
        model (torch.nn.Module): The model architecture to load weights into.
        path (str): Path to the saved model state dictionary file.
    Returns:
        torch.nn.Module: The model with loaded weights.
    Example:
        >>> model = MyModel()
        >>> model = load_model(model, 'path/to/weights.pth')
    """

    model.load_state_dict(torch.load(path))
    return model


def load_latest_model(model: torch.nn.Module, checkpoint_dir: str) -> torch.nn.Module:
    """Load the most recent model checkpoint from the given directory.

    Args:
        model: The model to load the state into
        checkpoint_dir: Directory containing checkpoint files named 'epoch_X.pth'

    Returns:
        The model with loaded state
    """
    # Find checkpoint files and extract epoch numbers
    checkpoint_pattern = re.compile(r'epoch_(\d+)\.pth')
    epoch_numbers = [
        int(match.group(1)) for file in os.listdir(checkpoint_dir) if (match := checkpoint_pattern.match(file))
    ]

    if not epoch_numbers:
        raise FileNotFoundError(f'No checkpoint files found in {checkpoint_dir}')

    latest_epoch = max(epoch_numbers)

    return load_model(model, f'{checkpoint_dir}/epoch_{latest_epoch}.pth')
