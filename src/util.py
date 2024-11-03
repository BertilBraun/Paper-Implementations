import os
import re
from tokenizers import Tokenizer
import torch

from src.dataset import PAD_TOKEN


def create_src_mask(src_batch: torch.Tensor, tokenizer: Tokenizer, device: torch.device) -> torch.Tensor:
    src_mask = (src_batch != tokenizer.token_to_id(PAD_TOKEN)).unsqueeze(1).unsqueeze(2).to(device)
    return src_mask


def create_tgt_mask(tgt_input: torch.Tensor, tokenizer: Tokenizer, device: torch.device) -> torch.Tensor:
    tgt_padding_mask = (tgt_input != tokenizer.token_to_id(PAD_TOKEN)).unsqueeze(1).unsqueeze(2).to(device)
    size = tgt_input.size(1)
    tgt_subsequent_mask = torch.triu(torch.ones((1, size, size), device=device), diagonal=1).bool()
    tgt_mask = tgt_padding_mask & ~tgt_subsequent_mask
    return tgt_mask


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
