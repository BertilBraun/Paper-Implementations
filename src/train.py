import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from typing import Callable

from src.util import create_src_mask, create_tgt_mask, save_model


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
        for src_batch, tgt_batch in tqdm(train_dataloader, total=len(train_dataloader), desc=f'Epoch {epoch+1}'):
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            optimizer.zero_grad()

            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]

            src_mask = create_src_mask(src_batch, tokenizer, device)
            tgt_mask = create_tgt_mask(tgt_input, tokenizer, device)

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

        avg_val_loss = evaluate(model, test_dataloader, loss_fn, tokenizer, device)
        print(f'Epoch {epoch+1}, Average Validation Loss {avg_val_loss:.4f}')

        save_model(model, f'{checkpoint_path}/epoch_{epoch+1}.pth')


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable,
    tokenizer: Tokenizer,
    device: torch.device,
) -> float:
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for src_batch, tgt_batch in dataloader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]

            src_mask = create_src_mask(src_batch, tokenizer, device)
            tgt_mask = create_tgt_mask(tgt_input, tokenizer, device)

            outputs = model(src_batch, tgt_input, src_mask, tgt_mask)
            outputs = outputs.view(-1, outputs.size(-1))
            tgt_output = tgt_output.contiguous().view(-1)

            loss = loss_fn(outputs, tgt_output)
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(dataloader)
    return avg_val_loss
