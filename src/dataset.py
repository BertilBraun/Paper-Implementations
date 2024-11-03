import os
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

PAD_TOKEN = '<pad>'
BOS_TOKEN = '<bos>'
END_OF_SRC_TOKEN = '<eos>'
END_OF_TGT_TOKEN = '</eos>'
UNK_TOKEN = '<unk>'


def get_tokenizer(
    train_dataset: Dataset,
    path='tokenizer.json',
    vocab_size=37000,
    min_frequency=2,
    special_tokens=[PAD_TOKEN, END_OF_SRC_TOKEN, END_OF_TGT_TOKEN, BOS_TOKEN, UNK_TOKEN],
) -> Tokenizer:
    """Train and return a BPE tokenizer for text processing.
    This function either loads a pre-trained tokenizer from a file or trains a new one
    using the Byte-Pair Encoding (BPE) algorithm on the provided dataset.
    Args:
        train_dataset (Dataset): A dataset containing 'src' and 'tgt' fields, where:
            - 'src': Source text sequences
            - 'tgt': Target text sequences
            Both fields should be lists or arrays of text strings.
        path (str, optional): Path to save/load the tokenizer. Defaults to 'tokenizer.json'.
        vocab_size (int, optional): Maximum vocabulary size. Defaults to 37000.
        min_frequency (int, optional): Minimum frequency for a token to be included. Defaults to 2.
        special_tokens (list, optional): List of special tokens to be added to the vocabulary.
            Defaults to ['<pad>', '<eos>', '</eos>', '<bos>', '<unk>'].
    Returns:
        Tokenizer: A trained tokenizer object with padding enabled.
    Notes:
        - The training process uses batches of 1000 sequences combining both source and target texts.
        - The tokenizer uses whitespace pre-tokenization.
        - Padding is automatically enabled with '<pad>' token after loading/training.
    """

    if not os.path.exists(path):
        print('Training tokenizer...')
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()  # type: ignore
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,  # type: ignore
            min_frequency=min_frequency,  # type: ignore
            special_tokens=special_tokens,  # type: ignore
        )

        def batch_iterator():
            for i in range(0, len(train_dataset), 1000):
                batch = train_dataset[i : i + 1000]['src'] + train_dataset[i : i + 1000]['tgt']
                yield batch

        tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
        tokenizer.save(path)

    return load_tokenizer(path)


def load_tokenizer(path='tokenizer.json') -> Tokenizer:
    """Load a BPE tokenizer for text processing.
    This function loads a pre-trained tokenizer from a file.
    Args:
        path (str, optional): Path to save/load the tokenizer. Defaults to 'tokenizer.json'.
    Returns:
        Tokenizer: A trained tokenizer object with padding enabled.
    """

    assert os.path.exists(path), f'Tokenizer file not found at {path}'

    tokenizer: Tokenizer = Tokenizer.from_file(path)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id(PAD_TOKEN), pad_token=PAD_TOKEN)
    return tokenizer


def prepare_dataloader(
    dataset: Dataset,
    tokenizer: Tokenizer,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    """Prepares a DataLoader for training/evaluation by tokenizing and batching the dataset.
    This function processes a raw dataset by tokenizing the source and target texts, and creates
    a DataLoader with proper collation function that handles padding and masking.
    Args:
        dataset (Dataset): The input dataset containing 'src' and 'tgt' text pairs
        tokenizer (Tokenizer): Tokenizer instance used to encode the text
        batch_size (int, optional): Batch size for the DataLoader. Defaults to 64
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True
    Returns:
        DataLoader: A DataLoader instance with the processed dataset, providing batches of:
            - src_batch: Padded source sequences tensor of shape (batch_size, max_src_len)
            - tgt_batch: Padded target sequences tensor of shape (batch_size, max_tgt_len)
    Example:
        >>> dataloader = prepare_dataloader(dataset, tokenizer, batch_size=32)
        >>> src_batch, tgt_batch = next(iter(dataloader))
    """

    def encode(example, tokenizer: Tokenizer):
        example['src'] = tokenizer.encode(BOS_TOKEN + example['src'] + END_OF_SRC_TOKEN).ids
        example['tgt'] = tokenizer.encode(BOS_TOKEN + example['tgt'] + END_OF_TGT_TOKEN).ids
        return example

    def collate_fn(batch, tokenizer):
        src_batch = [torch.tensor(item['src']) for item in batch]
        tgt_batch = [torch.tensor(item['tgt']) for item in batch]

        src_batch = pad_sequence(src_batch, padding_value=tokenizer.token_to_id(PAD_TOKEN), batch_first=True)
        tgt_batch = pad_sequence(tgt_batch, padding_value=tokenizer.token_to_id(PAD_TOKEN), batch_first=True)

        return src_batch, tgt_batch

    print('Tokenizing dataset...')
    dataset = dataset.map(lambda x: encode(x, tokenizer), batched=False)

    print('Creating DataLoader...')
    dataloader = DataLoader(
        dataset,  # type: ignore
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: collate_fn(x, tokenizer),
    )

    return dataloader
