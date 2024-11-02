# Transformer Implementation

This directory contains an implementation of the Transformer model from the paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) using PyTorch.

## Overview

The Transformer is a neural network architecture designed for sequence-to-sequence tasks, removing the reliance on recurrence in favor of attention mechanisms. This implementation focuses on machine translation from German to English using the WMT14 dataset.

## Repository Structure

- `src/transformer.py`: Contains the model definition of the Transformer, including encoder, decoder, and attention mechanisms.
- `src/train.py`: Script to train the Transformer model.
- `data/`: Directory to store datasets, tokenizer files, and checkpoints.
- `paper.pdf`: Copy of the original paper for reference.

## Prerequisites

Make sure to have followed the installation and setup instructions presented in the [main README](../../README.md) of this repository.

## Dataset Preparation

The model uses the WMT14 German-English dataset. The `datasets` library handles downloading and preprocessing:

- The dataset is automatically downloaded when running `train.py`.
- Source sentences are in German (`'de'`), and target sentences are in English (`'en'`).

## Training the Model

To train the Transformer model, run:

```sh
python -m papers.attention_is_all_you_need.src.train
```

Training includes:

- Loading and preprocessing the dataset.
- Training a Byte-Pair Encoding (BPE) tokenizer using the training data.
- Initializing the Transformer model with specified hyperparameters.
- Training the model with checkpoints saved in `data/checkpoints`.

### Hyperparameters

All Hyperparameters were chosen to match the original papers base model. The hyperparameters can all be adjusted in the creation of the `Transformer` object and for evaluation in the `train.py` file.

## Evaluation

Evaluation is performed during training using a validation set:

- Average validation loss is printed after each epoch.
- Future work may include BLEU score calculation for more comprehensive evaluation.

## Reproducing the Paper Results

While this implementation aims to replicate the Transformer model, the complete training and evaluation results presented in the paper were not reproduced due to computational constraints. The model was trained for a limited number of epochs to demonstrate functionality and ensure correctness.

## Notes

- Checkpoints are saved after each epoch in `data/checkpoints`.
- The tokenizer is saved to `data/tokenizer.json` after training.
- Ensure sufficient computational resources due to the model's size.

## References

- Vaswani, A., et al. (2017). [Attention is All You Need](https://arxiv.org/abs/1706.03762). *Advances in Neural Information Processing Systems*. (Also included as [paper.pdf](paper.pdf) here in the repo.)
