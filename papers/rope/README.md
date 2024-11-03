# RoFormer: Enhanced Transformer with Rotary Position Embedding

This directory contains an implementation of the RoFormer model from the paper ["RoFormer: Enhanced Transformer with Rotary Position Embedding"](https://arxiv.org/pdf/2104.09864) using PyTorch.

## Overview

The RoFormer introduces Rotary Position Embedding (RoPE) to the Transformer architecture, enhancing the model's ability to capture positional information without explicit sinusoidal positional encodings. This implementation modifies the original Transformer model by adjusting the scaled dot-product attention mechanism to incorporate RoPE. The rest of the architecture remains largely the same as in the "Attention is All You Need" paper.

## Repository Structure

- `src/rope.py`: Contains the model definition of the RoFormer, including the adjusted attention mechanism with RoPE.
- `paper.pdf`: Copy of the original paper for reference.

## Prerequisites

Ensure you have followed the installation and setup instructions presented in the [main README](../../README.md) of this repository.

## Training and Evaluating the Model

Since the RoFormer model is an extension of the Transformer, the training and evaluation procedures are similar to the original Transformer implementation. Please refer to the training and evaluation instructions in the [Transformer implementation](../Attention_is_all_you_need/README.md#training-the-model). Simply replace the Transformer model with the RoPETransformer model in the training script.

## Notes

- The primary modification from the original Transformer is in the `RoPEScaledDotProductAttention` class within `rope.py`.

## References

- Su, J., et al. (2021). [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/pdf/2104.09864). (Also included as [paper.pdf](paper.pdf) here in the repo.)
