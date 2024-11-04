# Transformer with KV Cache Implementation

This directory contains an implementation of a decoder-only Transformer model with Key-Value (KV) Cache, based on the paper ["KV Cache"](https://arxiv.org/pdf/2211.05102).

## Overview

The KV Cache accelerates inference by caching the computed key and value tensors from previous decoder steps, eliminating the need to recompute them at each time step. This optimization reduces redundant computations and speeds up autoregressive generation tasks.

This implementation is built upon the base Transformer model from the "Attention is All You Need" paper, adapting the decoder to utilize the KV Cache. The model expects one token at a time during inference, but the `generate.py` script abstracts this limitation, enabling sequence generation.

## Repository Structure

- `src/`: Source code for the KV Cache implementation.
  - `kv_cache.py`: Contains the model definition with KV Cache.
  - `generate.py`: Script for sequence generation using the KV Cache.
- `paper.pdf`: The original paper for reference.

## Prerequisites

Ensure you have followed the installation and setup instructions in the [main README](../../README.md) of this repository.

## Limitations

- The model expects one token at a time during inference.
- Currently, there's no training or evaluation setup provided.

## TODO

- Implement training and evaluation scripts.
- Optimize the KV Cache for batch processing.
- Explore different model configurations and document performance.

## References

- [KV Cache](https://arxiv.org/pdf/2211.05102). (Also included as [paper.pdf](paper.pdf) in this directory.)
