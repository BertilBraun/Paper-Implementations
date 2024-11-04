# Sampling Strategies in Language Generation

This directory contains an implementation of various sampling strategies used in language generation models, focusing on methods like greedy decoding, top-k sampling, nucleus (top-p) sampling, and beam search.

## Overview

In language generation tasks, selecting the next word in a sequence is crucial for generating coherent and contextually appropriate text. Different sampling strategies can significantly affect the quality and diversity of the generated output. This implementation demonstrates how these strategies can be applied to a decoder-only Transformer model.

### Implemented Sampling Methods

- **Greedy Decoding**: Selects the word with the highest probability at each step, which may lead to repetitive or uninteresting text.

- **Top-K Sampling**: Limits the selection of the next word to the top K most probable words, allowing for more varied outputs.

- **Nucleus (Top-P) Sampling**: Considers the smallest possible set of words whose cumulative probability exceeds a threshold P, offering a balance between diversity and coherence.

- **Beam Search**: Keeps track of the top sequences at each step, expanding them to find the most probable overall sequence, often resulting in more coherent but less diverse text.

## Repository Structure

- `src/`: Source code for the sampling strategies.
  - `decoder_only_transformer.py`: Contains the model definition of the decoder-only Transformer.
  - `sampling.py`: Contains the sampling methods and testing code.
- `data/`: Directory for tokenizer files and any required datasets.

## Prerequisites

Ensure you have followed the installation and setup instructions in the [main README](../../README.md) of this repository.

## Running the Code

To test the different sampling strategies, run the `sampling.py` script:

```sh
python -m papers.sampling.src.sampling
```

The script initializes the model and tokenizer, then generates text using each of the implemented sampling methods. The generated text for each strategy will be printed to the console.

## Notes

- The model used is a decoder-only Transformer similar to GPT models.
- Adjust the hyperparameters and sampling parameters in the script to experiment with different settings.

## References

- For a detailed explanation of these sampling methods, you can refer to resources like [The Art and Science of Sampling in Language Generation](https://huyenchip.com/2024/01/16/sampling.html).
