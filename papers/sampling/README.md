# Sampling Strategies in Language Generation

This directory contains an implementation of various sampling strategies used in language generation models, focusing on methods like greedy decoding, top-k sampling, nucleus (top-p) sampling, beam search, and structured output sampling.

## Overview

In language generation tasks, selecting the next word in a sequence is crucial for generating coherent and contextually appropriate text. Different sampling strategies can significantly affect the quality and diversity of the generated output. This implementation demonstrates how these strategies can be applied to a decoder-only Transformer model.

### Implemented Sampling Methods

- **Greedy Decoding**: Selects the word with the highest probability at each step, which may lead to repetitive or uninteresting text.

- **Top-K Sampling**: Limits the selection of the next word to the top K most probable words, allowing for more varied outputs.

- **Nucleus (Top-P) Sampling**: Considers the smallest possible set of words whose cumulative probability exceeds a threshold P, offering a balance between diversity and coherence.

- **Beam Search**: Keeps track of the top sequences at each step, expanding them to find the most probable overall sequence, often resulting in more coherent but less diverse text.

- **Structured Output Sampling**: This method allows you to enforce a specific structure in the generated text by filtering the model's logits before sampling. This ensures that the output adheres to predefined formatting constraints, which is particularly useful when generating text that must follow a certain template or protocol. The structured output filtering method modifies the logits output by the model at each time step to enforce the desired structure.

- **Structured JSON Output Sampling**: Uses a `StreamingJSONValidator` to enforce schema-conforming JSON during generation. It binds to Pydantic models, generating only text that will deserialize correctly. This ensures schema-compliant outputs without post-hoc correction. See `structured_json_output.py` and `schema_validator.py`.

```
Pydantic model  →  internal schema  →  grammar-constrained sampler → JSON text (valid for the schema)  →  Pydantic model instance
```

- **Schema Validation**: The `schema_validator.py` file implements an incremental JSON validator that supports per-character validation against a schema. This allows for real-time feedback on the validity of the generated text as it is being produced, ensuring that the output adheres to the specified schema.

- **Fast JSON Output sampling**: The `fast_structured_json_output.py` file implements a fast JSON output sampling method that uses the fact that the JSON structure is known in advance and just the values need to be sampled. This method is faster by orders of magnitude than the structured JSON output sampling method, as it does not require the overhead of validating the entire JSON structure at each step, does not require sampling the JSON structure itself from the model and only needs fast sampling of primitive values (strings, numbers, booleans) for which the allowed tokens can be precomputed once. Therefore, we need to sample less, dont need to validate and do not need to compute the allowed tokens at each step.

## Repository Structure

- `src/`: Source code for the sampling strategies.
  - `decoder_only_transformer.py`: Contains the model definition of the decoder-only Transformer.
  - `sampling.py`: Main script that initializes the model and tests the sampling methods.
  - `strategies/`: Directory containing separate files for each sampling strategy.
    - `greedy.py`: Implementation of greedy decoding.
    - `top_k.py`: Implementation of top-K sampling.
    - `top_p.py`: Implementation of nucleus (top-P) sampling.
    - `beam_search.py`: Implementation of beam search.
    - `structured_output.py`: Implementation of the structured output filtering method.
    - `structured_json_output.py`: Pydantic-driven structured generation using the StreamingJSONValidator.
    - `fast_structured_json_output.py`: Pythonic-driven fast structured generation based on optimizations described above.
    - `schema_validator.py`: Incremental JSON validator supporting per-character validation against schema.
- `data/`: Directory for tokenizer files and any required datasets.

## Prerequisites

Ensure you have followed the installation and setup instructions in the [main README](../../README.md) of this repository.

## Running the Code

To test the different sampling strategies, run the `sampling.py` script:

```sh
python -m papers.sampling.src.sampling
```

To try structured JSON sampling based on a Pydantic model:

```sh
python -m papers.sampling.src.strategies.structured_json_output
```

## Notes

- The model used is a decoder-only Transformer similar to GPT models.
- Adjust the hyperparameters and sampling parameters in the script to experiment with different settings.
- The structured output sampling demonstrates how to guide the model's output format without retraining.
- The structured JSON sampling (`structured_json_output.py`) is a powerful, grammar-constrained generation method, but **may be slow for string fields** due to high token branching. Optimization suggestions include:
  - Compiling schema rules to more efficient forms.
  - Parallelizing token filtering (easy, embarrassingly parallel).
  - Implementing the filtering logic in C++/Rust/CUDA for GPU acceleration.

## References

- For a detailed explanation of these sampling methods, you can refer to resources like [The Art and Science of Sampling in Language Generation](https://huyenchip.com/2024/01/16/sampling.html).
