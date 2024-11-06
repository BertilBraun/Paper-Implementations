from typing import Literal
import torch
import tiktoken
from tiktoken_ext.openai_public import ENDOFTEXT
from torch import nn
from src.util import load_latest_model
from papers.sampling.src.decoder_only_transformer import DecoderOnlyTransformer, ModelConfig
from papers.sampling.src.strategies.top_k import top_k_sampling
from papers.sampling.src.strategies.top_p import top_p_sampling
from papers.sampling.src.strategies.beam_search import beam_search
from papers.sampling.src.strategies.structured_output import structured_sampling


def generate_sequence(
    model: nn.Module,
    tokenizer: tiktoken.Encoding,
    start_sequence: str,
    max_length: int = 50,
    strategy: Literal['greedy', 'top-k', 'top-p', 'beam', 'structured'] = 'greedy',
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    beam_width: int = 5,
) -> str:
    """
    Generates a sequence using the specified sampling strategy.

    Args:
        model: The decoder-only Transformer model.
        tokenizer: The tokenizer for encoding and decoding text.
        start_sequence: The initial text to start generation.
        max_length: The maximum length of the generated sequence.
        strategy: The sampling strategy ('greedy', 'top-k', 'top-p', 'beam', 'structured').
        temperature: Temperature for sampling.
        top_k: Top-K value for Top-K sampling.
        top_p: Top-P value for nucleus (Top-P) sampling.
        beam_width: Beam width for beam search.

    Returns:
        The generated text sequence.
    """
    model.eval()
    device = next(model.parameters()).device

    # Encode the start sequence
    generated = tokenizer.encode(start_sequence)
    input_ids = torch.tensor([generated], dtype=torch.long, device=device)

    if strategy == 'beam':
        return beam_search(model, tokenizer, input_ids, beam_width, max_length)

    for _ in range(max_length):
        # Get model predictions
        with torch.no_grad():
            outputs = model(input_ids, mask=None)
            next_token_logits = outputs[:, -1, :] / temperature

        # Apply the selected sampling strategy
        if strategy == 'greedy':
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        elif strategy == 'top-k':
            next_token = top_k_sampling(next_token_logits, top_k)
        elif strategy == 'top-p':
            next_token = top_p_sampling(next_token_logits, top_p)
        elif strategy == 'structured':
            next_token = structured_sampling(next_token_logits, input_ids, tokenizer, temperature)
        else:
            raise ValueError(f'Unknown sampling strategy: {strategy}')

        # Append the predicted token to the input
        input_ids = torch.cat([input_ids, next_token], dim=1)

        # Stop if end-of-sequence token is generated
        if next_token.item() == tokenizer.encode_single_token(ENDOFTEXT):
            break

    # Decode the generated tokens
    generated_sequence = tokenizer.decode(input_ids[0].cpu().numpy())
    return generated_sequence


if __name__ == '__main__':
    data_path = 'papers/sampling/data'

    tokenizer = tiktoken.get_encoding('cl100k_base')

    # Define model configuration
    config = ModelConfig(
        vocab_size=tokenizer.max_token_value + 1,
        d_model=512,
        N=6,
        heads=8,
        d_ff=2048,
        max_len=512,
        dropout=0.1,
    )

    # Initialize the model
    model = DecoderOnlyTransformer(config)
    try:
        load_latest_model(model, f'{data_path}/checkpoints')
    except FileNotFoundError:
        pass

    # Define the start sequence
    start_sequence = '## Test different sampling strategies\n'

    # Test different sampling strategies
    strategies = {
        'greedy': {},
        'top-k': {'top_k': 10},
        'top-p': {'top_p': 0.9},
        'beam': {'beam_width': 5},
        'structured': {},
    }

    for strategy, params in strategies.items():
        print(f'\nSampling Strategy: {strategy}')
        generated_text = generate_sequence(
            model=model,
            tokenizer=tokenizer,
            start_sequence=start_sequence,
            strategy=strategy,  # type: ignore
            max_length=50,
            **params,
        )
        print(f'Generated Text: {generated_text}')
