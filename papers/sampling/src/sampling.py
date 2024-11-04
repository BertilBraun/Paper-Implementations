from typing import Literal
from tokenizers import Tokenizer
import torch
from torch import nn
from src.dataset import END_OF_TGT_TOKEN
from src.util import load_latest_model
from papers.sampling.src.decoder_only_transformer import DecoderOnlyTransformer, ModelConfig


def generate_sequence(
    model: nn.Module,
    tokenizer,
    start_sequence: str,
    max_length: int = 50,
    strategy: Literal['greedy', 'top-k', 'top-p', 'beam'] = 'greedy',
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
        strategy: The sampling strategy ('greedy', 'top-k', 'top-p', 'beam').
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
    generated = tokenizer.encode(start_sequence).ids
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
        else:
            raise ValueError(f'Unknown sampling strategy: {strategy}')

        # Append the predicted token to the input
        input_ids = torch.cat([input_ids, next_token], dim=1)

        # Stop if end-of-sequence token is generated
        if next_token.item() == tokenizer.token_to_id(END_OF_TGT_TOKEN):
            break

    # Decode the generated tokens
    generated_sequence = tokenizer.decode(input_ids[0].cpu().numpy())
    return generated_sequence


def top_k_sampling(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Top-K sampling: select the next token from the top K most probable tokens.
    """
    top_k = max(top_k, 1)
    values, indices = torch.topk(logits, top_k, dim=-1)
    probabilities = torch.softmax(values, dim=-1)
    next_token = torch.multinomial(probabilities, num_samples=1)
    return indices.gather(-1, next_token)


def top_p_sampling(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Nucleus (Top-P) sampling: select the minimum number of most probable tokens
    whose cumulative probability exceeds top_p.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_keep = cumulative_probs <= top_p
    # Ensure at least one token is kept
    sorted_indices_to_keep[:, 0] = 1

    indices_to_keep = sorted_indices[sorted_indices_to_keep]
    logits_filtered = logits.clone()
    logits_filtered[:, ~torch.isin(torch.arange(logits.size(-1)), indices_to_keep)] = float('-inf')

    probabilities = torch.softmax(logits_filtered, dim=-1)
    next_token = torch.multinomial(probabilities, num_samples=1)
    return next_token


def beam_search(
    model: nn.Module,
    tokenizer: Tokenizer,
    input_ids: torch.Tensor,
    beam_width: int,
    max_length: int,
) -> str:
    """
    Beam search decoding.
    """
    device = next(model.parameters()).device
    sequences = [[input_ids[0].tolist(), 0]]  # [sequence, score]

    for _ in range(max_length):
        all_candidates = []
        for seq, score in sequences:
            if seq[-1] == tokenizer.token_to_id(END_OF_TGT_TOKEN):
                # If EOS token is generated, keep the sequence as is
                all_candidates.append([seq, score])
                continue
            with torch.no_grad():
                input_seq = torch.tensor(seq, device=device).unsqueeze(0)
                outputs = model(input_seq, mask=None)

            logits = outputs[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            # Get the top beam_width tokens
            top_log_probs, top_indices = torch.topk(log_probs, beam_width, dim=-1)

            for i in range(beam_width):
                candidate_seq = seq + [top_indices[0, i].item()]
                candidate_score = score - top_log_probs[0, i].item()
                all_candidates.append([candidate_seq, candidate_score])

        # Select the best beam_width sequences
        ordered = sorted(all_candidates, key=lambda x: x[1])
        sequences = ordered[:beam_width]

        # Check if the highest probability sequence ends with EOS token
        if sequences[0][0][-1] == tokenizer.token_to_id(END_OF_TGT_TOKEN):
            break

    # Return the sequence with the highest score
    best_sequence = sequences[0][0]
    generated_sequence = tokenizer.decode(best_sequence)
    return generated_sequence


if __name__ == '__main__':
    from src.dataset import load_tokenizer

    data_path = 'papers/sampling/data'

    # Initialize tokenizer
    tokenizer = load_tokenizer(f'{data_path}/tokenizer.json')

    # Define model configuration
    config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
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
    start_sequence = 'Once upon a time'

    # Test different sampling strategies
    strategies = {
        'greedy': {},
        'top-k': {'top_k': 10},
        'top-p': {'top_p': 0.9},
        'beam': {'beam_width': 5},
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
