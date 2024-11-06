import torch


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
    logits_filtered[:, ~torch.isin(torch.arange(logits.size(-1)), indices_to_keep)] = -1e9

    probabilities = torch.softmax(logits_filtered, dim=-1)
    next_token = torch.multinomial(probabilities, num_samples=1)
    return next_token
