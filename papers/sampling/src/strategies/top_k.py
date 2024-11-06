import torch


def top_k_sampling(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Top-K sampling: select the next token from the top K most probable tokens.
    """
    top_k = max(top_k, 1)
    values, indices = torch.topk(logits, top_k, dim=-1)
    probabilities = torch.softmax(values, dim=-1)
    next_token = torch.multinomial(probabilities, num_samples=1)
    return indices.gather(-1, next_token)
