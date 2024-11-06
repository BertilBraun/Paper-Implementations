import torch
from torch import nn
import tiktoken
from tiktoken_ext.openai_public import ENDOFTEXT


def beam_search(
    model: nn.Module,
    tokenizer: tiktoken.Encoding,
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
            if seq[-1] == tokenizer.encode_single_token(ENDOFTEXT):
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
        if sequences[0][0][-1] == tokenizer.encode_single_token(ENDOFTEXT):
            break

    # Return the sequence with the highest score
    best_sequence = sequences[0][0]
    generated_sequence = tokenizer.decode(best_sequence)
    return generated_sequence
