import torch
import tiktoken
from tiktoken_ext.openai_public import ENDOFTEXT

# TODO Update README with the new structured sampling strategy
# TODO Structured sampling with some sort of nice framework to define the structure


def structured_sampling(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer: tiktoken.Encoding,
    temperature: float = 1.0,
) -> torch.Tensor:
    next_token_logits = filter_logits_structured(logits, input_ids, tokenizer)
    probabilities = torch.softmax(next_token_logits / temperature, dim=-1)
    next_token = torch.multinomial(probabilities, num_samples=1)
    return next_token


def filter_logits_structured(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer: tiktoken.Encoding,
) -> torch.Tensor:
    """
    Filters logits to enforce the structured output format:
    - A title line starting with '## '
    - Followed by 3 to 5 lines starting with '- '
    - Each line limited to at most 50 characters
    """
    # Clone logits to avoid modifying the original tensor
    filtered_logits = logits.clone()
    # Get the generated tokens so far
    generated_ids = input_ids[0].tolist()  # Assuming batch size of 1
    # Decode the generated tokens to text
    generated_text = tokenizer.decode(generated_ids)
    # Split the text into lines
    lines = generated_text.split('\n')
    num_lines = len(lines)
    # Get the last line
    current_line = lines[-1] if lines else ''
    # Get special token IDs
    newline_token_id = tokenizer.encode_single_token('\n')
    eos_token_id = tokenizer.encode_single_token(ENDOFTEXT)
    # Determine the structure state
    if num_lines == 1:
        # Title line
        if len(current_line) < 3:
            # Enforce starting with '## '
            prefix = '## '
            prefix_ids = tokenizer.encode(prefix)
            next_token_id = prefix_ids[len(generated_ids) - len(tokenizer.encode('\n'.join(lines[:-1])))]
            filtered_logits[0, :] = float('-inf')
            filtered_logits[0, next_token_id] = logits[0, next_token_id]
        elif len(current_line) >= 50:
            # Enforce newline
            filtered_logits[0, :] = float('-inf')
            filtered_logits[0, newline_token_id] = logits[0, newline_token_id]
    elif 2 <= num_lines <= 6:
        # List items
        num_list_items = num_lines - 1  # Exclude title line
        if num_list_items == 5 and current_line.endswith('\n'):
            # Enforce EOS after 5 items
            filtered_logits[0, :] = float('-inf')
            filtered_logits[0, eos_token_id] = logits[0, eos_token_id]
        else:
            if len(current_line) == 0:
                # Start new item with '- '
                prefix = '- '
                prefix_ids = tokenizer.encode(prefix)
                next_token_id = prefix_ids[0]
                filtered_logits[0, :] = float('-inf')
                filtered_logits[0, next_token_id] = logits[0, next_token_id]
            elif not current_line.startswith('- '):
                # Enforce '- ' at the start
                prefix = '- '
                prefix_ids = tokenizer.encode(prefix)
                idx_in_line = len(tokenizer.encode(current_line))
                if idx_in_line < len(prefix_ids):
                    next_token_id = prefix_ids[idx_in_line]
                    filtered_logits[0, :] = float('-inf')
                    filtered_logits[0, next_token_id] = logits[0, next_token_id]
            elif len(current_line) >= 50:
                # Enforce newline
                filtered_logits[0, :] = float('-inf')
                filtered_logits[0, newline_token_id] = logits[0, newline_token_id]
    else:
        # Enforce EOS after 5 items
        filtered_logits[0, :] = float('-inf')
        filtered_logits[0, eos_token_id] = logits[0, eos_token_id]
    return filtered_logits
