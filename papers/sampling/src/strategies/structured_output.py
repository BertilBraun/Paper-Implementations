import torch
import string
import tiktoken
from tiktoken_ext.openai_public import ENDOFTEXT


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


# Caching of tokens for each tokenizer to avoid repeated decoding - which is extremely slow
TOKENIZER_TOKENS: dict[tiktoken.Encoding, list[tuple[int, str]]] = {}


def get_all_tokens(tokenizer: tiktoken.Encoding) -> list[tuple[int, str]]:
    if tokenizer not in TOKENIZER_TOKENS:
        # For some reason, 20 tokens less than the max_token_value are defined in the tokenizer
        TOKENIZER_TOKENS[tokenizer] = [
            (i, tokenizer.decode_single_token_bytes(i).decode('utf-8', errors='replace'))
            for i in range(tokenizer.max_token_value - 20)
        ] + [
            (tokenizer.encode_single_token(ENDOFTEXT), ENDOFTEXT)  # Add the ENDOFTEXT token separately
        ]

    return TOKENIZER_TOKENS[tokenizer]


def _enforce(
    generated_text: str,
    text: str,
    tokenizer: tiktoken.Encoding,
    logits: torch.Tensor,
) -> torch.Tensor:
    """Filter logits to enforce that the generated text will end with the given text.
    For example, if the generated text is 'Hello, world!', and the text is '!\n', then
    the logits for the next token will be filtered to only allow '\n' as the next token.
    """

    # Find how much of 'text' has already been generated
    remaining_text = text
    while generated_text.endswith(remaining_text) and remaining_text:
        remaining_text = remaining_text[:-1]

    if not remaining_text:
        return logits

    # Clone logits to avoid modifying the original tensor
    filtered_logits = logits.clone()
    filtered_logits[0, :] = -1e9

    entire_text = generated_text + remaining_text

    # For each token in the tokenizer's vocabulary
    for token_id, token_text in get_all_tokens(tokenizer):
        concatenated_text = generated_text + token_text
        if len(concatenated_text) <= len(entire_text) and entire_text.startswith(concatenated_text):
            filtered_logits[0, token_id] = logits[0, token_id]

    return filtered_logits


def _enforce_allowed_characters(
    logits: torch.Tensor,
    tokenizer: tiktoken.Encoding,
) -> torch.Tensor:
    """Filter logits to allow only alphanumeric characters, punctuation, and spaces."""
    allowed_chars = set(string.ascii_letters + string.digits + '.!?,: ')
    filtered_logits = logits.clone()
    filtered_logits[0, :] = -1e9

    # For each token in the tokenizer's vocabulary
    for token_id, token_text in get_all_tokens(tokenizer):
        if all(c in allowed_chars for c in token_text):
            filtered_logits[0, token_id] = logits[0, token_id]

    return filtered_logits


def filter_logits_structured(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer: tiktoken.Encoding,
) -> torch.Tensor:
    """
    Filters logits to enforce the structured output format:
    - A title line starting with '## '
    - Followed by 3 to 5 lines starting with '- '
    - Each line limited to at most 30 characters but at least 15 characters, ending with '\n'
    - Each line may only contain alphanumeric characters, punctuation, and spaces
    """
    # Get the generated tokens so far
    generated_ids = input_ids[0].tolist()  # Assuming batch size of 1
    # Decode the generated tokens to text
    generated_text = tokenizer.decode(generated_ids)
    # Split the text into lines
    lines = generated_text.split('\n')
    num_lines = len(lines)
    # Get the last line
    current_line = lines[-1] if lines else ''

    # Determine the structure state
    if num_lines == 1:
        # Title line
        if len(current_line) < 3:
            # Enforce starting with '## '
            logits = _enforce(generated_text, '## ', tokenizer, logits)
        elif len(current_line) >= 30:
            # Enforce newline after max length
            logits = _enforce(generated_text, '\n', tokenizer, logits)
        elif len(current_line) < 15:
            # Allow alphanumeric, punctuation, and spaces
            logits = _enforce_allowed_characters(logits, tokenizer)
        else:
            # We are between 15 and 30 characters -> allow alphanumeric, punctuation, and spaces AND '\n'
            logits = _enforce_allowed_characters(logits, tokenizer) + _enforce(generated_text, '\n', tokenizer, logits)
    elif 2 <= num_lines <= 6:
        # List items
        if len(current_line) < 2:
            # Start new item with '- '
            logits = _enforce(generated_text, '- ', tokenizer, logits)
            if num_lines > 4:
                # Allow EOS after 3 List items
                logits += _enforce(generated_text, ENDOFTEXT, tokenizer, logits)
        elif len(current_line) >= 30:
            # Enforce newline after max length
            logits = _enforce(generated_text, '\n', tokenizer, logits)
        elif len(current_line) < 15:
            # Allow alphanumeric, punctuation, and spaces
            logits = _enforce_allowed_characters(logits, tokenizer)
        else:
            # We are between 15 and 30 characters -> allow alphanumeric, punctuation, and spaces AND '\n'
            logits = _enforce_allowed_characters(logits, tokenizer) + _enforce(generated_text, '\n', tokenizer, logits)
    else:
        # Enforce EOS after 5 items
        logits = _enforce(generated_text, ENDOFTEXT, tokenizer, logits)

    return logits
