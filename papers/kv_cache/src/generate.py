import tiktoken
from tiktoken_ext.openai_public import ENDOFTEXT
import torch

from papers.kv_cache.src.kv_cache import Transformer, ModelConfig


def generate_sequence(
    model: Transformer,
    src_sequence: str,
    tokenizer: tiktoken.Encoding,
    max_length=50,
) -> str:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    with torch.no_grad():
        ys = torch.tensor([tokenizer.encode(src_sequence)], dtype=torch.long, device=device)

        for i, token in enumerate(ys[0:-1].tolist()):
            _ = model(torch.tensor([[token]], dtype=torch.long, device=device), i, mask=None)

        for i in range(max_length):
            out = model(ys[:, -1:], i + ys.size(1) - 1, mask=None)  # (batch_size, 1, vocab_size)

            next_word = torch.argmax(out, dim=2).item()  # Greedy decoding

            if next_word == tokenizer.encode_single_token(ENDOFTEXT):
                break

            ys = torch.cat([ys, torch.tensor([[next_word]], dtype=torch.long, device=device)], dim=1)

        generated_tokens = ys[0].cpu().numpy()
        generated_sequence = tokenizer.decode(generated_tokens)
        return generated_sequence


if __name__ == '__main__':
    data_dir = 'papers/kv_cache/data'

    tokenizer = tiktoken.get_encoding('cl100k_base')

    config = ModelConfig(
        vocab_size=tokenizer.max_token_value + 1,
        d_model=512,
        N=6,
        heads=8,
        d_ff=2048,
        max_len=512,
        dropout=0.1,
    )

    model = Transformer(config)

    src_sequence = 'Once upon a time'
    generated_sequence = generate_sequence(model, src_sequence, tokenizer)
    print(generated_sequence)
