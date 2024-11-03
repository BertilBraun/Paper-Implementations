import torch
from tokenizers import Tokenizer

from src.dataset import BOS_TOKEN, END_OF_SRC_TOKEN, END_OF_TGT_TOKEN
from src.util import create_src_mask, create_tgt_mask
from papers.attention_is_all_you_need.src.transformer import Transformer


def generate_sequence(
    model: Transformer,
    src_sequence: str,
    tokenizer: Tokenizer,
    max_length=50,
) -> str:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    with torch.no_grad():
        src_tokens = tokenizer.encode(BOS_TOKEN + src_sequence + END_OF_SRC_TOKEN).ids
        src_batch = torch.tensor([src_tokens], dtype=torch.long, device=device)
        src_mask = create_src_mask(src_batch, tokenizer, device)

        memory = model.encoder(src_batch, src_mask)
        ys = torch.tensor([[tokenizer.token_to_id(BOS_TOKEN)]], dtype=torch.long, device=device)
        for _ in range(max_length):
            tgt_mask = create_tgt_mask(ys, tokenizer, device)
            out = model.decoder(ys, memory, src_mask, tgt_mask)
            out = model.out(out[:, -1, :])  # Get the last time step
            _, next_word = torch.max(out, dim=1)  # Greedy decoding
            next_word = next_word.item()
            ys = torch.cat([ys, torch.tensor([[next_word]], dtype=torch.long, device=device)], dim=1)
            if next_word == tokenizer.token_to_id(END_OF_TGT_TOKEN):
                break
        generated_tokens = ys[0].cpu().numpy()
        generated_sequence = tokenizer.decode(generated_tokens)
        return generated_sequence


if __name__ == '__main__':
    from papers.attention_is_all_you_need.src.transformer import Transformer
    from src.dataset import load_tokenizer
    from src.util import load_latest_model

    data_dir = 'papers/attention_is_all_you_need/data'

    tokenizer = load_tokenizer(f'{data_dir}/tokenizer.json')

    model = Transformer(vocab_size=tokenizer.get_vocab_size(), d_model=512, N=6, heads=8, d_ff=2048)
    model: Transformer = load_latest_model(model, f'{data_dir}/checkpoints')  # type: ignore

    src_sequence = 'Once upon a time'
    generated_sequence = generate_sequence(model, src_sequence, tokenizer)
    print(generated_sequence)
