import torch
import datasets
from torch import nn

from src.train import load_latest_model, train_model
from src.dataset import get_tokenizer, prepare_dataloader
from papers.attention_is_all_you_need.src.transformer import Transformer


def map_to_src_tgt(example):
    example['src'] = example['translation']['de']
    example['tgt'] = example['translation']['en']
    return example


def load_dataset(split) -> datasets.Dataset:
    dataset = datasets.load_dataset('wmt14', 'de-en', split=split)
    dataset = dataset.map(map_to_src_tgt, remove_columns=['translation'])
    return dataset  # type: ignore


d_model = 512
d_ff = 2048
N = 6
heads = 8
warmup_steps = 4000
data_path = 'papers/transformer/data'
checkpoint_path = f'{data_path}/checkpoints'


print('Loading datasets...')
train_dataset = load_dataset('train[:1000]')
test_dataset = load_dataset('test[:1000]')

print('Initializing tokenizer...')
tokenizer = get_tokenizer(train_dataset, path=f'{data_path}/tokenizer.json')

train_dataloader = prepare_dataloader(train_dataset, tokenizer, batch_size=64, shuffle=True)
test_dataloader = prepare_dataloader(test_dataset, tokenizer, batch_size=64, shuffle=False)

print('Initializing model...')
model = Transformer(vocab_size=tokenizer.get_vocab_size(), d_model=d_model, N=N, heads=heads, d_ff=d_ff)

try:
    load_latest_model(model, checkpoint_path)
except FileNotFoundError:
    print('No checkpoints found, starting from scratch.')


# def lr_lambda(step):
#     return (d_model**-0.5) * min((step + 1) ** -0.5, (step + 1) * warmup_steps**-1.5)
#
# optimizer = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9)
# lr_scheduler = LambdaLR(optimizer, lr_lambda)

optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('<pad>'))

print('Starting training...')
train_model(
    model,
    train_dataloader,
    test_dataloader,
    loss_fn,
    optimizer,
    tokenizer,
    epochs=5,
    checkpoint_path=checkpoint_path,
)
