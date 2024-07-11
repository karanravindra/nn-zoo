import torch
from ml_zoo.models import GPTConfig, GPT


def test_gpt():
    config = GPTConfig(
        vocab_size=512,
        block_size=32,
        emb_size=32,
        heads=2,
        num_layers=2,
        attn_dropout=0.1,
        ff_mult=4,
        ff_dropout=0.1,
    )
    model = GPT(config)
    assert model

    x = torch.randint(0, 512, (1, 32))
    y = model(x)
    print(y.shape)
    assert y.shape == (1, 32, 512)


if __name__ == "__main__":
    test_gpt
