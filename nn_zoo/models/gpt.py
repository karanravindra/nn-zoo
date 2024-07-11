from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from nn_zoo.models.components import SelfAttention

__all__ = ["GPTConfig", "GPT"]


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    emb_size: int
    heads: int
    num_layers: int
    attn_dropout: float
    ff_mult: int
    ff_dropout: float


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.fc1 = nn.Linear(config.emb_size, config.ff_mult * config.emb_size)
        self.fc2 = nn.Linear(config.ff_mult * config.emb_size, config.emb_size)
        self.act = nn.GELU()

        if config.ff_dropout > 0:
            self.dropout = nn.Dropout(config.ff_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        if hasattr(self, "dropout"):
            x = self.dropout(x)

        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.emb_size)
        self.attn = SelfAttention(config.emb_size, config.heads, config.attn_dropout)
        self.ln_2 = nn.LayerNorm(config.emb_size)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x), True)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.emb_size)
        self.pos_emb = nn.Embedding(config.block_size, config.emb_size)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])

        self.ln = nn.LayerNorm(config.emb_size)
        self.head = nn.Linear(config.emb_size, config.vocab_size, bias=False)

        # tie weights
        self.head.weight = self.token_emb.weight

        # initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        B, T = x.size()
        assert (
            not T > self.config.block_size
        ), "Sequence length is longer than block size"

        emb = self.token_emb(x)
        pe = self.pos_emb(torch.arange(T - 1, -1, step=-1))

        x = emb + pe

        for block in self.blocks:
            x = block(x)

        x = self.ln(x)
        return self.head(x)

    def loss(self, y, y_pred):
        # Input is a contiguous tensor
        y = y.flatten()
        y_pred = y_pred.view(-1, y_pred.size(-1))

        return F.cross_entropy(y_pred, y)

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(
        self,
        start_seq: list[int],
        max_len: int = 128,
        temperature: float = 1.0,
        top_k: int | None = None,
    ):
        if top_k is None:
            top_k = self.config.vocab_size
        self.eval()

        generated = start_seq
        primer_t = torch.as_tensor(start_seq).unsqueeze(0)

        for _ in range(max_len):
            if primer_t.size(1) >= self.config.block_size:
                primer_t = primer_t[:, -self.config.block_size :]

            out = self(primer_t)
            out = out[:, -1, :] / temperature
            out = F.softmax(out, dim=-1)
            out = torch.topk(out, top_k, dim=-1)[0]
            out = torch.multinomial(out, num_samples=1)

            gen = out.item()

            generated.append(int(gen))

            primer_t = torch.cat((primer_t, out), dim=1)

        return generated


if __name__ == "__main__":
    config = GPTConfig(
        vocab_size=32,
        block_size=4,
        emb_size=8,
        heads=1,
        num_layers=1,
        attn_dropout=0.1,
        ff_mult=4,
        ff_dropout=0.1,
    )

    model = GPT(config)
    print(model.get_param_count())
    print(model.generate([0, 1, 2]))
