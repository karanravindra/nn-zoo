import torch
from torch import nn
import torch.nn.functional as F

__all__ = ["SelfAttention"]


class SelfAttention(nn.Module):
    def __init__(
        self, n_embd: int, n_head: int, attn_dropout: float = 0.0, is_causal=True
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.n_head = n_head
        self.n_embd = n_embd
        self.attn_dropout = attn_dropout
        self.is_causal = is_causal

    def forward(self, x) -> torch.Tensor:
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = (
            F.scaled_dot_product_attention(
                q, k, v, is_causal=self.is_causal, dropout_p=self.attn_dropout
            )
            .transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        # output projection
        y = self.c_proj(y)
        return y
