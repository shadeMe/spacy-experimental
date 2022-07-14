from typing import Optional

import torch
from torch.nn import Module
from torch import Tensor

from .attention import MultiHeadAttention


class PointwiseFeedForwardLayer(Module):
    def __init__(self, hidden_dim: int, model_dim: int, *, dropout: float = 0.1):
        super().__init__()

        self.first_linear = torch.nn.Linear(model_dim, hidden_dim)
        self.second_linear = torch.nn.Linear(hidden_dim, model_dim)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Shapes:
            x - (batch, seq_len, model_dim)
        """
        out = self.first_linear(x)
        out = self.activation(out)
        out = self.second_linear(out)
        out = self.dropout(out)
        return out


class EncoderLayer(Module):
    def __init__(
        self,
        model_dim: int,
        ffn_dim: int,
        num_attn_heads: int,
        *,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()

        self.mha = MultiHeadAttention(
            model_dim, n_heads=num_attn_heads, dropout=dropout
        )
        self.attn_layernorm = torch.nn.LayerNorm(model_dim, eps=layer_norm_eps)
        self.attn_dropout = torch.nn.Dropout(p=dropout)

        self.ffn = PointwiseFeedForwardLayer(
            hidden_dim=ffn_dim, model_dim=model_dim, dropout=dropout
        )
        self.ffn_layernorm = torch.nn.LayerNorm(model_dim, eps=layer_norm_eps)
        self.ffn_dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Shapes:
            x - (batch, seq_len, model_dim)
            mask - (batch, seq_len)

        `mask` indicates elements to be masked with values of `1`
        """
        attn_out = self.mha(x, x, x, mask)
        attn_out = self.attn_layernorm(x + attn_out)
        attn_out = self.attn_dropout(attn_out)

        ffn_out = self.ffn(attn_out)
        ffn_out = self.ffn_layernorm(attn_out + ffn_out)
        ffn_out = self.ffn_dropout(ffn_out)

        return ffn_out
