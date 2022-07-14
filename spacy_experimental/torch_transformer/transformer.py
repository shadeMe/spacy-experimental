import math
from typing import Optional

import torch
from torch.nn import Module
from torch import Tensor

from .layers import EncoderLayer
from .embedding import SinusoidalPositionalEmbedding, LearnablePositionalEmbedding


class TransformerEncoder(Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        input_dropout: float,
        other_dropout: float,
        max_len: int,
        vocab_size: int,
        *,
        learnable_pos_embeddings=False,
        layer_norm_eps: float = 1e-5,
        padding_idx: int = 0
    ):
        super().__init__()

        self.input_embeddings = torch.nn.Embedding(
            vocab_size, input_dim, padding_idx=padding_idx
        )
        self.padding_idx = padding_idx
        if learnable_pos_embeddings:
            self.pos_embeddings = LearnablePositionalEmbedding(input_dim, max_len)
        else:
            self.pos_embeddings = SinusoidalPositionalEmbedding(input_dim, max_len)

        self.input_dropout = torch.nn.Dropout(p=input_dropout)
        self.layers = torch.nn.ModuleList(
            [
                EncoderLayer(
                    input_dim,
                    hidden_dim,
                    n_heads,
                    dropout=other_dropout,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(n_layers)
            ]
        )

    def _create_mask(self, x: Tensor) -> Tensor:
        return x.eq(self.padding_idx).int()

    def forward(self, input: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Shapes:
            input - (batch, seq_len)

        `attn_mask` indicates elements to be masked with values of `1`
        """
        if not mask:
            mask = self._create_mask(input)

        emb = self.input_embeddings(input)
        pos = self.pos_embeddings(input)

        x = emb + pos
        out = self.input_dropout(x)

        for layer in self.layers:
            out = layer(out, mask)

        return out
