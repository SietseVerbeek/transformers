import math
from typing import Optional, Self

import torch
import torch.nn as nn
from torch import Tensor
from utils.encoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        d_model: int,
        padding_idx: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dropout: float,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_tokens = num_tokens
        self.padding_idx = padding_idx
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(num_tokens, d_model, padding_idx=padding_idx)

        self.positional_encoding = PositionalEncoding(
            d_model=d_model, dropout=dropout, max_len=5000
        )

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.linear = nn.Linear(d_model, num_tokens)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        src_padding_mask: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
    ):
        src = self.positional_encoding(self.embedding(src) * math.sqrt(self.d_model))

        tgt = self.positional_encoding(self.embedding(tgt) * math.sqrt(self.d_model))

        transformer_out = self.transformer(
            src,
            tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )

        return self.linear(transformer_out)


    def to_file(self, filename: str) -> None:
        """
        Saves model parameters to a file, save file can be used to recreate
        the model object without the model state using the from_file classmethod.

        Filename is relative
        """

        params = {
            "num_token": self.num_tokens,
            "d_model": self.d_model,
            "padding_idx": self.padding_idx,
            "num_heads": self.num_heads,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "dropout": self.dropout,
        }

        torch.save(params, filename)

    @classmethod
    def from_file(cls, filename: str) -> Self:
        """
        Loads model parameters from file, use on files created with to_file method.

        Filename is relative
        """


        params = torch.load(filename, weights_only=True)

        num_tokens = params["num_token"]
        d_model = params["d_model"]
        padding_idx = params["padding_idx"]
        num_heads = params["num_heads"]
        num_encoder_layers = params["num_encoder_layers"]
        num_decoder_layers = params["num_decoder_layers"]
        dropout = params["dropout"]

        return cls(
            num_tokens = num_tokens,
            d_model = d_model,
            padding_idx = padding_idx,
            num_heads = num_heads,
            num_encoder_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers,
            dropout = dropout,
        )
