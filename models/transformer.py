from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

from utils.encoding import PositionalEncoding
import math

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
            batch_first=True
        )

        self.linear = nn.Linear(d_model, num_tokens)
        
    def forward(self,
                src: Tensor,
                tgt: Tensor,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                src_padding_mask: Optional[Tensor] = None,
                tgt_padding_mask: Optional[Tensor] = None):

        src = self.positional_encoding(
                    self.embedding(src) * math.sqrt(self.d_model)
                )

        tgt = self.positional_encoding(
                    self.embedding(tgt) * math.sqrt(self.d_model)
                )
        
        transformer_out = self.transformer(src,
                                           tgt,
                                           src_mask=src_mask,
                                           tgt_mask=tgt_mask,
                                           src_key_padding_mask=src_padding_mask,
                                           tgt_key_padding_mask=tgt_padding_mask)

        return self.linear(transformer_out)

class CorrelationTransformer(nn.Module):
    def __init__(self,
        d_model: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dropout: float,
        ):
        super().__init__()
        self.d_model = d_model
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            batch_first=True
        )

        self.linear = nn.Linear(d_model, d_model)

    def forward(self,
                src: Tensor,
                tgt: Tensor,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                src_padding_mask: Optional[Tensor] = None,
                tgt_padding_mask: Optional[Tensor] = None):

        transformer = self.transformer(src,
                                       tgt,
                                       src_mask=src_mask,
                                       tgt_mask=tgt_mask,
                                       src_key_padding_mask=src_padding_mask,
                                       tgt_key_padding_mask=tgt_padding_mask)

        return self.linear(transformer)
