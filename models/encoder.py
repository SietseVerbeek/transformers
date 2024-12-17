from typing import Self
import json
import math

import torch
import torch.nn.functional as F
from torch import nn
from utils.encoding import PositionalEncoding


class SequenceEncoder(nn.Transformer):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(
        self,
        num_tokens: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        num_hidden: int,
        dropout=0.5,
    ):
        super().__init__(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=num_hidden,
            num_encoder_layers=num_layers,
            batch_first=True,
        )

        self.num_tokens = num_tokens
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.dropout = dropout

        self.model_type = "Transformer"
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout)

        self.input_emb = nn.Embedding(self.num_tokens, self.d_model)
        self.ninp = self.d_model
        self.decoder = nn.Linear(self.d_model, self.num_tokens)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz, sz)))

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            src_len = src.size(1)
            if self.src_mask is None or self.src_mask.size(0) != src_len:
                mask = self._generate_square_subsequent_mask(src_len).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.input_emb(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.encoder(src, mask=self.src_mask)
        output = self.decoder(output)
        return output
        return F.log_softmax(output, dim=-1)

    def to_file(self, filename: str) -> None:
        params = {
            "num_tokens": self.num_tokens,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "num_hidden": self.num_hidden,
            "dropout": self.dropout,
        }

        with open(filename, "w+") as file:
            json.dump(params, file)

    @classmethod
    def from_file(cls, filename: str) -> Self:
        """
        Loads model parameters from file, use on files created with to_file method.

        Filename is relative
        """


        with open(filename, "r") as file:
            params = json.load(file)

            num_tokens = params["num_tokens"]
            d_model = params["d_model"]
            num_heads = params["num_heads"]
            num_layers = params["num_layers"]
            num_hidden = params["num_hidden"]
            dropout = params["dropout"]

        return cls(
            num_tokens = num_tokens,
            d_model = d_model,
            num_heads = num_heads,
            num_layers = num_layers,
            num_hidden = num_hidden,
            dropout = dropout,
        )

if __name__ == "__main__":
    model = SequenceEncoder(5, 16, 4, 200, 2)
    model.eval()
    out = model(torch.tensor([[0, 1, 0, 1]]))
    out = out.view(-1, 5)

    print(out)
