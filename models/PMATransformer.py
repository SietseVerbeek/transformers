import json
import math
from typing import Self
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.encoding import PositionalEncoding

class PMA(nn.Module):
    def __init__(self, embed_dim, num_heads, num_queries):
        """
        Pooling by Multihead Attention (PMA)
        
        Args:
            embed_dim: Dimension of input embeddings.
            num_heads: Number of attention heads.
            num_queries: Number of learnable query vectors (number of groups to predict).
        """
        super().__init__()
        self.num_queries = num_queries
        self.query = nn.Parameter(torch.randn(num_queries, embed_dim))  # Learnable queries
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (set_size, embed_dim) - Input sequences
        
        Returns:
            pooled_output: Tensor of shape (num_queries, embed_dim)
        """
        batch_size = x.shape[0]
        query = self.query.expand(batch_size, -1, -1)  # Expand queries for batch
        output, attn_weights = self.attention(query, x, x)  # Attention over input variables
        return output, attn_weights  # Pooled output & attention weights

class PMA_GroupingModel(nn.Module):
    def __init__(self,
                 embed_dim=16,
                 enc_layers=1,
                 enc_heads=2,
                 pma_heads=2,
                 dec_heads=2,
                 queries=2,
                 ):
        """
        PMA-based model for grouping binary variables into `num_queries` groups.
        
        Args:
            embed_dim: Embedding size for binary variables.
            num_heads: Number of attention heads.
            num_queries: Number of group labels to predict.
        """
        self.kwargs = locals()
        del self.kwargs["self"]
        del self.kwargs["__class__"]
        super().__init__()

        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(2, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)

        # self attention in sequences
        attention_layer = nn.TransformerEncoderLayer(embed_dim, enc_heads, batch_first=True)
        self.self_attention = nn.TransformerEncoder(attention_layer, enc_layers)

        def mean_closure(dim):
            def mean(x):
                return torch.mean(x, dim=dim)
            return mean

        # pooling over all sequences in a set for every variable
        self.pool = mean_closure(-3)

        # attention between pooled sets and group query vectors
        self.pma = PMA(embed_dim, pma_heads, queries)

        # attention between query vectors and output
        cross_attention_layer = nn.TransformerDecoderLayer(embed_dim, dec_heads, batch_first=True)
        self.cross_attention = nn.TransformerDecoder(cross_attention_layer, 1)

        # output into probabilities for group labels
        self.linear_out = nn.Linear(embed_dim, queries)

    def forward(self, src, tgt, tgt_mask=None):
        """
        Args:
            src: Tensor of shape (batch, set_size, seq_length) - Binary input sequences
            tgt: Tensor of shape (batch, tgt_lenght)
        
        Returns:
            group_scores: Tensor of shape (batch, seq_length, num_queries) - Group scores for each variable
        """

        batch_size = src.shape[0]

        # embed the sequences, (batch, set_size, seq_lenght, embed_dim)
        src = self.positional_encoding(self.embedding(src) * math.sqrt(self.embed_dim))
        # embed the target, (batch, tgt_lenght, embed_dim)
        tgt = self.positional_encoding(self.embedding(tgt) * math.sqrt(self.embed_dim))

        src = src.view(-1, *src.shape[-2:])
        src = self.self_attention(src)
        src = src.view(batch_size, -1, *src.shape[-2:])

        # pool across dim -2, (batch_size, set_size, embed_dim)
        src = self.pool(src)
        
        # attention pooling, (batch_size, num_queries, embed_dim)
        pooled_output, attn_weights = self.pma(src)  # PMA pooling
        
        # (tgt_length, embed_dim)
        cross_output = self.cross_attention(tgt, pooled_output, tgt_mask=tgt_mask)

        # (tgt_length, num_queries)
        output = self.linear_out(cross_output)

        return output

    def to_file(self, filename: str) -> None:

        with open(filename, "w+") as file:
            json.dump(self.kwargs, file)

    @classmethod
    def from_file(cls, filename: str) -> Self:
        with open(filename, "r") as file:
            params = json.load(file)

        return cls(**params)

if __name__ == "__main__":
    batch_size = 10
    set_size = 4
    num_queries = 2
    x = torch.randint(0, 2, (batch_size, set_size, 6))  # Random binary sequences
    tgt = torch.full((batch_size, 2), 0)

    model = PMA_GroupingModel()
    # output = model(x, tgt)

    # print("output: ", output)
    model.to_file('')
