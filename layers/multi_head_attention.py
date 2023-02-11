import torch
import torch.nn as nn
import math
from utils import clones

def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention

    Args:
        query, key, value: (B, T, C)
    """

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / \
        math.sqrt(d_k)  # (B,T,C) @ (B,C,T) = (B, T, T)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn_output_weights = scores.softmax(dim=-1)
    if dropout is not None:
        attn_output_weights = dropout(attn_output_weights)
    # (B,T,T) @ (B,T,C) = (B, T, C)
    attn_output = torch.matmul(attn_output_weights, value)
    return attn_output, attn_output_weights

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Implements Figure 2
        Agrs:
            query, key, value: (B, T, d_model)
            mask: (1, T, T)
        """

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # (1, T, T) ->  (1, 1, T, T)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h,
                        self.d_k).transpose(1, 2)  # (B, h, T, d_k)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        ) # (B, h, T, d_k)

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )  # (B, T, d_model)

        del query
        del key
        del value
        return self.linears[-1](x) # (B, T, d_model)