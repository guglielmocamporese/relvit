##################################################
# Imports
##################################################

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


##################################################
# Attention functions
##################################################

def attention_scores(q, k, v, mask=None):
    """
    Implementation of the cross-attention and the self-attention.
    Args:
        q: tensor of shape [B, T_Q, D_K]
        k: tensor of shape [B, T_V, D_K]
        v: tensor of shape [B, T_V, D_V]
    Output:
        out: tensor of shape [B, T_Q, D_v]
    """
    B = q.shape[0]
    scale = math.sqrt(k.shape[2])
    att = torch.bmm(q, k.transpose(1, 2)) / scale # [B, T_Q, T_V]
    if mask is not None:
        mask = mask.unsqueeze(0).repeat(B, 1, 1)
        att = torch.where(mask > 0.0, att, - math.inf * torch.ones_like(att))
    return att


##################################################
# Attention layers
##################################################

class HeadScores(nn.Module):
    """
    Attention is all you need, Vaswani et al, 2017.
    https://arxiv.org/abs/1706.03762
    ::
             Linear proj.     Linear proj.     Linear proj.
               (query: q)       (key: k)        (value: v)
                  ↓                ↓                ↓
                   --------        |        --------
                           ↓       ↓       ↓
                          Attention (q, k, v)
    """
    def __init__(self, h_dim, head_out_dim):
        super().__init__()
        self.q_lin = nn.Linear(h_dim, head_out_dim, bias=False)
        self.k_lin = nn.Linear(h_dim, head_out_dim, bias=False)
        self.v_lin = nn.Linear(h_dim, head_out_dim, bias=False)

    def forward(self, q, k=None, v=None, mask=None):
        if k is None:
            k = q
        if v is None:
            v = k
        q = self.q_lin(q)
        k = self.k_lin(k)
        v = self.v_lin(v)
        x = attention_scores(q, k, v, mask=mask) # [B, NUM_PATCH, NUM_PATCH]
        return x

class MultiHeadAttentionScores(nn.Module):
    """
    Attention is all you need, Vaswani et al, 2017.
    https://arxiv.org/abs/1706.03762
    ::
            [Head_1, Head_2, ..., Head_h]
                           ↓
                       Cat (dim=2)
                           ↓
            Linear (in=h * h_dim, out=h_dim)
    """
    def __init__(self, h_dim, num_heads):
        super().__init__()
        self.h_dim = h_dim
        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            HeadScores(h_dim, h_dim // num_heads) for _ in range(num_heads)
        ])
        #self.linear = nn.Linear((h_dim // num_heads) * num_heads, h_dim)

    def forward(self, q, k=None, v=None, mask=None):
        x = [head(q, k, v, mask=mask) for head in self.heads]
        x = torch.stack(x, 1) # [B, num_heads, NUM_PATCH, NUM_PATCH]
        return x
