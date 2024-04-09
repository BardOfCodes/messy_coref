import copy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import feature_alpha_dropout
import torch.nn as nn
import torch
import torch.nn.functional as F
import einops

class DMLP(nn.Module):
    def __init__(self, ind, hdim1, hdim2, odim, DP):
        super(DMLP, self).__init__()
        
        self.l1 = nn.Linear(ind, hdim1)
        self.l2 = nn.Linear(hdim1, hdim2)
        self.l3 = nn.Linear(hdim2, odim)
        self.d1 = nn.Dropout(p=DP)
        self.d2 = nn.Dropout(p=DP)
                
    def forward(self, x):
        x = self.d1(F.relu(self.l1(x)))
        x = self.d2(F.relu(self.l2(x)))
        return self.l3(x)


class SDMLP(nn.Module):
    def __init__(self, ind, mid_dim, odim, DP):
        super(SDMLP, self).__init__()
        
        self.l1 = nn.Linear(ind, mid_dim)
        self.l2 = nn.Linear(mid_dim, odim)
        self.d1 = nn.Dropout(p=DP)
                
    def forward(self, x):
        x = self.d1(F.leaky_relu(self.l1(x), 0.2))
        return self.l2(x)

class AttnLayer(nn.Module):
    def __init__(self, nh, hd, dropout):
        super(AttnLayer, self).__init__()
        self.num_heads = nh
        self.hidden_dim = hd

        self.self_attn = nn.MultiheadAttention(self.hidden_dim, self.num_heads)

        self.l1 = nn.Linear(hd, hd)
        self.l2 = nn.Linear(hd, hd)

        self.d1 = nn.Dropout(dropout)
        self.d2 = nn.Dropout(dropout)
        self.d3 = nn.Dropout(dropout)        

        self.n1 = nn.LayerNorm(hd)
        self.n2 = nn.LayerNorm(hd)
                
    def forward(self, src, attn_mask, key_padding_mask):
        
        src = src.transpose(0, 1)
            
        src2 = self.self_attn(
            src,
            src,
            src,
            attn_mask=attn_mask,
            key_padding_mask = key_padding_mask,
            need_weights=False
        )[0]

        src = src + self.d1(src2)
        src = self.n1(src)
        src2 = self.l2(self.d2(F.leaky_relu(self.l1(self.n2(src)), .2)))
        src = src + self.d2(src2)
        src = self.n2(src)
        return src.transpose(0, 1)
        
