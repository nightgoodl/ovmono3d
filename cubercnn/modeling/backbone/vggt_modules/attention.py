import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        proj_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        qk_norm=False,
        rope=None,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.qk_norm = qk_norm
        if self.qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
        
        self.rope = rope

    def chunk_attention(self, q, k, v, chunk_size=1024):
        B, H, N, D = q.shape
        
        out = torch.zeros_like(q)
        normalizer = torch.zeros((B, H, N, 1), device=q.device)
        
        # Process query in chunks
        for i in range(0, N, chunk_size):
            q_chunk = q[:, :, i:min(i + chunk_size, N)]
            
            # Process key/value in chunks
            for j in range(0, N, chunk_size):
                k_chunk = k[:, :, j:min(j + chunk_size, N)]
                v_chunk = v[:, :, j:min(j + chunk_size, N)]
                
                # Compute attention scores for this chunk
                scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * self.scale
                
                # Apply softmax normalization
                attn_weights = scores.softmax(dim=-1)
                attn_weights = self.attn_drop(attn_weights)
                
                # Update output and normalizer
                out[:, :, i:i + chunk_size] += torch.matmul(attn_weights, v_chunk)
                normalizer[:, :, i:i + chunk_size] += attn_weights.sum(dim=-1, keepdim=True)
        
        # Normalize the output
        out = out / (normalizer + 1e-6)
        return out

    def forward(self, x, pos: Optional[torch.Tensor] = None):
        B, N, C = x.shape
        
        # Compute QKV matrices
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each [B, H, N, D]

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        # Use chunked attention
        x = self.chunk_attention(q, k, v)
        
        # Project output
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
