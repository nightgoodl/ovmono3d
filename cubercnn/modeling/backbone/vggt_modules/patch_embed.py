import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple, Union

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        bias=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        # Support non-square images
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_size = img_size
        
        # Create projection layer
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Calculate grid size based on actual input dimensions
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        
        # Ensure input dimensions are divisible by patch size
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"Input image dimensions ({H}, {W}) must be divisible by patch size {self.patch_size}"
        
        # (B, C, H, W) -> (B, embed_dim, H//patch_size, W//patch_size)
        x = self.proj(x)
        
        # (B, embed_dim, H', W') -> (B, embed_dim, H'*W') -> (B, H'*W', embed_dim)
        x = x.flatten(2).transpose(1, 2)
        
        return x
