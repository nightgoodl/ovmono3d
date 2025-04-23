import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY

from .vggt_modules import Aggregator

class DepthAttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.depth_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, depth):
        # Generate attention weights from depth features
        depth_weights = self.depth_attention(depth)
        
        # Apply depth-guided attention
        depth_features = depth * depth_weights
        
        # Concatenate and fuse
        fused = torch.cat([x, depth_features], dim=1)
        return self.fusion_conv(fused)

class VGGTBackbone(Backbone):
    def __init__(self, cfg, input_shape, priors=None):
        super().__init__()
        
        # Get parameters from config
        self.img_size = cfg.MODEL.VGGT.IMG_SIZE
        self.patch_size = cfg.MODEL.VGGT.PATCH_SIZE
        self.embed_dim = cfg.MODEL.VGGT.EMBED_DIM
        self.depth = cfg.MODEL.VGGT.DEPTH
        self.num_heads = cfg.MODEL.VGGT.NUM_HEADS
        self.mlp_ratio = cfg.MODEL.VGGT.MLP_RATIO
        self.num_register_tokens = cfg.MODEL.VGGT.NUM_REGISTER_TOKENS
        
        # Enhanced depth encoder with residual blocks
        self.depth_encoder = nn.Sequential(
            # Initial convolution without stride
            nn.Conv2d(1, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # First residual block
            self._make_residual_block(32),
            
            # Middle convolution without stride
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Second residual block
            self._make_residual_block(64),
            
            # Final convolution to get 3 channels
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # Create VGGT Aggregator
        self.aggregator = Aggregator(
            img_size=self.img_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            num_register_tokens=self.num_register_tokens,
            aa_order=["frame", "global"],
            aa_block_size=1
        )
        
        # Create FPN-like converter
        self.fpn_converter = nn.ModuleDict({
            "p2": nn.Conv2d(self.embed_dim * 2, cfg.MODEL.FPN.OUT_CHANNELS, kernel_size=1),
            "p3": nn.Conv2d(self.embed_dim * 2, cfg.MODEL.FPN.OUT_CHANNELS, kernel_size=1),
            "p4": nn.Conv2d(self.embed_dim * 2, cfg.MODEL.FPN.OUT_CHANNELS, kernel_size=1)
        })
        
        # Define output feature information
        self._out_features = ["p2", "p3", "p4"]
        self._out_feature_channels = {k: cfg.MODEL.FPN.OUT_CHANNELS for k in self._out_features}
        self._out_feature_strides = {"p2": 4, "p3": 8, "p4": 16}
    
    def _make_residual_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, images, prompt_depth=None, prompt_nocs=None):
        batch_size = images.shape[0]
        
        # Calculate actual grid size from input dimensions
        h, w = images.shape[-2:]
        
        # Ensure input dimensions are divisible by patch_size
        h_pad = (self.patch_size - h % self.patch_size) % self.patch_size
        w_pad = (self.patch_size - w % self.patch_size) % self.patch_size
        
        if h_pad > 0 or w_pad > 0:
            images = F.pad(images, (0, w_pad, 0, h_pad))
            if prompt_depth is not None:
                prompt_depth = F.pad(prompt_depth, (0, w_pad, 0, h_pad))
        
        # Calculate grid size after padding
        grid_h = (h + h_pad) // self.patch_size
        grid_w = (w + w_pad) // self.patch_size
        
        if prompt_depth is not None:
            # Process depth to get 3-channel features
            depth_features = self.depth_encoder(prompt_depth)
            # Stack image and depth features
            inputs = torch.stack([images, depth_features], dim=1)
        else:
            # If no depth, duplicate image
            inputs = torch.stack([images, images], dim=1)
        
        # Process through VGGT Aggregator
        output_list, patch_start_idx = self.aggregator(inputs)
        
        # Convert to FPN features
        features = {}
        
        # Use last layer as p4
        p4_features = output_list[-1]
        
        # Calculate feature map shape based on grid size
        num_patches = p4_features.shape[2] - patch_start_idx
        assert num_patches == grid_h * grid_w, f"Expected {grid_h * grid_w} patches but got {num_patches}"
        
        # Only take patch tokens, exclude special tokens
        p4_tokens = p4_features[:, 0, patch_start_idx:, :].reshape(batch_size, grid_h, grid_w, -1).permute(0, 3, 1, 2)
        features["p4"] = self.fpn_converter["p4"](p4_tokens)
        
        # Upsample to get p3 (2x resolution)
        p3_tokens = F.interpolate(p4_tokens, scale_factor=2, mode='bilinear', align_corners=False)
        features["p3"] = self.fpn_converter["p3"](p3_tokens)
        
        # Upsample to get p2 (4x resolution)
        p2_tokens = F.interpolate(p3_tokens, scale_factor=2, mode='bilinear', align_corners=False)
        features["p2"] = self.fpn_converter["p2"](p2_tokens)
        
        return features

@BACKBONE_REGISTRY.register()
def build_vggt_backbone(cfg, input_shape, priors=None):
    """Build VGGT backbone"""
    return VGGTBackbone(cfg, input_shape, priors)