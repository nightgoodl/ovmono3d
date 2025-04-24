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
        
        # Enhanced depth encoder with more powerful architecture
        self.depth_encoder = nn.Sequential(
            # Initial convolution 
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # First residual block
            self._make_residual_block(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second residual block
            self._make_residual_block(64, 128),
            
            # Final convolution to match channels with image input
            nn.Conv2d(128, 3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            
            # Upsample to match the input size (compensate for the downsampling)
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )
        
        # Create VGGT Aggregator with alternating attention
        # Calculate the maximum dimension and ensure it's divisible by patch_size
        max_size = max(self.img_size)
        img_size_adjusted = ((max_size + self.patch_size - 1) // self.patch_size) * self.patch_size
        
        self.aggregator = Aggregator(
            img_size=img_size_adjusted,  # Use adjusted square size
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            num_register_tokens=self.num_register_tokens,
            aa_order=["frame", "global"],
            aa_block_size=1,
            qk_norm=True,
            rope_freq=100,
            init_values=0.01,
        )
        
        # Output projection layers to map VGGT features to detection features
        # We'll use the same output channels as specified in FPN config
        self.out_channels = cfg.MODEL.FPN.OUT_CHANNELS
        
        # Projection layer to convert concatenated features to the right dimension
        self.projection = nn.Sequential(
            nn.Conv2d(self.embed_dim * 2, self.out_channels, kernel_size=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Define output feature information - use a single feature level
        feature_stride = self.patch_size  # assuming patch_size is the stride of feature map
        self._out_features = ["p4"]
        self._out_feature_channels = {k: self.out_channels for k in self._out_features}
        self._out_feature_strides = {"p4": feature_stride}
        self._size_divisibility = feature_stride
    
    @property
    def size_divisibility(self):
        return self._size_divisibility
    
    def _make_residual_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, images, prompt_depth=None, prompt_nocs=None):
        batch_size = images.shape[0]
        
        # Calculate actual grid size from input dimensions
        h, w = images.shape[-2:]
        
        # Calculate the target size (should match the size used in initialization)
        max_size = max(self.img_size)
        target_size = ((max_size + self.patch_size - 1) // self.patch_size) * self.patch_size
        
        # Calculate padding to make the image square
        h_pad = target_size - h
        w_pad = target_size - w
        
        # Apply padding to make the input square
        if h_pad > 0 or w_pad > 0:
            images = F.pad(images, (0, w_pad, 0, h_pad))
            if prompt_depth is not None:
                prompt_depth = F.pad(prompt_depth, (0, w_pad, 0, h_pad))
        
        # Calculate grid size after padding
        grid_h = target_size // self.patch_size
        grid_w = target_size // self.patch_size
        
        if prompt_depth is not None:
            # Process depth to get 3-channel features with similar structure as image
            depth_features = self.depth_encoder(prompt_depth)
            
            if depth_features.shape[-2:] != images.shape[-2:]:
                depth_features = F.interpolate(
                    depth_features, 
                    size=images.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            # Stack image and depth features for the alternating attention mechanism
            inputs = torch.stack([images, depth_features], dim=1)
        else:
            # If no depth, duplicate image
            inputs = torch.stack([images, images], dim=1)
        
        # Process through VGGT Aggregator
        output_list, patch_start_idx = self.aggregator(inputs)
        
        # Use the last output from the aggregator
        last_output = output_list[-1]  # Shape: [B, S, P, C]
        
        # Only take patch tokens, exclude special tokens
        patch_tokens = last_output[:, 0, patch_start_idx:, :]  # Shape: [B, num_patches, C]
        
        # Reshape to 2D feature map (B, C, H, W)
        feature_map = patch_tokens.reshape(batch_size, grid_h, grid_w, -1).permute(0, 3, 1, 2)
        
        # Pass through projection layer
        p4 = self.projection(feature_map)
        
        # Create the output dictionary
        features = {"p4": p4}
        
        return features

@BACKBONE_REGISTRY.register()
def build_vggt_backbone(cfg, input_shape, priors=None):
    return VGGTBackbone(cfg, input_shape, priors)