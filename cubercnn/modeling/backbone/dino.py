from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import LastLevelMaxPool, FPN
from detectron2.modeling.backbone.vit import SimpleFeaturePyramid
import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
import einops as E
import unittest

class ResidualConvUnit(nn.Module):
    """Residual convolution module similar to PromptDA."""
    
    def __init__(self, features):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        
        self.relu = nn.ReLU(inplace=True)
        self.skip_add = nn.quantized.FloatFunctional()
        
    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        return self.skip_add.add(out, x)

def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module

class DepthFusionBlock(nn.Module):
    """Improved depth fusion block based on PromptDA's FeatureFusionDepthBlock."""
    
    def __init__(self, features):
        super().__init__()
        
        # Residual convolution units for feature processing
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)
        
        # Depth feature extraction with zero module at the end
        self.resConfUnit_depth = nn.Sequential(
            nn.Conv2d(1, features // 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features // 4, features // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            zero_module(
                nn.Conv2d(features // 2, features, kernel_size=3, stride=1, padding=1)
            )
        )
        
        # Output convolution (optional for channel adjustment)
        self.out_conv = nn.Conv2d(
            features, features, kernel_size=1, stride=1, padding=0
        )
        
        self.skip_add = nn.quantized.FloatFunctional()
    
    def forward(self, x, depth, additional_feat=None):
        """
        Forward pass with optional additional feature input.
        
        Args:
            x: Main feature tensor
            depth: Depth tensor
            additional_feat: Optional lower-level feature for multi-scale fusion
        """
        # Store input as residual
        output = x
        
        # Process and fuse additional feature if provided
        if additional_feat is not None:
            res = self.resConfUnit1(additional_feat)
            output = self.skip_add.add(output, res)
        
        # Apply second residual unit
        output = self.resConfUnit2(output)
        
        # Process depth feature
        if depth is not None:
            # Resize depth to match feature map
            if depth.shape[-2:] != output.shape[-2:]:
                depth = F.interpolate(
                    depth, output.shape[2:], mode='bilinear', align_corners=False
                )
            
            # Extract depth features and add to main path
            depth_feat = self.resConfUnit_depth(depth)
            output = self.skip_add.add(output, depth_feat)
        
        # Optional final convolution
        output = self.out_conv(output)
        
        return output

class NOCSFusionBlock(nn.Module):
    """Improved NOCS fusion block based on the depth fusion approach."""
    
    def __init__(self, features):
        super().__init__()
        
        # Residual convolution units for feature processing
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)
        
        # NOCS feature extraction with zero module at the end
        self.resConfUnit_nocs = nn.Sequential(
            nn.Conv2d(3, features // 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features // 4, features // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            zero_module(
                nn.Conv2d(features // 2, features, kernel_size=3, stride=1, padding=1)
            )
        )
        
        # Output convolution
        self.out_conv = nn.Conv2d(
            features, features, kernel_size=1, stride=1, padding=0
        )
        
        self.skip_add = nn.quantized.FloatFunctional()
    
    def forward(self, x, nocs, additional_feat=None):
        """
        Forward pass with optional additional feature input.
        
        Args:
            x: Main feature tensor
            nocs: NOCS tensor
            additional_feat: Optional lower-level feature for multi-scale fusion
        """
        # Store input as residual
        output = x
        
        # Process and fuse additional feature if provided
        if additional_feat is not None:
            res = self.resConfUnit1(additional_feat)
            output = self.skip_add.add(output, res)
        
        # Apply second residual unit
        output = self.resConfUnit2(output)
        
        # Process NOCS feature
        if nocs is not None:
            # Resize NOCS to match feature map
            if nocs.shape[-2:] != output.shape[-2:]:
                nocs = F.interpolate(
                    nocs, output.shape[2:], mode='bilinear', align_corners=False
                )
            
            # Extract NOCS features and add to main path
            nocs_feat = self.resConfUnit_nocs(nocs)
            output = self.skip_add.add(output, nocs_feat)
        
        # Final convolution
        output = self.out_conv(output)
        
        return output

class DINOBackbone(Backbone):
    def __init__(self, cfg, input_shape, dino_name="dino", model_name="vitb16", output="dense", layer=-1, return_multilayer=False, out_feature="last_feat"):
        super().__init__()
        feat_dims = {
            "vitb8": 768,
            "vitb16": 768,
            "vitb14": 768,
            "vitb14_reg": 768,
            "vitl14": 1024,
            "vitg14": 1536,
        }

        # get model
        self.model_name = dino_name
        self.checkpoint_name = f"{dino_name}_{model_name}"
        dino_vit = torch.hub.load(f"facebookresearch/{dino_name}", self.checkpoint_name)
        self.vit = dino_vit
        self.has_registers = "_reg" in model_name

        assert output in ["cls", "gap", "dense", "dense-cls"]
        self.output = output
        self.patch_size = self.vit.patch_embed.proj.kernel_size[0]

        feat_dim = feat_dims[model_name]
        feat_dim = feat_dim * 2 if output == "dense-cls" else feat_dim

        num_layers = len(self.vit.blocks)
        multilayers = [
            num_layers // 4 - 1,
            num_layers // 2 - 1,
            num_layers // 4 * 3 - 1,
            num_layers - 1,
        ]
        
        self.use_depth_fusion = cfg.MODEL.FPN.USE_DEPTH_FUSION
        self.use_nocs_fusion = cfg.MODEL.FPN.USE_NOCS_FUSION
        self.multilevel_fusion = cfg.MODEL.FPN.MULTILEVEL_FUSION
        # Create fusion blocks for each layer if using multilevel fusion
        if return_multilayer:
            self.feat_dim = [feat_dim, feat_dim, feat_dim, feat_dim]
            self.multilayers = multilayers
            
            if self.use_depth_fusion and self.multilevel_fusion:
                self.depth_fusion_blocks = nn.ModuleList([
                    DepthFusionBlock(feat_dim) for _ in range(len(multilayers))
                ])
            elif self.use_depth_fusion:
                self.depth_fusion_block = DepthFusionBlock(feat_dim)
                
            if self.use_nocs_fusion and self.multilevel_fusion:
                self.nocs_fusion_blocks = nn.ModuleList([
                    NOCSFusionBlock(feat_dim) for _ in range(len(multilayers))
                ])
            elif self.use_nocs_fusion:
                self.nocs_fusion_block = NOCSFusionBlock(feat_dim)
        else:
            self.feat_dim = feat_dim
            layer = multilayers[-1] if layer == -1 else layer
            self.multilayers = [layer]
            
            if self.use_depth_fusion:
                self.depth_fusion_block = DepthFusionBlock(feat_dim)
                
            if self.use_nocs_fusion:
                self.nocs_fusion_block = NOCSFusionBlock(feat_dim)

        # define layer name (for logging)
        self.layer = "-".join(str(_x) for _x in self.multilayers)

        # Set up output features
        if return_multilayer:
            self._out_features = [f"{out_feature}_{i}" for i in range(len(multilayers))]
            self._out_feature_channels = {f"{out_feature}_{i}": feat_dim for i in range(len(multilayers))}
            self._out_feature_strides = {f"{out_feature}_{i}": self.patch_size * (2 ** i) for i in range(len(multilayers))}
        else:
            self._out_feature_channels = {out_feature: feat_dim}
            self._out_feature_strides = {out_feature: self.patch_size}
            self._out_features = [out_feature]

    def forward(self, images, prompt_depth=None, prompt_nocs=None):
        h, w = images.shape[-2:]
        h, w = h // self.patch_size, w // self.patch_size

        if self.model_name == "dinov2":
            x = self.vit.prepare_tokens_with_masks(images, None)
        else:
            x = self.vit.prepare_tokens(images)

        embeds = []
        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if i in self.multilayers:
                embeds.append(x)
                if len(embeds) == len(self.multilayers):
                    break

        num_spatial = h * w
        outputs = {}
        
        for idx, x_i in enumerate(embeds):
            cls_tok = x_i[:, 0]
            spatial = x_i[:, -1 * num_spatial:]
            x_i = tokens_to_output(self.output, spatial, cls_tok, (h, w))
            
            # Apply multi-level fusion if enabled
            if self.multilevel_fusion:
                # Depth fusion at multiple levels
                if self.use_depth_fusion and prompt_depth is not None:
                    x_i = self.depth_fusion_blocks[idx](x_i, prompt_depth)
                
                # NOCS fusion at multiple levels
                if self.use_nocs_fusion and prompt_nocs is not None:
                    x_i = self.nocs_fusion_blocks[idx](x_i, prompt_nocs)
            # Apply fusion at the last layer only if not using multilevel fusion
            elif idx == len(embeds) - 1:
                # Depth fusion
                if self.use_depth_fusion and prompt_depth is not None:
                    x_i = self.depth_fusion_block(x_i, prompt_depth)
                
                # NOCS fusion
                if self.use_nocs_fusion and prompt_nocs is not None:
                    x_i = self.nocs_fusion_block(x_i, prompt_nocs)
            
            outputs[self._out_features[idx]] = x_i

        return outputs
    
    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


@BACKBONE_REGISTRY.register()
def build_dino_backbone(cfg, input_shape: ShapeSpec, priors=None):
    dino_name = cfg.MODEL.DINO.NAME
    model_name = cfg.MODEL.DINO.MODEL_NAME
    output = cfg.MODEL.DINO.OUTPUT
    layer = cfg.MODEL.DINO.LAYER
    return_multilayer = cfg.MODEL.DINO.RETURN_MULTILAYER

    bottom_up = DINOBackbone(
        cfg,
        input_shape,
        dino_name=dino_name,
        model_name=model_name,
        output=output,
        layer=layer,
        return_multilayer=return_multilayer,
    )

    in_feature = cfg.MODEL.FPN.IN_FEATURE
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    scale_factors = (2.0, 1.0, 0.5)
    backbone = SimpleFeaturePyramid(
        net=bottom_up,
        in_feature=in_feature,
        out_channels=out_channels,
        scale_factors=scale_factors,
        norm=cfg.MODEL.FPN.NORM,
        top_block=None,
        square_pad=cfg.MODEL.FPN.SQUARE_PAD
    )
    return backbone

def tokens_to_output(output_type, dense_tokens, cls_token, feat_hw):
    if output_type == "cls":
        assert cls_token is not None
        output = cls_token
    elif output_type == "gap":
        output = dense_tokens.mean(dim=1)
    elif output_type == "dense":
        h, w = feat_hw
        dense_tokens = E.rearrange(dense_tokens, "b (h w) c -> b c h w", h=h, w=w)
        output = dense_tokens.contiguous()
    elif output_type == "dense-cls":
        assert cls_token is not None
        h, w = feat_hw
        dense_tokens = E.rearrange(dense_tokens, "b (h w) c -> b c h w", h=h, w=w)
        cls_token = cls_token[:, :, None, None].repeat(1, 1, h, w)
        output = torch.cat((dense_tokens, cls_token), dim=1).contiguous()
    else:
        raise ValueError()

    return output

class TestDINOBackbone(unittest.TestCase):
    def setUp(self):
        # Mock configuration
        self.cfg = type('', (), {})()
        self.cfg.MODEL = type('', (), {})()
        self.cfg.MODEL.DINO = type('', (), {})()
        self.cfg.MODEL.DINO.NAME = "dino"
        self.cfg.MODEL.DINO.MODEL_NAME = "vitb16"
        self.cfg.MODEL.DINO.OUTPUT = "dense"
        self.cfg.MODEL.DINO.LAYER = -1
        self.cfg.MODEL.DINO.RETURN_MULTILAYER = True
        self.cfg.MODEL.FPN = type('', (), {})()
        self.cfg.MODEL.FPN.IN_FEATURE = 'last_feat'
        self.cfg.MODEL.FPN.OUT_CHANNELS = 256
        self.cfg.MODEL.FPN.NORM = "LN"
        self.cfg.MODEL.FPN.FUSE_TYPE = "sum"
        self.cfg.MODEL.FPN.SQUARE_PAD = 0
        self.cfg.MODEL.FPN.USE_DEPTH_FUSION = True
        self.cfg.MODEL.FPN.USE_NOCS_FUSION = True
        self.cfg.MODEL.FPN.MULTILEVEL_FUSION = True
        self.input_shape = ShapeSpec(channels=3, height=512, width=512)

    def test_dino_backbone_forward(self):
        # Create the backbone
        backbone = build_dino_backbone(self.cfg, self.input_shape)
        # Generate random input tensors
        x = torch.randn(1, 3, 512, 512)
        depth = torch.randn(1, 1, 512, 512)
        nocs = torch.randn(1, 3, 512, 512)
        
        # Run forward pass
        outputs = backbone(x, prompt_depth=depth, prompt_nocs=nocs)
        
        print("Backbone output shape:")
        print(backbone.net.output_shape())
        
        print("\nFeature outputs:")
        for key, output in outputs.items():
            print(f"{key}: {output.shape}")

if __name__ == "__main__":
    unittest.main()