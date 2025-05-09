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

class DepthFusionBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        # depth feature extraction with normalization
        self.depth_conv = nn.Sequential(
            nn.Conv2d(1, features // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features // 4),
            nn.ReLU(True),
            nn.Conv2d(features // 4, features // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features // 2),
            nn.ReLU(True),
            nn.Conv2d(features // 2, features, kernel_size=3, stride=1, padding=1)
        )
        # zero init last conv to start with identity behavior (depth zero)
        nn.init.zeros_(self.depth_conv[-1].weight)
        nn.init.zeros_(self.depth_conv[-1].bias)

        # fusion conv initialized to identity
        self.fusion_conv = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        # identity init: delta kernel
        with torch.no_grad():
            k = self.fusion_conv.kernel_size[0]
            self.fusion_conv.weight.zero_()
            for c in range(features):
                self.fusion_conv.weight[c, c, k//2, k//2] = 1.0
            self.fusion_conv.bias.zero_()

    def forward(self, x, depth):
        # assume depth already resized outside
        depth_feat = self.depth_conv(depth)
        fused = x + depth_feat
        out = self.fusion_conv(fused)
        return out

class NOCSFusionBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.nocs_conv = nn.Sequential(
            nn.Conv2d(3, features // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features // 4),
            nn.ReLU(True),
            nn.Conv2d(features // 4, features // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features // 2),
            nn.ReLU(True),
            nn.Conv2d(features // 2, features, kernel_size=3, stride=1, padding=1)
        )
        nn.init.zeros_(self.nocs_conv[-1].weight)
        nn.init.zeros_(self.nocs_conv[-1].bias)

        self.fusion_conv = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        with torch.no_grad():
            k = self.fusion_conv.kernel_size[0]
            self.fusion_conv.weight.zero_()
            for c in range(features):
                self.fusion_conv.weight[c, c, k//2, k//2] = 1.0
            self.fusion_conv.bias.zero_()

    def forward(self, x, nocs):
        nocs_feat = self.nocs_conv(nocs)
        fused = x + nocs_feat
        out = self.fusion_conv(fused)
        return out

class DINOBackbone(Backbone):
    def __init__(self, cfg, input_shape, dino_name="dino", model_name="vitb16", output="dense", layer=-1, return_multilayer=False, out_feature="last_feat",):
        super().__init__()
        feat_dims = {"vitb8": 768, "vitb16": 768, "vitb14": 768, "vitb14_reg": 768, "vitl14": 1024, "vitg14": 1536}

        # load model
        self.model_name = dino_name
        self.checkpoint_name = f"{dino_name}_{model_name}"
        dino_vit = torch.hub.load(f"facebookresearch/{dino_name}", self.checkpoint_name)
        self.vit = dino_vit
        self.patch_size = self.vit.patch_embed.proj.kernel_size[0]

        feat_dim = feat_dims[model_name]
        if output == "dense-cls":
            feat_dim *= 2

        num_layers = len(self.vit.blocks)
        multilayers = [num_layers//4-1, num_layers//2-1, num_layers//4*3-1, num_layers-1]

        self.use_depth_fusion = cfg.MODEL.FPN.USE_DEPTH_FUSION
        self.use_nocs_fusion = cfg.MODEL.FPN.USE_NOCS_FUSION
        if self.use_depth_fusion:
            self.depth_fusion_block = DepthFusionBlock(feat_dim)
        if self.use_nocs_fusion:
            self.nocs_fusion_block = NOCSFusionBlock(feat_dim)

        if return_multilayer:
            self.feat_dim = [feat_dim]*4
            self.multilayers = multilayers
        else:
            self.feat_dim = feat_dim
            layer = multilayers[-1] if layer == -1 else layer
            self.multilayers = [layer]

        self._out_feature_channels = {out_feature: feat_dim}
        self._out_feature_strides = {out_feature: self.patch_size}
        self._out_features = [out_feature]

    def forward(self, images, prompt_depth=None, prompt_nocs=None):
        B, _, H, W = images.shape
        h, w = H//self.patch_size, W//self.patch_size

        x = self.vit.prepare_tokens(images)
        embeds = []
        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if i in self.multilayers:
                embeds.append(x)
                if len(embeds) == len(self.multilayers): break

        outputs = {}
        num_spatial = h * w
        for idx, x_i in enumerate(embeds):
            cls_tok = x_i[:, 0]
            spatial = x_i[:, -num_spatial:]
            x_i = tokens_to_output(self.output, spatial, cls_tok, (h, w))

            if self.use_depth_fusion and prompt_depth is not None:
                depth_scaled = F.interpolate(prompt_depth, size=x_i.shape[-2:], mode='bilinear', align_corners=False)
                x_i = self.depth_fusion_block(x_i, depth_scaled)
            if self.use_nocs_fusion and prompt_nocs is not None:
                nocs_scaled = F.interpolate(prompt_nocs, size=x_i.shape[-2:], mode='bilinear', align_corners=False)
                x_i = self.nocs_fusion_block(x_i, nocs_scaled)

            outputs[self._out_features[idx]] = x_i
        return outputs

@BACKBONE_REGISTRY.register()
def build_dino_backbone(cfg, input_shape: ShapeSpec, priors=None):
    bottom_up = DINOBackbone(cfg, input_shape,
                             dino_name=cfg.MODEL.DINO.NAME,
                             model_name=cfg.MODEL.DINO.MODEL_NAME,
                             output=cfg.MODEL.DINO.OUTPUT,
                             layer=cfg.MODEL.DINO.LAYER,
                             return_multilayer=cfg.MODEL.DINO.RETURN_MULTILAYER)
    return SimpleFeaturePyramid(
        net=bottom_up,
        in_feature=cfg.MODEL.FPN.IN_FEATURE,
        out_channels=cfg.MODEL.FPN.OUT_CHANNELS,
        scale_factors=(2.0,1.0,0.5),
        norm=cfg.MODEL.FPN.NORM,
        top_block=None,
        square_pad=cfg.MODEL.FPN.SQUARE_PAD
    )

def tokens_to_output(output_type, dense_tokens, cls_token, feat_hw):
    # unchanged
    if output_type == "cls":
        return cls_token
    elif output_type == "gap":
        return dense_tokens.mean(dim=1)
    elif output_type == "dense":
        h,w = feat_hw
        return E.rearrange(dense_tokens, "b (h w) c -> b c h w", h=h, w=w).contiguous()
    elif output_type == "dense-cls":
        h,w = feat_hw
        dense = E.rearrange(dense_tokens, "b (h w) c -> b c h w", h=h, w=w)
        cls = cls_token[:,:,None,None].repeat(1,1,h,w)
        return torch.cat((dense, cls), dim=1).contiguous()
    else:
        raise ValueError()


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
        self.cfg.MODEL.DINO.RETURN_MULTILAYER = False
        self.cfg.MODEL.FPN = type('', (), {})()
        self.cfg.MODEL.FPN.IN_FEATURE = 'last_feat'
        self.cfg.MODEL.FPN.OUT_CHANNELS = 256
        self.cfg.MODEL.FPN.NORM = "LN"
        self.cfg.MODEL.FPN.FUSE_TYPE = "sum"
        self.input_shape = ShapeSpec(channels=3, height=512, width=512)

    def test_dino_backbone_forward(self):
        # Create the backbone
        backbone = build_dino_backbone(self.cfg, self.input_shape)
        # Generate a random input tensor
        x = torch.randn(1, 3, 512, 512)
        # Run forward pass
        outputs = backbone(x)
        print(backbone.net.output_shape())
        for key, output in outputs.items():
            print(key, output.shape)

        # print(backbone.net.vit)


if __name__ == "__main__":
    unittest.main()