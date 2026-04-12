import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.features import MultimodalFeatures


class CrossModalDBGSeg(nn.Module):
    def __init__(self, image_size=224):
        super().__init__()
        self.feature_extractor = MultimodalFeatures(image_size=image_size)
        self.feature_extractor.deep_feature_extractor.rgb_backbone.requires_grad_(False)
        self.feature_extractor.deep_feature_extractor.freq_backbone.requires_grad_(False)

        self.rgb_projection = nn.Conv2d(448, 256, kernel_size=1)
        self.freq_projection = nn.Conv2d(1152, 256, kernel_size=1)

        self.debackground_gate = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        self.decoder_stage2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.decoder_stage3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.output_head = nn.Conv2d(64, 1, kernel_size=1)

    def _patches_to_map(self, patch_tokens, channels):
        if patch_tokens.dim() == 2:
            patch_tokens = patch_tokens.unsqueeze(0)

        batch_size, seq_len, feature_dim = patch_tokens.shape
        spatial_dim = int(math.sqrt(seq_len))

        if spatial_dim * spatial_dim != seq_len:
            raise ValueError(f"Patch sequence length {seq_len} is not a square number.")
        if feature_dim != channels:
            raise ValueError(f"Expected {channels} channels, got {feature_dim}.")

        return patch_tokens.transpose(1, 2).contiguous().view(batch_size, channels, spatial_dim, spatial_dim)

    def forward(self, rgb, freq_img):
        rgb_patch, freq_patch = self.feature_extractor.get_features_maps(rgb, freq_img)

        rgb_map = self._patches_to_map(rgb_patch, 448)
        freq_map = self._patches_to_map(freq_patch, 1152)

        rgb_proj = self.rgb_projection(rgb_map)
        freq_proj = self.freq_projection(freq_map)

        gate = self.debackground_gate(torch.cat([rgb_proj, freq_proj], dim=1))
        rgb_clean = rgb_proj * gate

        fused = torch.cat([rgb_clean, freq_proj], dim=1)

        x = self.decoder_stage1(fused)
        x = F.interpolate(x, size=(112, 112), mode="bilinear", align_corners=False)
        x = self.decoder_stage2(x)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = self.decoder_stage3(x)

        return self.output_head(x)
