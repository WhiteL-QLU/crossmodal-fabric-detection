import torch
import torch.nn as nn
import timm

class DynamicFrequencyMask(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.global_prior = nn.Parameter(torch.zeros(1, in_channels, 224, 224))

    def forward(self, x):
        dynamic_mask = self.spatial_attention(x)
        final_mask = dynamic_mask * torch.sigmoid(self.global_prior)
        return x * final_mask

class FreqEncoder(nn.Module):
    def __init__(self, in_channels=3, embed_dim=1152): 
        super().__init__()
        self.dfm = DynamicFrequencyMask(in_channels=in_channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, embed_dim, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )

    def forward(self, x):
        x_masked = self.dfm(x)
        return self.net(x_masked) 

class FeatureExtractors(torch.nn.Module):
    def __init__(self, device, rgb_backbone_name='vit_base_patch8_224.dino'):
        super().__init__()
        self.device = device
        self.layers_keep = 12

        self.rgb_backbone = timm.create_model(model_name=rgb_backbone_name, pretrained=True)
        self.rgb_backbone.blocks = torch.nn.Sequential(*self.rgb_backbone.blocks[:self.layers_keep])
        self.freq_backbone = FreqEncoder(in_channels=3, embed_dim=1152)

    def forward_rgb_features(self, x):
        x = self.rgb_backbone.patch_embed(x)
        x = self.rgb_backbone._pos_embed(x)
        x = self.rgb_backbone.norm_pre(x)
        x = self.rgb_backbone.blocks(x) 
        x = self.rgb_backbone.norm(x)
        feat = x[:,1:].permute(0, 2, 1).view(x.shape[0], -1, 28, 28) 
        return feat

    def forward(self, rgb, freq_img):
        rgb_features = self.forward_rgb_features(rgb)
        freq_features = self.freq_backbone(freq_img)
        return rgb_features, freq_features