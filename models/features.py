import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.full_models import FeatureExtractors

dino_backbone_name = 'vit_base_patch8_224.dino'


class MultimodalFeatures(torch.nn.Module):
    def __init__(self, image_size=224):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_size = image_size

        # 保留原始深层特征提取器，主要为了使用 freq_backbone
        self.deep_feature_extractor = FeatureExtractors(
            device=self.device,
            rgb_backbone_name=dino_backbone_name
        ).to(self.device)

        # RGB backbone: ResNet18 + 去掉 maxpool
        self.rgb_backbone = models.resnet18(pretrained=True).to(self.device)
        self.rgb_backbone.maxpool = nn.Identity()

        # 只解冻 layer3，其余冻结
        for name, param in self.rgb_backbone.named_parameters():
            if "layer3" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # 让 BatchNorm 适应当前数据分布
        self.rgb_backbone.train()

    def forward_resnet_multiscale(self, x):
        x = self.rgb_backbone.conv1(x)
        x = self.rgb_backbone.bn1(x)
        x = self.rgb_backbone.relu(x)
        x = self.rgb_backbone.maxpool(x)  # Identity

        # 尺寸：
        # layer1: (B, 64, 112, 112)
        # layer2: (B, 128, 56, 56)
        # layer3: (B, 256, 28, 28)
        layer1 = self.rgb_backbone.layer1(x)
        layer2 = self.rgb_backbone.layer2(layer1)
        layer3 = self.rgb_backbone.layer3(layer2)

        # 56x56 多尺度对齐
        layer1_aligned = F.avg_pool2d(layer1, kernel_size=2, stride=2)  # -> (B, 64, 56, 56)
        layer2_aligned = layer2                                          # -> (B, 128, 56, 56)
        layer3_aligned = F.interpolate(
            layer3, size=(56, 56), mode='bilinear', align_corners=False
        )                                                                # -> (B, 256, 56, 56)

        fused_features = torch.cat(
            [layer1_aligned, layer2_aligned, layer3_aligned], dim=1
        )  # (B, 448, 56, 56)

        return fused_features

    def get_features_maps(self, rgb, freq_img):
        rgb = rgb.to(self.device)
        freq_img = freq_img.to(self.device)

        # RGB 分支允许梯度
        rgb_feature_maps = self.forward_resnet_multiscale(rgb)

        # freq_backbone 冻结，作为稳定频域特征源
        with torch.no_grad():
            freq_feature_maps = self.deep_feature_extractor.freq_backbone(freq_img)
            freq_feature_maps = F.interpolate(
                freq_feature_maps, size=(56, 56), mode='bilinear', align_corners=False
            )

        B = rgb.shape[0]

        # 展平成 patch/token 序列
        rgb_patch = rgb_feature_maps.view(B, 448, -1).transpose(1, 2)      # (B, 3136, 448)
        freq_patch = freq_feature_maps.view(B, 1152, -1).transpose(1, 2)   # (B, 3136, 1152)

        if B == 1:
            return rgb_patch.squeeze(0), freq_patch.squeeze(0)

        return rgb_patch, freq_patch