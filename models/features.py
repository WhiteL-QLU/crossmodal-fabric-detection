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
        
        # 1. 强行保留原来的特征提取器，保住 freq_backbone
        self.deep_feature_extractor = FeatureExtractors(device=self.device, rgb_backbone_name=dino_backbone_name).to(self.device)
        
        # 2. 🌟 V2.2 基因解冻架构
        self.rgb_backbone = models.resnet18(pretrained=True).to(self.device)
        
        # 核心手术：解冻 Layer 3，让网络开始学习织物的高频特征
        for name, param in self.rgb_backbone.named_parameters():
            if "layer3" in name:
                param.requires_grad = True 
            else:
                param.requires_grad = False 
        
        # 必须开启 train()，让 BatchNorm 适应布料数据的统计分布
        self.rgb_backbone.train() 

    def forward_resnet_multiscale(self, x):
        # 像剥洋葱一样剥离 ResNet
        x = self.rgb_backbone.conv1(x)
        x = self.rgb_backbone.bn1(x)
        x = self.rgb_backbone.relu(x)
        x = self.rgb_backbone.maxpool(x)
        
        layer1 = self.rgb_backbone.layer1(x)   # (B, 64, 56, 56)
        layer2 = self.rgb_backbone.layer2(layer1) # (B, 128, 28, 28)
        layer3 = self.rgb_backbone.layer3(layer2) # (B, 256, 14, 14)
        
        # 🌟 多尺度对齐与融合
        layer1_aligned = F.avg_pool2d(layer1, kernel_size=2, stride=2) 
        layer3_aligned = F.interpolate(layer3, size=(28, 28), mode='bilinear', align_corners=False) 
        
        fused_features = torch.cat([layer1_aligned, layer2, layer3_aligned], dim=1) # (B, 448, 28, 28)
        return fused_features

    def get_features_maps(self, rgb, freq_img):
        rgb = rgb.to(self.device)
        freq_img = freq_img.to(self.device)
        
        # 🌟 V2.2 移除 no_grad 限制，允许 Backbone 计算梯度进行微调
        rgb_feature_maps = self.forward_resnet_multiscale(rgb)
            
        with torch.no_grad():
            freq_feature_maps = self.deep_feature_extractor.freq_backbone(freq_img)
        
        B = rgb.shape[0]
        rgb_patch = rgb_feature_maps.view(B, 448, -1).transpose(1, 2)
        freq_patch = freq_feature_maps.view(B, 1152, -1).transpose(1, 2)
        
        if B == 1:
            return rgb_patch.squeeze(0), freq_patch.squeeze(0)
        return rgb_patch, freq_patch