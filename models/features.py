import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.full_models import FeatureExtractors

dino_backbone_name = 'vit_base_patch8_224.dino' # 仅做兼容保留

class MultimodalFeatures(torch.nn.Module):
    def __init__(self, image_size=224):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_size = image_size
        
        # 1. 强行保留原来的特征提取器，仅仅是为了保住里面的 freq_backbone (频域流绝对不能断)
        self.deep_feature_extractor = FeatureExtractors(device=self.device, rgb_backbone_name=dino_backbone_name).to(self.device)
        
        # 2. 🌟 降维打击：植入 ResNet-18 多尺度高清骨干网络
        # 使用 ResNet-18 既能满足 C++ 极速推理，又能提取完美的高清边缘
        self.rgb_backbone = models.resnet18(pretrained=True).to(self.device)
        self.rgb_backbone.eval() # 必须冻结！我们只把它当高清摄像头用
        for param in self.rgb_backbone.parameters():
            param.requires_grad = False

    def forward_resnet_multiscale(self, x):
        # 像剥洋葱一样剥离 ResNet 的每一层，抓取不同分辨率的特征
        x = self.rgb_backbone.conv1(x)
        x = self.rgb_backbone.bn1(x)
        x = self.rgb_backbone.relu(x)
        x = self.rgb_backbone.maxpool(x)
        
        # Layer 1: 高清线稿层 (B, 64, 56, 56) -> 锁定 1 像素物理边缘！
        layer1 = self.rgb_backbone.layer1(x)
        # Layer 2: 纹理结构层 (B, 128, 28, 28) -> 识别正常布料的编织规律
        layer2 = self.rgb_backbone.layer2(layer1)
        # Layer 3: 深层语义层 (B, 256, 14, 14) -> 提供大局观，防止正常区域误报
        layer3 = self.rgb_backbone.layer3(layer2)
        
        # 🌟 多尺度特征融合 (Multi-scale Fusion)：将它们全部强制对齐到 28x28
        layer1_aligned = F.avg_pool2d(layer1, kernel_size=2, stride=2) # 56 -> 28
        layer3_aligned = F.interpolate(layer3, size=(28, 28), mode='bilinear', align_corners=False) # 14 -> 28
        
        # 暴力拼接通道：64 + 128 + 256 = 448 维的多尺度超级特征！
        fused_features = torch.cat([layer1_aligned, layer2, layer3_aligned], dim=1) # (B, 448, 28, 28)
        return fused_features

    def get_features_maps(self, rgb, freq_img):
        rgb = rgb.to(self.device)
        freq_img = freq_img.to(self.device)
        
        with torch.no_grad():
            # 🌟 弃用抽象的 DINO，改用充满物理坐标的 ResNet 多尺度特征
            rgb_feature_maps = self.forward_resnet_multiscale(rgb)
            
        # 频域特征依然走原来的高速公路
        freq_feature_maps = self.deep_feature_extractor.freq_backbone(freq_img)
        
        B = rgb.shape[0]
        # 注意：现在融合后的特征维度是 448 (不再是 DINO 的 768)
        rgb_patch = rgb_feature_maps.view(B, 448, -1).transpose(1, 2)
        freq_patch = freq_feature_maps.view(B, 1152, -1).transpose(1, 2)
        
        if B == 1:
            return rgb_patch.squeeze(0), freq_patch.squeeze(0)
        return rgb_patch, freq_patch