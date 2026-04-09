import torch
import torch.nn as nn
import numpy as np
import math

class SineActivation(nn.Module):
    def __init__(self, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x):
        return torch.sin(self.omega_0 * x)

class DynamicPromptSIREN(nn.Module):
    """上帝视角：频域报警流 (处理 RGB -> Freq)，已注入全局视野"""
    def __init__(self, in_features=None, out_features=None):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        
        # 🌟 核心大改：输入维度变为 in_features * 2，因为我们要把“局部”和“全局”拼在一起看
        self.prompt_generator = nn.Sequential(
            nn.Linear(in_features * 2, in_features // 2),
            nn.GELU(),
            nn.Linear(in_features // 2, in_features),
            nn.Sigmoid()
        )
        
        self.act_fcn = SineActivation(omega_0=30.0)
        hidden_dim = (in_features + out_features) // 2
        
        self.input = nn.Linear(in_features, hidden_dim)
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, out_features)

        with torch.no_grad():
            self.input.weight.uniform_(-1 / in_features, 1 / in_features)
            self.projection.weight.uniform_(-np.sqrt(6 / hidden_dim) / 30.0, np.sqrt(6 / hidden_dim) / 30.0)
            self.output.weight.uniform_(-np.sqrt(6 / hidden_dim) / 30.0, np.sqrt(6 / hidden_dim) / 30.0)

    def forward(self, x):
        x_norm = self.norm(x) # [B, L, C]
        
        # 🌟 全局视野注入 (Global Context Injection)
        # 把整张图的 L 个像素特征取平均，得到一个代表整张图“宏观走势”的特征
        x_global = x_norm.mean(dim=1, keepdim=True) # [B, 1, C]
        # 将局部特征与全局特征拼接，让每个像素在预测时都知道全局长什么样
        x_concat = torch.cat([x_norm, x_global.expand_as(x_norm)], dim=-1) # [B, L, C*2]
        
        prompt_weights = self.prompt_generator(x_concat)
        x_prompted = x_norm + (x_norm * prompt_weights) 
        
        out = self.input(x_prompted)
        out = self.act_fcn(out)
        out = self.projection(out)
        out = self.act_fcn(out)
        out = self.output(out)
        return out

class MultiScale_SpatialDecoder(nn.Module):
    """手术刀视角：负责从频域特征还原回 2D 物理空间 (Freq -> RGB)"""
    def __init__(self, in_features=1152, out_features=448):
        super().__init__()
        self.spatial_restore = nn.Sequential(
            nn.Conv2d(in_features, 512, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(512, out_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_features),
            nn.GELU(),
            nn.Conv2d(out_features, out_features, kernel_size=1)
        )

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L)) 
        x_spatial = x.transpose(1, 2).view(B, C, H, H)
        out_spatial = self.spatial_restore(x_spatial)
        out = out_spatial.flatten(2).transpose(1, 2)
        return out

FeatureProjectionMLP = DynamicPromptSIREN
class FeatureProjectionMLP_big(DynamicPromptSIREN):
    pass