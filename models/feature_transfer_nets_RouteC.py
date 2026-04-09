# feature_transfer_nets_RouteC.py
import torch
import torch.nn as nn
import numpy as np

class SineActivation(nn.Module):
    def __init__(self, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x):
        return torch.sin(self.omega_0 * x)

class DynamicPromptSIREN(nn.Module):
    """Route C: Dynamic Prompt + SIREN activation."""
    def __init__(self, in_features=None, out_features=None):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        
        self.prompt_generator = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.GELU(),
            nn.Linear(in_features // 4, in_features),
            nn.Sigmoid()
        )
        
        self.act_fcn = SineActivation(omega_0=30.0)
        hidden_dim = (in_features + out_features) // 2
        
        self.input = nn.Linear(in_features, hidden_dim)
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, out_features)

        # Critical SIREN Initialization
        with torch.no_grad():
            self.input.weight.uniform_(-1 / in_features, 1 / in_features)
            self.projection.weight.uniform_(-np.sqrt(6 / hidden_dim) / 30.0, np.sqrt(6 / hidden_dim) / 30.0)
            self.output.weight.uniform_(-np.sqrt(6 / hidden_dim) / 30.0, np.sqrt(6 / hidden_dim) / 30.0)

    def forward(self, x):
        x_norm = self.norm(x)
        prompt_weights = self.prompt_generator(x_norm)
        x_prompted = x_norm + (x_norm * prompt_weights) 
        
        out = self.input(x_prompted)
        out = self.act_fcn(out)
        out = self.projection(out)
        out = self.act_fcn(out)
        out = self.output(out)
        return out

FeatureProjectionMLP = DynamicPromptSIREN
class FeatureProjectionMLP_big(DynamicPromptSIREN):
    pass