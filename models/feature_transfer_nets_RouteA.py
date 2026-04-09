# feature_transfer_nets_RouteA.py
import torch
import torch.nn as nn

class DynamicPromptMLP(nn.Module):
    """Route A: Dynamic Prompt only, using standard GELU."""
    def __init__(self, in_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        self.act_fcn = act_layer()
        hidden_dim = (in_features + out_features) // 2
        
        self.norm = nn.LayerNorm(in_features)
        
        self.prompt_generator = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.GELU(),
            nn.Linear(in_features // 4, in_features),
            nn.Sigmoid()
        )
        
        self.input = nn.Linear(in_features, hidden_dim)
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, out_features)

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

FeatureProjectionMLP = DynamicPromptMLP
class FeatureProjectionMLP_big(DynamicPromptMLP):
    pass