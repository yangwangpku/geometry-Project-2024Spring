import torch
import torch.nn as nn
import numpy as np

class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dim=3, mapping_size=256, scale=10.0):
        super(FourierFeatureMapping, self).__init__()
        self.B = nn.Parameter(torch.randn(input_dim, mapping_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class SDFMLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, output_dim=1, num_layers=8, use_fourier=False, mapping_size=256, scale=10.0):
        super(SDFMLP, self).__init__()
        
        if use_fourier:
            self.fourier_mapping = FourierFeatureMapping(input_dim, mapping_size, scale)
            input_dim = 2 * mapping_size  # Fourier feature mapping doubles the input dimension
        else:
            self.fourier_mapping = None

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.fourier_mapping is not None:
            x = self.fourier_mapping(x)
        return self.model(x)
