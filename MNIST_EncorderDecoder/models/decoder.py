from torch import nn
import torch
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=256, output_dim=28*28):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        out = torch.sigmoid(self.fc2(h))
        return out