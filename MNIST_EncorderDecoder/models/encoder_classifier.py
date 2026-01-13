from torch import nn
import torch.nn.functional as F

class EncoderClassifier(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim=256, latent_dim=20, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        z = self.fc2(h)
        logits = self.classifier(z)
        return {"logits": logits, "z": z, "h": h}