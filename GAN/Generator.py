import torch
from torch import nn

# Latent data is the noise. We want to generate real data (1, 28, 28) and values -1, 1
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 7*7*64) # (n, 256, 7, 7)
        self.conv1 = nn.ConvTranspose2d(64, 32, 4, stride=2) # (n, 64, 16, 16)
        self.conv2 = nn.ConvTranspose2d(32, 16, 4, stride=2) # (n, 16, 34, 34)
        self.conv3 = nn.Conv2d(16, 1, kernel_size=7) #(n, 1, 28, 28)
        self.act = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # Pass latent space input into linear layer and reshape
        x = self.fc(x)
        x = self.act(x)
        x = x.view(-1, 64, 7, 7)
        
        # Upsample
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        
        # Downsample
        return x