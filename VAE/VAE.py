from torch import nn
import torch

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=200, z_dim=20):
        super().__init__()
        # Encoder
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear_mu = nn.Linear(hidden_dim, z_dim)
        self.linear_sigma = nn.Linear(hidden_dim, z_dim)
        
        # Decoder
        self.linear_2h = nn.Linear(z_dim, hidden_dim)
        self.linear_2img = nn.Linear(hidden_dim, input_dim)
        
        # Multipurpose
        self.relu = nn.ReLU() # LeakyReLU
        self.flat = nn.Flatten()
    
    def encode(self, x):
        #q_phi(z|x)
        #x = self.flat(x)
        z = self.linear1(x)
        z = self.relu(z)
        mu = self.linear_mu(z)
        sigma = self.linear_sigma(z)
        
        return mu, sigma
    
    def decode(self, z):
        # p_theta(x|z)
        h = self.linear_2h(z)
        h = self.relu(h)
        img = self.linear_2img(h)
        
        return torch.sigmoid(img)
    
    def forward(self, x):
        mu, sigma = self.encode(x)
        e = torch.randn_like(sigma)
        z_reparametrized = mu+sigma*e
        x_reconstructed = self.decode(z_reparametrized)
        return x_reconstructed, mu, sigma
    