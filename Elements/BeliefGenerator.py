import torch
import torch.utils.data
from torch import nn

class Generator(nn.Module):
    def __init__(self, feature_size, context_size, latent_size, noise_scale=1):
        super(Generator, self).__init__()
        self.feature_size = feature_size
        self.context_size = context_size  # Will be dynamic
        self.noise_scale = noise_scale

        self.fc1 = nn.Linear(feature_size + context_size, 400)
        self.fc21 = nn.Linear(400, latent_size)  # mu
        self.fc22 = nn.Linear(400, latent_size)  # logvar
        self.fc3 = nn.Linear(latent_size + context_size, 400)
        self.fc4 = nn.Linear(400, feature_size)
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()

    def encode(self, x, c):
        inputs = torch.cat([x, c], 1)
        h1 = self.elu(self.fc1(inputs))
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) * self.noise_scale
        return mu + eps * std

    def decode(self, z, c):
        inputs = torch.cat([z, c], 1)
        h2 = self.fc3(inputs)  # Original code had h2 = self.elu(self.fc3(inputs)), then h3=h2. Simplified.
        h3 = self.elu(h2)
        h4 = self.fc4(h3)
        x_recon = self.tanh(h4)  # Output scaled to [-1, 1]
        return x_recon

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, self.feature_size), c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar
