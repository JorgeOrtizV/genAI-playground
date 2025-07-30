import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels=1, latent_channels=4):
        super().__init__()
        # Input: [batch_size, 1, 28, 28] -> Output: [batch_size, 4, 8, 8]
        self.conv1 = nn.Conv2d(
            in_channels, 32, kernel_size=4, stride=2, padding=1
        )  # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=4, stride=2, padding=2
        )  # 14x14 -> 8x8
        self.conv3 = nn.Conv2d(
            64, latent_channels, kernel_size=3, padding=1
        )  # 8x8 -> 8x8

        self.norm1 = nn.GroupNorm(8, 32)
        self.norm2 = nn.GroupNorm(8, 64)
        self.norm3 = nn.GroupNorm(1, latent_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)

        x = self.conv3(x)
        x = self.norm3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_channels=4, out_channels=1):
        super().__init__()
        # Decoder architecture
        # Input: [batch_size, 4, 8, 8] -> Output: [batch_size, 1, 28, 28]
        self.conv1 = nn.Conv2d(latent_channels, 64, kernel_size=3, padding=1)  # 8x8
        self.conv2 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=2
        )  # 14x14
        self.conv3 = nn.ConvTranspose2d(
            32, out_channels, kernel_size=4, stride=2, padding=1
        )  # 28x28

        self.norm1 = nn.GroupNorm(8, 64)
        self.norm2 = nn.GroupNorm(8, 32)
        self.act = nn.SiLU()

    def forward(self, x):
        # First convolution maintains spatial dimensions
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        # print("X_1:",x.size())

        # First upsample: 8x8 -> 16x16
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        # print("X_2:",x.size())

        # Second upsample: 16x16 -> 28x28
        x = self.conv3(x)
        # print("X_3:",x.size())
        return torch.tanh(x)  # normalize output to [-1, 1]


class LatentDiffusionModel(nn.Module):
    def __init__(self, encoder, decoder, diffusion_unet, diffusion):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.unet = diffusion_unet
        self.diffusion = (
            diffusion  # cold diff or normal diff based on the instance sent
        )

    def freeze_autoencoder(self):
        """Freeze encoder and decoder weights"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

    def unfreeze_autoencoder(self):
        """Unfreeze encoder and decoder weights"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.decoder.parameters():
            param.requires_grad = True

    def forward(self, x, t):

        z = self.encoder(x)  # encode to latent
        # plot_images(z)
        z_noised = self.diffusion.degradation(
            z, t
        )  # apply diffusion noise in latent space
        # plot_images(z_noised)
        z_denoised = self.unet(z_noised, t)
        # plot_images(z_denoised)
        x_recon = self.decoder(z_denoised)

        return x_recon

    def encode(self, x):
        """Encode images to latent space"""
        return self.encoder(x)

    def decode(self, z):
        """Decode from latent space to image space"""
        return self.decoder(z)

    def autoencoder_forward(self, x):
        """Forward pass through just the autoencoder"""
        z = self.encoder(x)
        return self.decoder(z)

    def sample(self, model, batch_size):
        model.eval()
        # Algo 1 - Sampling
        with torch.no_grad():
            # Sample from random noise in latent space
            samples = torch.randn((batch_size, 4, 8, 8)).to(self.unet.device)
            # print("Samples:", samples.size())
            x_prev = samples
            t = self.steps - 1
            for i in range(self.diffusion.steps):
                t = torch.full(
                    (batch_size,), t, device=self.unet.device, dtype=torch.long
                )
                # print("T:", t.size())
                x_prev = self.unet(x_prev, t)
                # print("X_prev:", x_prev.size())
                if i != self.diffusion.steps - 1:
                    x_prev = self.diffusion.noise(x_prev, t)
            # Decode the final latent sample to image space
            x_samples = self.decoder(x_prev)
            # print("X_samples:", x_samples.size())
            return x_samples
