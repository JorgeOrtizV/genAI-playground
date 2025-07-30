import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


class ColdDiff:
    def __init__(
        self,
        steps=300,
        size=28,
        loss_type="L1",
        degradation_type="blur",
        device=None,
        blur_sigma=0.33,
    ):

        self.steps = steps
        self.img_size = size
        self.loss_type = loss_type
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.blur_sigma = blur_sigma

        self.beta_schedule = self.get_beta_schedule(self.steps)
        self.alpha = 1.0 - self.beta_schedule
        alphas = torch.cat(
            [torch.tensor([1.0]), self.alpha]
        )  # Add a leading 1.0 to the alphas tensor
        self.alpha_hat = torch.cumprod(alphas, dim=0).to(device)

        # Degradation type can be 'blur', 'pixellate', 'inpainting' or 'snow'
        if degradation_type == "blur":
            self.degradation = self.blur
        # elif degradation_type == 'pixellate':
        #    self.degradation = self.pixellate
        elif degradation_type == "inpainting":
            self.degradation = self.inpainting
        elif degradation_type == "snow":
            self.degradation = self.snow
        elif degradation_type == "gaussian":
            self.degradation = self.gaussian_noise
        else:
            raise ValueError(
                "Invalid degradation type. Choose from 'blur', 'pixellate', 'inpainting' or 'snow'."
            )

    # SAMPLING FUNCTIONS
    def sample_timesteps(self, n):
        return torch.randint(
            low=1, high=self.steps, size=(n,)
        )  # Should this be self.steps+1?

    def sample(
        self,
        model,
        batch_size,
        initial_image="real_degraded",
        data_loader=None,
        gmm=None,
    ):
        output_shape = (batch_size, 1, 28, 28)
        t = self.steps
        with torch.no_grad():
            model.eval()
            # x_prev = x_t
            # black image
            if initial_image == "black":
                x_prev = torch.zeros(output_shape).to(self.device)

            if initial_image == "random":
                # random and degraded
                dummy_x0 = torch.randn(output_shape).to(
                    self.device
                )  # Or: real_batch_from_train_loader
                timestep_tensor = torch.tensor([self.steps] * batch_size).to(
                    self.device
                )
                x_prev = self.degradation(dummy_x0, timestep_tensor)
            if initial_image == "real_degraded":
                if data_loader is None:
                    raise ValueError(
                        "data_loader is not provided. Please provide a DataLoader for sampling."
                    )
                real_batch, _ = next(iter(data_loader))  # get real MNIST digits
                real_batch = real_batch.to(self.device)
                real_batch = real_batch[:batch_size]  # in case batch size mismatch
                timestep_tensor = torch.tensor([self.steps] * real_batch.shape[0]).to(
                    self.device
                )
                x_prev = self.degradation(real_batch, timestep_tensor)

            if initial_image == "gmm":
                # Gaussian Mixture Model
                if gmm is None:
                    raise ValueError(
                        "GMM is not provided. Please provide a GMM model for sampling."
                    )
                # Sample from the GMM
                gmm_samples, _ = gmm.sample(batch_size)  # Sample from the GMM
                gmm_samples = gmm_samples.astype(np.float32)
                gmm_samples = torch.from_numpy(gmm_samples)  # Convert to tensor
                gmm_samples = gmm_samples.view(
                    batch_size, 1, self.img_size, self.img_size
                )  # Reshape to the desired output shape
                gmm_samples = gmm_samples.to(
                    self.device
                )  # Move to the appropriate device
                # Set the initial image to the GMM samples
                x_prev = gmm_samples

            for s in range(t, 0, -1):
                s_ = (torch.ones(batch_size) * s).long().to(self.device)
                pred_x0 = model(x_prev, s_)
                x_prev = (
                    x_prev
                    - self.degradation(pred_x0, s_)
                    + self.degradation(pred_x0, s_ - 1)
                )
        return x_prev

    def sample_ddpm(self, model, batch_size):
        model.eval()
        # Algo 2 - Sampling
        with torch.no_grad():
            x = torch.randn((batch_size, 1, self.img_size, self.img_size)).to(
                self.device
            )
            for i in tqdm(reversed(range(1, self.steps)), position=0):
                t = (torch.ones(batch_size) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta_schedule[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = (
                    1
                    / torch.sqrt(alpha)
                    * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise)
                    + torch.sqrt(beta) * noise
                )
            model.train()
            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)
            return x

    # Normal gaussian noise degradation
    def gaussian_noise(self, x0, t):
        """if len(t.shape) == 1:
            t = t[:, None, None, None]

        sqrt_alpha_bar = self.alpha_bar[t.squeeze().long().clamp(max=self.steps)].sqrt().to(x0.device)[:, None, None, None]
        sqrt_one_minus_alpha_bar = (1 - self.alpha_bar[t.squeeze().long().clamp(max=self.steps)]).sqrt().to(x0.device)[:, None, None, None]

        noise = torch.randn_like(x0)

        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        """

        e = torch.randn_like(x0)
        x_t = (
            x0 * torch.sqrt(self.alpha_hat[t])[:, None, None, None]
            + torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None] * e
        )
        return x_t

    # DEGRADATION FUNCTIONS
    def blur(self, x0, t):
        if self.blur_sigma <= 0:
            return x0
        sigma_t = torch.sqrt(t * self.blur_sigma**2).detach().cpu().numpy()
        x0_cpu = x0.detach().cpu().numpy()
        blurred_imgs = [
            gaussian_filter(x0_cpu[i], sigma=(0, sigma_t[i], sigma_t[i]))
            for i in range(len(sigma_t))
        ]
        return torch.from_numpy(np.stack(blurred_imgs)).to(self.device)
        # return GaussianBlur(5, sigma_t)

    def inpainting(self, x0, t, base_variance=1):  # Base variance is beta in the paper
        w, h = x0.shape[2], x0.shape[3]
        center_x, center_y = np.random.randint(
            0, w, size=x0.shape[0]
        ), np.random.randint(0, h, x0.shape[0])
        variance = base_variance + 0.5 * t.detach().cpu().numpy()
        # 2d gaussian curve with center at rand_x,rand_y and peak value = 1, discretized

        x0 = x0.detach().cpu().numpy()
        gaussian_mask = np.zeros_like(x0, dtype=float)

        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        gaussian = np.stack(
            [
                np.exp(
                    -((x - center_x[i]) ** 2 + (y - center_y[i]) ** 2)
                    / (2 * variance[i])
                )
                for i in range(len(x0))
            ]
        )
        gaussian = 1 - gaussian

        # Normalize the gaussian mask
        gaussian = gaussian / np.max(gaussian, axis=(1, 2), keepdims=True)

        # Add extra channel dimension as 2nd dimension
        gaussian = gaussian[:, None, :, :]

        # Apply the mask to the image
        inpainted_imgs = x0 * gaussian

        return torch.from_numpy(inpainted_imgs).float().to(self.device)

    ## Scheduling
    def get_beta_schedule(
        self, timesteps, start=1e-4, end=0.02, schedule_type="linear"
    ):
        if schedule_type == "linear":
            return torch.linspace(start, end, timesteps)
        elif schedule_type == "cosine":
            steps = torch.arange(timesteps + 1, dtype=torch.float32)
            f = (
                lambda t: torch.cos(((t / timesteps) + 0.008) / 1.008 * torch.pi / 2)
                ** 2
            )
            alphas_bar = f(steps) / f(torch.tensor(0.0))
            betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
            return torch.clip(betas, 1e-5, 0.999)
        else:
            raise ValueError("Unknown schedule type")
