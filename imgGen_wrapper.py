import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import SubsetRandomSampler, DataLoader
import numpy as np
from matplotlib import pyplot as plt
import argparse
import sys
import torchvision
import os
import subprocess
import math
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
import pickle
import time
from itertools import product
from sklearn.mixture import GaussianMixture

# import scripts
from DDPM.Diffusion import Diffusion
from DDPM.Unet import UNet
from VAE.VAE import VAE
from GAN.Generator import Generator
from GAN.Discriminator import Discriminator
from ColdDiffusion.ColdDiffusion import ColdDiff
from LDM.LDM import LatentDiffusionModel, Encoder, Decoder


# Seed numpy
np.random.seed(32)


def argparser(args):
    parser = argparse.ArgumentParser()
    model_selection = parser.add_argument_group()
    model_params = parser.add_argument_group()
    training_params = parser.add_argument_group()
    model_eval = parser.add_argument_group()
    dataset_params = parser.add_argument_group()

    model_selection.add_argument(
        "--model",
        type=str,
        dest="model_sel",
        default="DDPM",
        help="Selection of Generative Model. Available options are: DDPM, GAN, LDM, EBM, VAE, ColdDiff",
        required=True,
    )

    model_params.add_argument(
        "--noise-steps",
        type=int,
        nargs="+",
        dest="noise_steps",
        default=[1000],
        help="Noise steps to be used in the DDPM forward and backward processes.",
    )
    model_params.add_argument(
        "--beta-start",
        type=float,
        nargs="+",
        dest="beta_start",
        default=[1e-4],
        help="Set the beta start parameter for DDPM",
    )
    model_params.add_argument(
        "--beta-end",
        type=float,
        nargs="+",
        default=[0.02],
        dest="beta_end",
        help="Set the beta end parameter for DDPM",
    )
    model_params.add_argument(
        "--latent-dim",
        nargs="+",
        default=[100],
        type=int,
        dest="latent_dim",
        help="Selects latent dimension size. Required parameter for GANs",
    )
    model_params.add_argument(
        "--hidden-dim",
        nargs="+",
        default=[200],
        type=int,
        dest="hidden_dim",
        help="Hidden dimension for VAEs",
    )
    model_params.add_argument(
        "--z-dim",
        nargs="+",
        dest="z_dim",
        default=[20],
        type=int,
        help="Sets bottleneck dimension for VAEs",
    )
    model_params.add_argument(
        "--blur-sigma",
        nargs="+",
        default=[0.3],
        type=float,
        dest="blur_sigma",
        help="Set the blur sigma parameter for Cold Diffusion (default: 0.3)",
    )

    training_params.add_argument(
        "--epochs",
        type=int,
        dest="epochs",
        default=100,
        help="Set the number of training iterations",
    )
    training_params.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        dest="batch_size",
        default=[32],
        help="Selects batch size for training.",
    )
    training_params.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        nargs="+",
        default=[3e-4],
        dest="lr",
        help="Set training learning rate",
    )
    # TODO: Add optimizer/loss options if necessary. If not remove the parameter
    training_params.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        dest="opt",
        help="Select optimizer to be used for training. Accepted values: Adam, SGD, RMSprop, AdamW, ...",
    )
    training_params.add_argument(
        "--loss",
        type=str,
        default="MSE",
        dest="loss",
        help="Select loss function for model training. Accepted values: MSE, L1, BCE, ...",
    )
    training_params.add_argument(
        "--model-output",
        type=str,
        required=True,
        dest="model_output_dir",
        help="Provide a directory to story your trained model.",
    )
    training_params.add_argument(
        "--verbose",
        "-v",
        dest="v",
        action="store_const",
        const=True,
        default=False,
        help="If given activates verbose mode.",
    )
    training_params.add_argument(
        "--random-seed",
        dest="random_seed",
        default=32,
        type=int,
        help="If provided np.random.seed is fixed to the given value. Default value is 32.",
    )

    # TODO: enable give a path for train, test, val datasets
    dataset_params.add_argument(
        "--MNIST",
        dest="MNIST",
        action="store_const",
        const=True,
        default=False,
        help="Selects MNIST dataset for training, validation, and test",
    )
    dataset_params.add_argument(
        "--train-dataset",
        dest="train_dataset",
        default=None,
        type=str,
        help="Give the path for a folder to use as training dataset",
    )
    dataset_params.add_argument(
        "--test-dataset",
        dest="test_dataset",
        default=None,
        type=str,
        help="Give the path for a folder to use as test dataset",
    )
    dataset_params.add_argument(
        "--val-dataset",
        dest="val_dataset",
        default=None,
        type=str,
        help="Give the path for a folder to use as validation dataset",
    )
    dataset_params.add_argument(
        "--img-size",
        dest="img_size",
        required=True,
        type=int,
        help="Provide the size of the images used for training and therefore generation size. Please use square images",
    )
    dataset_params.add_argument(
        "--img-channels",
        dest="img_channels",
        type=int,
        default=1,
        help="If custom dataset, provide if your images are RGB or grayscale by indicating the number of channels.",
    )

    # TODO: Add options for evaluation. E.g. FID
    model_eval.add_argument(
        "--eval",
        dest="eval_method",
        default=None,
        type=str,
        help="Provide the evaluation method to be used once the model was trained. Available options: FID",
    )
    model_eval.add_argument(
        "--eval-steps",
        dest="eval_steps",
        default=None,
        type=int,
        help="Evaluate the model during training every n steps. If not given model is just evaluated until training is completed.",
    )
    model_eval.add_argument(
        "--eval-samples",
        dest="eval_samples",
        default=None,
        type=int,
        help="Proved the number of samples N to be generated to evaluate model distribution against N samples of data distribution",
    )

    argsvalue = parser.parse_args(args)
    return argsvalue


def split_indices(n, val_pct):
    n_val = int(val_pct * n)
    idxs = np.random.permutation(n)
    return idxs[n_val:], idxs[:n_val]


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(
        torch.cat([torch.cat([i for i in images.cpu()], dim=-1)], dim=-2)
        .permute(1, 2, 0)
        .cpu(),
        cmap="gray",
    )
    plt.show()


def train(
    epochs,
    train_loader,
    val_loader,
    test_dataset,
    model,
    optimizer,
    diffusion,
    device,
    loss_fn,
    verbose,
    eval,
    eval_steps,
    eval_samples,
    batch_size,
):
    train_losses = []
    val_losses = []
    fids = []

    sample_dir = "data/eval_sampled"
    test_dir = "data/eval_test"
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for epoch in tqdm(range(epochs)):
        train_loss = 0
        for x, _ in train_loader:
            optimizer.zero_grad()
            x = x.to(device)
            t = diffusion.sample_timesteps(x.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(x, t)

            predicted_noise = model(x_t, t)
            loss = loss_fn(noise, predicted_noise)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        val_loss = 0
        with torch.no_grad():
            model.eval()
            for x, _ in val_loader:
                # print(type(x))
                # print(x.shape)
                x = x.to(device)
                t = diffusion.sample_timesteps(x.shape[0]).to(device)
                x_t, noise = diffusion.noise_images(x, t)
                predicted_noise = model(x_t, t)
                batch_val_loss = loss_fn(noise, predicted_noise)
                val_loss += batch_val_loss.item()

        # Sample only every 3 epochs for opt purposes
        if verbose:
            if epoch % 5 == 0:
                sampled_images = diffusion.sample(model, n=x.shape[0])
                plot_images(sampled_images)

        if eval:
            it = math.ceil(eval_samples / batch_size)
            # Remove all existing elements from saved folders
            for folder in [sample_dir, test_dir]:
                for f in os.listdir(folder):
                    os.remove(os.path.join(folder, f))
            if epoch % eval_steps == 0:
                for j in range(it):
                    sampled_images = diffusion.sample(model, n=x.shape[0])
                    for i, img in enumerate(sampled_images):
                        if j * batch_size + i >= eval_samples:
                            break
                        save_image(
                            img, os.path.join(sample_dir, f"{j*batch_size+i}.png")
                        )
                test_dl = torch.utils.data.DataLoader(
                    test_dataset, batch_size=eval_samples, shuffle=True
                )
                test_batch = next(iter(test_dl))[0][:eval_samples]
                # Save sampled images and original images
                for i, img in enumerate(test_batch):
                    save_image(img, os.path.join(test_dir, f"{i}.png"))
                # Calculate FID
                fid_proc = subprocess.run(
                    ["python", "-m", "pytorch_fid", sample_dir, test_dir],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                fid_line = fid_proc.stdout.splitlines()[-1]
                fid_value = float(fid_line.strip().split()[-1])
                fids.append(fid_value)
                if verbose:
                    print(f"FID: {fid_value}")

        # Save loss
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        if verbose:
            print(
                f"Epoch [{epoch+1}/100] | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}"
            )
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

    return fids, train_losses, val_losses


def train_gan(
    epochs,
    train_loader,
    val_loader,
    test_dataset,
    generator,
    discriminator,
    opt_g,
    opt_d,
    device,
    loss_fn,
    latent_dim,
    verbose,
    eval,
    eval_steps,
    eval_samples,
    batch_size,
):
    train_losses = []
    val_losses = []
    fids = []

    for epoch in tqdm(range(epochs)):
        sample_dir = "data/eval_sampled"
        test_dir = "data/eval_test"
        os.makedirs(sample_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        epoch_d_losses = []
        epoch_g_losses = []

        for x, _ in train_loader:
            x_real = x.to(device)
            batch_size = x_real.size(0)

            # === Train Discriminator ===
            opt_d.zero_grad()
            y_real = torch.ones(batch_size, 1, device=device) * 0.9  # label smoothing
            y_fake = torch.zeros(batch_size, 1, device=device)

            # Real samples
            prediction_real = discriminator(x_real)
            loss_real = loss_fn(prediction_real, y_real)

            # Fake samples
            z = torch.randn(batch_size, latent_dim, device=device)
            x_fake = generator(z)
            prediction_fake = discriminator(x_fake.detach())
            loss_fake = loss_fn(prediction_fake, y_fake)

            loss_d = (loss_real + loss_fake) / 2
            loss_d.backward()
            opt_d.step()
            epoch_d_losses.append(loss_d.item())

            # === Train Generator ===
            if epoch % 2 == 0:
                opt_g.zero_grad()
                y_gen = torch.ones(batch_size, 1, device=device)
                prediction = discriminator(x_fake)
                loss_g = loss_fn(prediction, y_gen)
                loss_g.backward()
                opt_g.step()
                epoch_g_losses.append(loss_g.item())

        avg_d_loss = sum(epoch_d_losses) / len(epoch_d_losses)
        avg_g_loss = (
            sum(epoch_g_losses) / len(epoch_g_losses) if epoch_g_losses else 0.0
        )
        train_losses.append((avg_g_loss, avg_d_loss))

        if eval:
            it = math.ceil(eval_samples / batch_size)
            # Remove all existing elements from saved folders
            for folder in [sample_dir, test_dir]:
                for f in os.listdir(folder):
                    os.remove(os.path.join(folder, f))
            if epoch % eval_steps == 0:
                for j in range(it):
                    z = torch.randn(batch_size, latent_dim, device=device)
                    sampled_images = generator(z)
                    for i, img in enumerate(sampled_images):
                        if j * batch_size + i >= eval_samples:
                            break
                        save_image(
                            img, os.path.join(sample_dir, f"{j*batch_size+i}.png")
                        )
                test_dl = torch.utils.data.DataLoader(
                    test_dataset, batch_size=eval_samples, shuffle=True
                )
                test_batch = next(iter(test_dl))[0][:eval_samples]
                # Save sampled images and original images
                for i, img in enumerate(test_batch):
                    save_image(img, os.path.join(test_dir, f"{i}.png"))
                # Calculate FID
                fid_proc = subprocess.run(
                    ["python", "-m", "pytorch_fid", sample_dir, test_dir],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                fid_line = fid_proc.stdout.splitlines()[-1]
                fid_value = float(fid_line.strip().split()[-1])
                fids.append(fid_value)
                if verbose:
                    print(f"FID: {fid_value}")

        print(
            f"Epoch [{epoch+1}/{epochs}] | D Loss: {loss_d.item():.4f} | G Loss: {loss_g.item():.4f}"
        )

        if verbose and epoch % 5 == 0:
            with torch.no_grad():
                z = torch.randn(6, latent_dim, device=device)
                generated_imgs = generator(z)
                plot_images(generated_imgs)

    return fids, train_losses, []


def train_vae(
    epochs,
    train_loader,
    val_loader,
    test_dataset,
    model,
    optimizer,
    device,
    loss_fn,
    verbose,
    eval,
    eval_steps,
    eval_samples,
    batch_size,
    img_size,
):
    losses = []
    fids = []

    sample_dir = "data/eval_sampled"
    test_dir = "data/eval_test"
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for epoch in tqdm(range(epochs)):
        for x, _ in train_loader:
            x = x.to(device)
            x = x.view(x.size(0), img_size**2)
            optimizer.zero_grad()

            out, mu, sigma = model(x)
            reconstruction_loss = loss_fn(out, x)
            # KL_div = -0.5 * torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2, dim=1)
            # KL_div = -0.5*torch.sum(1+torch.log(torch.pow(sigma, 2)) - torch.pow(mu, 2) - torch.pow(sigma, 2)) # Minimize KL_div
            KL_div = (
                -0.5
                * torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2, dim=1).mean()
            )

            # backward
            loss = reconstruction_loss + KL_div
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print("Epoch {}/{} - Loss: {:.4f}".format(epoch + 1, epochs, loss.item()))

        if eval:
            it = math.ceil(eval_samples / batch_size)
            # Remove all existing elements from saved folders
            for folder in [sample_dir, test_dir]:
                for f in os.listdir(folder):
                    os.remove(os.path.join(folder, f))
            if epoch % eval_steps == 0:
                with torch.no_grad():
                    for j in range(it):
                        z = torch.randn(batch_size, model.latent_dim, device=device)
                        generated = model.decode(z).view(
                            -1, 1, int(img_size**0.5), int(img_size**0.5)
                        )
                        for i, img in enumerate(generated):
                            if j * batch_size + i >= eval_samples:
                                break
                            save_image(
                                img, os.path.join(sample_dir, f"{j*batch_size+i}.png")
                            )
                test_dl = torch.utils.data.DataLoader(
                    test_dataset, batch_size=eval_samples, shuffle=True
                )
                test_batch = next(iter(test_dl))[0][:eval_samples]
                # Save sampled images and original images
                for i, img in enumerate(test_batch):
                    save_image(img, os.path.join(test_dir, f"{i}.png"))
                # Calculate FID
                fid_proc = subprocess.run(
                    ["python", "-m", "pytorch_fid", sample_dir, test_dir],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                fid_line = fid_proc.stdout.splitlines()[-1]
                fid_value = float(fid_line.strip().split()[-1])
                fids.append(fid_value)
                if verbose:
                    print(f"FID: {fid_value}")

    return fids, losses, []


def train_coldDiff(
    epochs,
    train_loader,
    val_loader,
    test_dataset,
    model,
    optimizer,
    diffusion,
    device,
    loss_fn,
    eval,
    eval_steps,
    eval_samples,
    batch_size,
    steps,
    blur_sigma,
):
    FIT_GMM = True

    if FIT_GMM:
        all_blurred = []

        for x, _ in train_loader:
            x = x.to(device)
            batch_size_t = x.size(0)

            t = (
                torch.full((batch_size_t,), steps).long().to(device)
            )  # Full degradation (T steps)
            x_blurred = diffusion.degradation(x, t)

            all_blurred.append(x_blurred.cpu())

        # Stack all blurred images
        all_blurred = torch.cat(all_blurred, dim=0)  # shape: [N, 1, 28, 28]
        all_blurred = all_blurred.squeeze(1)  # [N, 28, 28]

        X = all_blurred.reshape(all_blurred.shape[0], -1)  # [N, 784]

        # Choose number of components (1–5 is usually enough for MNIST)
        gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=0)
        print("Fitting GMM to blurred images")
        try:
            gmm.fit(X.numpy())  # convert to numpy
        except Exception as e:
            print(f"Error fitting GMM: {e}")
            raise ValueError(
                "GMM fitting failed. Please check the input data and parameters."
            )

        print("GMM fitted successfully")

        # Save the GMM model
        # print("Saving GMM model")
        # Path("./gmm_models").mkdir(parents=True, exist_ok=True)
        # joblib.dump(
        #     gmm,
        #     f"./gmm_models/gmm_mnist_{degradation_type}_{blur_sigma}_{steps}_{num_samples}.pkl",
        # )
        fids = []
        train_losses = []
        val_losses = []
        # TRAINING LOOP
        for epoch in range(epochs):
            print("Epoch ", epoch)
            train_loss = 0
            model.train()
            for x, _ in tqdm(train_loader):
                optimizer.zero_grad()
                x = x.to(device)
                t = diffusion.sample_timesteps(x.shape[0]).to(device)
                x_t = noise_function(x, t)
                pred = model(x_t, t)
                loss = loss_fn(pred, x)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            val_loss = 0

            if eval:
                model.eval()
                with torch.no_grad():
                    for x, _ in val_loader:
                        x = x.to(device)
                        t = diffusion.sample_timesteps(x.shape[0]).to(device)
                        x_t = noise_function(x, t)
                        pred = model(x_t, t)
                        batch_val_loss = loss_fn(pred, x)
                        val_loss += batch_val_loss.item()
                        del x, x_t, t, pred, batch_val_loss

                # FID evaluation
                it = math.ceil(eval_samples / batch_size)
                sample_dir = "data/eval_sampled"
                test_dir = "data/eval_test"
                os.makedirs(sample_dir, exist_ok=True)
                os.makedirs(test_dir, exist_ok=True)
                # Remove all existing elements from saved folders
                for folder in [sample_dir, test_dir]:
                    for f in os.listdir(folder):
                        os.remove(os.path.join(folder, f))
                if epoch % eval_steps == 0:
                    for j in range(it):
                        sampled_images = diffusion.sample(
                            model,
                            batch_size=batch_size,
                            initial_image="GMM",
                            gmm=gmm,
                            data_loader=val_loader,
                        )
                        for i, img in enumerate(sampled_images):
                            if j * batch_size + i >= eval_samples:
                                break
                            save_image(
                                img, os.path.join(sample_dir, f"{j*batch_size+i}.png")
                            )
                    test_dl = torch.utils.data.DataLoader(
                        test_dataset, batch_size=eval_samples, shuffle=True
                    )
                    test_batch = next(iter(test_dl))[0][:eval_samples]
                    # Save sampled images and original images
                    for i, img in enumerate(test_batch):
                        save_image(img, os.path.join(test_dir, f"{i}.png"))
                    # Calculate FID
                    fid_proc = subprocess.run(
                        ["python", "-m", "pytorch_fid", sample_dir, test_dir],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    fid_line = fid_proc.stdout.splitlines()[-1]
                    fid_value = float(fid_line.strip().split()[-1])
                    fids.append(fid_value)
                    print(f"FID: {fid_value}")

            # Save loss
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            print(
                f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f} | Learning Rate: {scheduler.get_last_lr()[0]:.10f}"
            )

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

    return fids, train_losses, val_losses


def train_ldm(
    epochs,
    train_loader,
    val_loader,
    test_dataset,
    model,
    optimizer,
    diffusion,
    device,
    loss_fn,
    eval,
    eval_steps,
    eval_samples,
    batch_size,
    steps,
    latent_dim,
):
    # Autoencoder training loop
    print("Starting Autoencoder Training...")

    ae_train_losses = []
    ae_val_losses = []

    autoencoder_optimizer = torch.optim.Adam(model.autoencoder.parameters(), lr=1e-4)
    reconstruction_loss = nn.MSELoss()
    autoencoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        autoencoder_optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    autoencoder_epochs = 20

    for epoch in range(autoencoder_epochs):
        running_loss = 0.0
        model.train()
        # Training loop
        for batch_idx, (batch, _) in enumerate(train_loader):
            batch = batch.to(device)
            autoencoder_optimizer.zero_grad()
            recon = model.autoencoder_forward(batch)
            # print(batch.size(),recon.size())
            loss = reconstruction_loss(recon, batch)
            loss.backward()
            autoencoder_optimizer.step()
            running_loss += loss.item()

        # Validation loop
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for batch, _ in val_loader:
                batch = batch.to(device)
                recon = model.autoencoder_forward(batch)
                val_loss += reconstruction_loss(recon, batch).item()
                num_val_batches += 1

        avg_val_loss = val_loss / num_val_batches
        print(
            f"Epoch [{epoch+1}/{autoencoder_epochs}] | Autoencoder Validation Loss: {avg_val_loss:.6f}"
        )

        ae_train_losses.append(running_loss / len(train_loader))
        ae_val_losses.append(avg_val_loss)
        autoencoder_scheduler.step(avg_val_loss)

        print("Autoencoder Training Completed!")

        # Save the losses
        # torch.save(
        #    {
        #        "ae_train_losses": ae_train_losses,
        #        "ae_val_losses": ae_val_losses,
        #    },
        #    f"{store_path}/autoencoder_losses_{steps}_{latent_channels}.pth",
        # )

        print("\nStarting Diffusion Model Training...")

        # Freeze autoencoder weights
        model.freeze_autoencoder()

        # Diffusion training loop
        diff_train_losses = []
        diff_val_losses = []
        fids = []

        for epoch in range(epochs):
            running_loss = 0.0
            model.train()
            # Training loop
            for batch_idx, (batch, _) in enumerate(train_loader):
                batch = batch.to(device)
                diffusion_optimizer.zero_grad()
                t = diffusion.sample_timesteps(batch.shape[0]).to(device)
                output = model(batch, t)
                loss = loss_fn(output, batch)
                loss.backward()
                diffusion_optimizer.step()
                running_loss += loss.item()

            # Validation loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch, _ in val_loader:
                    batch = batch.to(device)
                    t = diffusion.sample_timesteps(batch.shape[0]).to(device)
                    output = model(batch, t)
                    val_loss += loss_fn(output, batch).item()

            avg_val_loss = val_loss / len(val_loader)
            diff_val_losses.append(avg_val_loss)
            # Plot images
            if epoch % 10 == 0:
                print(f"Epoch {epoch}")
                # plot_images(batch)
                # Sample images
                sampled_images_list = []
                num_samples = eval_samples if eval_samples is not None else 1000
                batch_size_sampling = batch_size
                sample_dir = "data/eval_sampled"
                test_dir = "data/eval_test"
                os.makedirs(sample_dir, exist_ok=True)
                os.makedirs(test_dir, exist_ok=True)
                # Remove all existing elements from saved folders
                for folder in [sample_dir, test_dir]:
                    for f in os.listdir(folder):
                        os.remove(os.path.join(folder, f))
                for i in range(0, num_samples, batch_size_sampling):
                    curr_batch_size = min(batch_size_sampling, num_samples - i)
                    rand_x = torch.randn((curr_batch_size, 1, 28, 28)).to(device)
                    t = torch.ones(curr_batch_size, dtype=torch.long).to(device) * steps
                    sample_images = model(rand_x, t)
                    sampled_images = (
                        sample_images.clamp(-1, 1) + 1
                    ) / 2  # Normalize to [0, 1]
                    for j, img in enumerate(sampled_images):
                        save_image(img, os.path.join(sample_dir, f"{i+j}.png"))
                    sampled_images_list.append(sampled_images.cpu())
                all_sampled_images = torch.cat(sampled_images_list, dim=0)
                # Save test images
                test_dl = torch.utils.data.DataLoader(
                    test_dataset, batch_size=num_samples, shuffle=True
                )
                test_batch = next(iter(test_dl))[0][:num_samples]
                for i, img in enumerate(test_batch):
                    save_image(img, os.path.join(test_dir, f"{i}.png"))
                # Calculate FID
                fid_proc = subprocess.run(
                    ["python", "-m", "pytorch_fid", sample_dir, test_dir],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                fid_line = fid_proc.stdout.splitlines()[-1]
                try:
                    fid_value = float(fid_line.strip().split()[-1])
                    fids.append(fid_value)
                    print(f"FID: {fid_value}")
                except Exception as e:
                    print(f"Could not parse FID value: {fid_line} ({e})")

            print(
                f"Epoch [{epoch+1}/{epochs}] | Diffusion Validation Loss: {avg_val_loss:.6f} | Learning Rate: {diffusion_optimizer.param_groups[0]['lr']:.10f}"
            )
            diff_train_losses.append(running_loss / len(train_loader))

            diffusion_scheduler.step(avg_val_loss)

    print("Diffusion Model Training Completed!")
    return fids, diff_train_losses, diff_val_losses


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model_sel == "DDPM":
        param_grid = {
            "noise_steps": args.noise_steps,
            "beta_start": args.beta_start,
            "beta_end": args.beta_end,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
        }
    elif args.model_sel == "GAN":
        param_grid = {
            "latent_dim": args.latent_dim,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
        }
    elif args.model_sel == "VAE":
        param_grid = {
            "hidden_dim": args.hidden_dim,
            "z_dim": args.z_dim,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
        }
    elif args.model_sel == "LDM":
        param_grid = {
            "noise_steps": args.noise_steps,
            "latent_dim": args.latent_dim,
            "beta_start": args.beta_start,
            "beta_end": args.beta_end,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
        }
    elif args.model_sel == "ColdDiff":
        param_grid = {
            "noise_steps": args.noise_steps,
            "blur_sigma": args.blur_sigma,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
        }

    keys, values = zip(*param_grid.items())
    runs = list(product(*values))
    for run_id, combo in enumerate(runs):
        config = dict(zip(keys, combo))
        print(f"Running config {run_id+1}/{len(runs)}: {config}")

        batch_size = config["batch_size"]
        learning_rate = config["learning_rate"]

        # Retrieve dataset
        if args.MNIST:
            transforms = torchvision.transforms.Compose(
                [
                    # torchvision.transforms.Resize(80),
                    # torchvision.transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5,), (0.5,)),
                ]
            )
            fid_transform = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
            )
            dataset = torchvision.datasets.MNIST(
                root="./data", train=True, transform=transforms, download=True
            )
            test_dataset = torchvision.datasets.MNIST(
                root="./data", train=False, transform=fid_transform, download=True
            )

            train_indices, val_indices = split_indices(len(dataset), 0.2)
            train_sampler = SubsetRandomSampler(train_indices)
            train_loader = torch.utils.data.DataLoader(
                dataset, batch_size, sampler=train_sampler
            )
            val_sampler = SubsetRandomSampler(val_indices)
            val_loader = DataLoader(dataset, batch_size, sampler=val_sampler)

            print("MNIST dataset loaded")

        elif (
            args.train_dataset != None
            and args.test_dataset != None
            and args.val_dataset != None
        ):
            from customDataset import customDataset

            if args.img_channels == 1:
                transforms = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.Resize((args.img_size, args.img_size)),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(
                            (0.5,), (0.5,)
                        ),  # For grayscale; change to (0.5, 0.5, 0.5) if RGB
                    ]
                )
            else:
                transforms = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.Resize((args.img_size, args.img_size)),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(
                            (0.5,), (0.5,), (0.5)
                        ),  # For grayscale; change to (0.5, 0.5, 0.5) if RGB
                    ]
                )

            fid_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize((args.img_size, args.img_size)),
                    torchvision.transforms.ToTensor(),
                ]
            )

            train_dataset = customDataset(
                root_dir=args.train_dataset,
                transform=transforms,
                channels=args.img_channels,
            )
            val_dataset = customDataset(
                root_dir=args.val_dataset,
                transform=transforms,
                channels=args.img_channels,
            )
            test_dataset = customDataset(
                root_dir=args.test_dataset,
                transform=fid_transform,
                channels=args.img_channels,
            )

            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            print("Custom datasets loaded from provided paths.")

        else:
            print(
                "Error retreiving train, test, and validation datasets. Please double check you provided a right path"
            )
            raise

        # Init model
        if args.model_sel == "DDPM" or args.model_sel == "ColdDiff":
            model = UNet(
                device=device, c_in=args.img_channels, c_out=args.img_channels
            ).to(device)
        elif args.model_sel == "LDM":
            pass
        elif args.model_sel == "GAN":
            # Due to adversarial training, this implementation usses a separate training func
            pass
        elif args.model_sel == "VAE":
            hidden_dim = config["hidden_dim"]
            z_dim = config["z_dim"]
            model = VAE(
                input_dim=args.img_size**2, hidden_dim=hidden_dim, z_dim=z_dim
            ).to(device)
        else:
            print("Given model is not available")
            raise ValueError(f"Unsupported model: {args.model_sel}")
        # Init optimizer
        if args.model_sel != "GAN":
            if args.opt == "Adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            elif args.opt == "SGD":
                optimizer = torch.optim.SGD(
                    model.parameters(), lr=learning_rate, momentum=0.9
                )
            elif args.opt == "RMSprop":
                optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
            elif args.opt == "AdamW":
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            else:
                print(f"Given optimizer '{args.opt}' is not available")
            raise ValueError(f"Unsupported optimizer: {args.opt}")
        elif args.model_sel == "GAN":
            # GAN optimizers are initialized later for generator/discriminator
            pass
        else:
            print("Given optimizer is not available")
            raise ValueError(f"Unsupported optimizer: {args.opt}")

        # Init loss fn
        if args.loss == "MSE":
            loss_fn = nn.MSELoss()
        elif args.loss == "BCE" and args.model_sel == "GAN":
            loss_fn = nn.BCEWithLogitsLoss()
        elif args.loss == "BCE" and args.model_sel == "VAE":
            loss_fn = nn.BCELoss(reduction="sum")
        elif args.loss == "L1":
            loss_fn = nn.L1Loss()
        else:
            print("Given loss function is not available")
            raise

        # Init diffusion or any other model
        if args.model_sel == "DDPM":
            noise_steps = config["noise_steps"]
            beta_start = config["beta_start"]
            beta_end = config["beta_end"]
            diffusion = Diffusion(
                noise_steps=noise_steps,
                beta_start=beta_start,
                beta_end=beta_end,
                img_size=args.img_size,
                device=device,
            )
            fids, train_losses, val_losses = train(
                args.epochs,
                train_loader,
                val_loader,
                test_dataset,
                model,
                optimizer,
                diffusion,
                device,
                loss_fn,
                args.v,
                args.eval_method,
                args.eval_steps,
                args.eval_samples,
                batch_size,
            )

        elif args.model_sel == "GAN":
            from GAN.Generator import Generator
            from GAN.Discriminator import Discriminator

            latent_dim = config["latent_dim"]

            generator = Generator(latent_dim=latent_dim).to(device)
            discriminator = Discriminator().to(device)

            opt_g = torch.optim.Adam(generator.parameters(), lr=learning_rate)
            opt_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

            fids, train_losses, val_losses = train_gan(
                args.epochs,
                train_loader,
                val_loader,
                test_dataset,
                generator,
                discriminator,
                opt_g,
                opt_d,
                device,
                loss_fn,
                latent_dim,
                args.v,
                args.eval_method,
                args.eval_steps,
                args.eval_samples,
                batch_size,
            )

            model = generator

        elif args.model_sel == "VAE":
            fids, train_losses, val_losses = train_vae(
                args.epochs,
                train_loader,
                val_loader,
                test_dataset,
                model,
                optimizer,
                device,
                loss_fn,
                args.v,
                args.eval_method,
                args.eval_steps,
                args.eval_samples,
                batch_size,
                args.img_size,
            )
        elif args.model_sel == "ColdDiff":
            noise_steps = config["noise_steps"]
            blur_sigma = config["blur_sigma"]
            diffusion = ColdDiff(
                noise_steps=noise_steps,
                blur_sigma=blur_sigma,
                img_size=args.img_size,
                device=device,
            )
            fids, train_losses, val_losses = train_coldDiff(
                args.epochs,
                train_loader,
                val_loader,
                test_dataset,
                model,
                optimizer,
                diffusion,
                device,
                loss_fn,
                args.eval_method,
                args.eval_steps,
                args.eval_samples,
                batch_size,
                noise_steps,
                blur_sigma,
            )

        elif args.model_sel == "LDM":
            pass

        # fids, train_losses, val_losses = train(args.epochs, train_loader, val_loader, test_dataset, model, optimizer, diffusion, device, loss_fn, args.v, args.eval_method, args.eval_steps, args.eval_samples, batch_size)

        # Save outputs
        t = time.localtime()
        timestamp = time.strftime("%b-%d-%Y_%H%M", t)
        os.makedirs(args.model_output_dir, exist_ok=True)
        torch.save(
            model.state_dict(),
            os.path.join(args.model_output_dir, f"output_model_{timestamp}.pt"),
        )
        out_dir = "data/temp"
        os.makedirs(out_dir, exist_ok=True)
        with open(out_dir + f"/train_losses_{timestamp}.pkl", "wb") as f:
            pickle.dump(train_losses, f)
        with open(out_dir + f"/val_losses_{timestamp}.pkl", "wb") as f:
            pickle.dump(val_losses, f)
        if args.eval_method:
            with open(out_dir + f"/fids_{timestamp}.pkl", "wb") as f:
                pickle.dump(fids, f)


if __name__ == "__main__":
    argsvalue = argparser(sys.argv[1:])
    main(argsvalue)
