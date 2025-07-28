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

# import scripts
from DDPM.Diffusion import Diffusion
from DDPM.Unet import UNet


# Seed numpy
np.random.seed(32)


def argparser(args):
    parser = argparse.ArgumentParser()
    model_selection = parser.add_argument_group()
    model_params = parser.add_argument_group()
    training_params = parser.add_argument_group()
    model_eval = parser.add_argument_group()
    dataset_params = parser.add_argument_group()

    model_selection.add_argument("--model", type=str, dest='model_sel', default='DDPM', help="Selection of Generative Model. Available options are: DDPM, GAN, LDM, EBM", required=True)

    model_params.add_argument("--noise-steps", type=int, nargs='+', dest="noise_steps", default=[1000], help="Noise steps to be used in the DDPM forward and backward processes.")
    model_params.add_argument("--beta-start", type=float, nargs='+', dest="beta_start", default=[1e-4], help="Set the beta start parameter for DDPM")
    model_params.add_argument("--beta-end", type=float, nargs='+', default=[0.02], dest="beta_end", help="Set the beta end parameter for DDPM")

    training_params.add_argument("--epochs", type=int, dest="epochs", default=100, help="Set the number of training iterations")
    training_params.add_argument("--batch-size", type=int, nargs='+', dest="batch_size", default=[32], help="Selects batch size for training.")
    training_params.add_argument("--learning-rate", "--lr", type=float, nargs='+', default=[3e-4], dest="lr", help="Set training learning rate")
    # TODO: Add optimizer/loss options if necessary. If not remove the parameter
    training_params.add_argument("--optimizer", type=str, default="Adam", dest="opt", help="Select optimizer to be used for training. Accepted values: Adam...")
    training_params.add_argument("--loss", type=str, default="MSE", dest="loss", help="Select loss function for model training. Accepted values: MSE, ...")
    training_params.add_argument("--model-output", type=str, required=True, dest="model_output_dir", help="Provide a directory to story your trained model.")
    training_params.add_argument("--verbose", "-v", dest='v', action='store_const', const=True, default=False, help="If given activates verbose mode.")
    training_params.add_argument("--random-seed", dest="random_seed", default=32, type=int, help="If provided np.random.seed is fixed to the given value. Default value is 32.")

    # TODO: enable give a path for train, test, val datasets
    dataset_params.add_argument("--MNIST", dest="MNIST", action='store_const', const=True, default=False, help="Selects MNIST dataset for training, validation, and test")
    dataset_params.add_argument("--train-dataset", dest="train_dataset", default=None, type=str, help="Give the path for a folder to use as training dataset")
    dataset_params.add_argument("--test-dataset", dest="test_dataset", default=None, type=str, help="Give the path for a folder to use as test dataset")
    dataset_params.add_argument("--val-dataset", dest="val_dataset", default=None, type=str, help="Give the path for a folder to use as validation dataset")
    dataset_params.add_argument("--img-size", dest="img_size", required=True, type=int, help="Provide the size of the images used for training and therefore generation size. Please use square images")
    dataset_params.add_argument("--img-channels", dest="img_channels", type=int, default=1, help="If custom dataset, provide if your images are RGB or grayscale by indicating the number of channels.")

    # TODO: Add options for evaluation. E.g. FID
    model_eval.add_argument("--eval", dest="eval_method", default=None, type=str, help="Provide the evaluation method to be used once the model was trained. Available options: FID")
    model_eval.add_argument("--eval-steps", dest='eval_steps', default=None, type=int, help="Evaluate the model during training every n steps. If not given model is just evaluated until training is completed.")
    model_eval.add_argument("--eval-samples", dest='eval_samples', default=None, type=int, help="Proved the number of samples N to be generated to evaluate model distribution against N samples of data distribution")



    argsvalue = parser.parse_args(args)
    return argsvalue


def split_indices(n, val_pct):
    n_val = int(val_pct*n)
    idxs = np.random.permutation(n)
    return idxs[n_val:], idxs[:n_val]


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([torch.cat([i for i in images.cpu()], dim=-1)], dim=-2).permute(1,2,0).cpu(), cmap='gray')
    plt.show()


def train(epochs, train_loader, val_loader, test_dataset, model, optimizer, diffusion, device, loss_fn, verbose, eval, eval_steps, eval_samples, batch_size):
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
                #print(type(x))
                #print(x.shape)
                x = x.to(device)
                t = diffusion.sample_timesteps(x.shape[0]).to(device)
                x_t, noise = diffusion.noise_images(x, t)
                predicted_noise = model(x_t, t)
                batch_val_loss = loss_fn(noise, predicted_noise)
                val_loss += batch_val_loss.item()
        
        # Sample only every 3 epochs for opt purposes
        if verbose:
            if epoch%5 == 0:
                sampled_images = diffusion.sample(model, n=x.shape[0])
                plot_images(sampled_images)

        if eval:
            it = math.ceil(eval_samples/batch_size)
            # Remove all existing elements from saved folders
            for folder in [sample_dir, test_dir]:
                for f in os.listdir(folder):
                    os.remove(os.path.join(folder, f))
            if epoch%eval_steps == 0:
                for j in range(it):
                    sampled_images = diffusion.sample(model, n=x.shape[0])
                    for i, img in enumerate(sampled_images):
                        if j*batch_size+i >= eval_samples:
                            break
                        save_image(img, os.path.join(sample_dir, f"{j*batch_size+i}.png"))
                test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=eval_samples, shuffle=True)
                test_batch = next(iter(test_dl))[0][:eval_samples]
                # Save sampled images and original images
                for i, img in enumerate(test_batch):
                    save_image(img, os.path.join(test_dir, f"{i}.png"))
                # Calculate FID
                fid_proc = subprocess.run(
                    ["python", "-m", "pytorch_fid", sample_dir, test_dir],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                fid_line = fid_proc.stdout.splitlines()[-1]
                fid_value = float(fid_line.strip().split()[-1])
                fids.append(fid_value)
                if verbose:
                    print(f"FID: {fid_value}")

            
        # Save loss
        avg_train_loss = train_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)
        if verbose:
            print(f"Epoch [{epoch+1}/100] | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

    return fids, train_losses, val_losses

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    param_grid = {
        'noise_steps' : args.noise_steps,
        'beta_start' : args.beta_start,
        'beta_end' : args.beta_end,
        'batch_size' : args.batch_size,
        'learning_rate' : args.lr
    }

    keys, values = zip(*param_grid.items())
    runs = list(product(*values))
    for run_id, combo in enumerate(runs):
        config = dict(zip(keys, combo))
        print(f"Running config {run_id+1}/{len(runs)}: {config}")
        
        noise_steps = config['noise_steps']
        beta_start = config['beta_start']
        beta_end = config['beta_end']
        batch_size = config['batch_size']
        learning_rate = config['learning_rate']

        # Retrieve dataset
        if args.MNIST:
            transforms = torchvision.transforms.Compose([
                #torchvision.transforms.Resize(80),
                #torchvision.transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (0.5,)) 
            ])
            fid_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ])
            dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transforms, download=True)
            test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=fid_transform, download=True)

            train_indices, val_indices = split_indices(len(dataset), 0.2)
            train_sampler = SubsetRandomSampler(train_indices)
            train_loader = torch.utils.data.DataLoader(dataset, batch_size, sampler=train_sampler)
            val_sampler = SubsetRandomSampler(val_indices)
            val_loader = DataLoader(dataset, batch_size, sampler=val_sampler)

            print("MNIST dataset loaded")

        elif args.train_dataset != None and args.test_dataset != None and args.val_dataset != None:
            from customDataset import customDataset
            if args.img_channels==1:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((args.img_size, args.img_size)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5,), (0.5,))  # For grayscale; change to (0.5, 0.5, 0.5) if RGB
                ])
            else:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((args.img_size, args.img_size)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5,), (0.5,), (0.5))  # For grayscale; change to (0.5, 0.5, 0.5) if RGB
                ])

            fid_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((args.img_size, args.img_size)),
                torchvision.transforms.ToTensor()
            ])

            train_dataset = customDataset(root_dir=args.train_dataset, transform=transforms, channels=args.img_channels)
            val_dataset = customDataset(root_dir=args.val_dataset, transform=transforms, channels=args.img_channels)
            test_dataset = customDataset(root_dir=args.test_dataset, transform=fid_transform, channels=args.img_channels)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            print("Custom datasets loaded from provided paths.")
        
        else:
            print("Error retreiving train, test, and validation datasets. Please double check you provided a right path")
            raise

        # Init model
        if args.model_sel == "DDPM":
            model = UNet(device=device, c_in=args.img_channels, c_out=args.img_channels).to(device)
        elif args.model_sel == "LDM":
            pass
        else:
            print("Given model is not available")
            raise

        # Init opt
        if args.opt == "Adam":
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        else:
            print("Given optimizer is not available")
            raise

        # Init loss fn
        if args.loss == "MSE":
            loss_fn = nn.MSELoss()
        else:
            print("Given loss function is not available")
            raise

        # Init diffusion or any other model
        if args.model_sel == "DDPM":
            diffusion = Diffusion(noise_steps=noise_steps, beta_start=beta_start, beta_end=beta_end, img_size=args.img_size, device=device)

        
        fids, train_losses, val_losses = train(args.epochs, train_loader, val_loader, test_dataset, model, optimizer, diffusion, device, loss_fn, args.v, args.eval_method, args.eval_steps, args.eval_samples, batch_size)
        
        # Save outputs
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H%M', t)
        os.makedirs(args.model_output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.model_output_dir, f'output_model_{timestamp}.pt'))
        out_dir = "data/temp"
        os.makedirs(out_dir, exist_ok=True)
        with open(out_dir+f'/train_losses_{timestamp}.pkl', 'wb') as f:
            pickle.dump(train_losses, f)
        with open(out_dir+f'/val_losses_{timestamp}.pkl', 'wb') as f:
            pickle.dump(val_losses, f)
        if args.eval_method:
            with open(out_dir+f'/fids_{timestamp}.pkl', 'wb') as f:
                pickle.dump(fids, f)
    

if __name__ == "__main__":
    argsvalue = argparser(sys.argv[1:])
    main(argsvalue)
