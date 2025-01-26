# --------------------------------------------------------------------------------------------------
# ----------------------------------- Diffusion :: Training Loop -----------------------------------
# --------------------------------------------------------------------------------------------------
from Utilities.Dataset import Dataset
from Utilities.Sampler import Sampler

from Diffusers.DDIM import DDIM
from Diffusers.DDPM import DDPM

import shutil
import os

import torch
import torch.nn.functional as functional
import torch.utils.data as datatools

shutil.rmtree("Outputs")
os.mkdir("Outputs")

# --------------------------------------------------------------------------------------------------
# --------------------------------- Environment and Infrastructure ---------------------------------
# --------------------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

diffuser = DDPM(
    timesteps  = 250,  # total denoising steps
    resolution = 256,  # maximum image resolution
    max_beta   = 2e-2, # minimum noise
    min_beta   = 1e-4, # maximum noise
    max_lr     = 1e-3,
    min_lr     = 1e-4,
    t_channels = 32,   # timestep embedding dimension
    s_channels = 32,   # spatial embedding dimension (for attention)
    cycles     = 920,  # restart every epoch
)

batch_size = 8
epochs = 128

sampler = Sampler(diffuser)
dataset = Dataset(diffuser)
batches = datatools.DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size)

# --------------------------------------------------------------------------------------------------
# ----------------------------------------- Training Loop ------------------------------------------
# --------------------------------------------------------------------------------------------------
for eid in range(epochs):
    for bid, (noisy, noise, t, s) in enumerate(batches):

        noisy = noisy.to(diffuser.device)
        noise = noise.to(diffuser.device)
        t = t.to(diffuser.device)
        s = s.to(diffuser.device)

        guess = diffuser.unet(noisy, t, s)
        error = functional.mse_loss(guess, noise) * 1000

        if not bid % 100:
            diffuser.reverse(n=1)

        print(f"[{eid:04}:{bid:04}] Loss: {error.item():.5f}")

        diffuser.learn(error)