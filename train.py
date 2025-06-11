# --------------------------------------------------------------------------------------------------
# ----------------------------------- Diffusion :: Training Loop -----------------------------------
# --------------------------------------------------------------------------------------------------
from torch.utils.data import DataLoader
from Diffusers.DDPM   import DDPM

import pathlib
import torch
import os
import shutil


# --------------------------------------------------------------------------------------------------
# --------------------------------- Environment and Infrastructure ---------------------------------
# --------------------------------------------------------------------------------------------------
ddpm = DDPM(
    resolution  = 256,  # maximum resolution
    timesteps   = 100,
    max_lr      = 1e-2,
    min_lr      = 1e-5,
    max_beta    = 0.0,  # vestigial
    min_beta    = 0.0,  # vestigial
    t_channels  = 32,   # timestep embedding channels
    s_channels  = 32,   # spatial embedding channels (for attention)
    epochs      = 2048,
    restarts    = 8,    # number of learning-rate restarts
    batch_size  = 32,
    dataset_len = len(list(pathlib.Path('Dataset').rglob('*.JPG'))),
)

batches = DataLoader(dataset=ddpm, sampler=ddpm, batch_size=ddpm.batch_size)

shutil.rmtree("Outputs", ignore_errors=True)
os.mkdir("Outputs")


# --------------------------------------------------------------------------------------------------
# -------------------------------------------- The Loop --------------------------------------------
# --------------------------------------------------------------------------------------------------
for eid in range(ddpm.epochs):
    for bid, (noisy, noise, t, s) in enumerate(batches):

        guess = ddpm.unet(noisy, t, s)
        error = torch.nn.functional.mse_loss(guess, noise) * 10000

        print(f"[{eid:04}:{bid:04}] Loss: {error.item()}")

        if not bid % 100:
            output = ddpm.reverse(n=1)[0]

        ddpm.learn(error)
