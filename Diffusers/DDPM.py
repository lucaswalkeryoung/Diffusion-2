# --------------------------------------------------------------------------------------------------
# ----------------------------------- DIFFUSERS :: DDPM Diffuser -----------------------------------
# --------------------------------------------------------------------------------------------------
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts as CAWRScheduler

import torchvision.transforms as transforms
import torch.optim.lr_scheduler as scheduling
import torch.optim as optimizing
import torch.nn as networks

import pathlib
import random
import torch
import os
import math
import shutil
import typing

from .  Diffuser import Diffuser
from Modules.UNet import UNet

from PIL import Image


# --------------------------------------------------------------------------------------------------
# ------------------------------------- CLASS :: DDPM Diffuser -------------------------------------
# --------------------------------------------------------------------------------------------------
class DDPM(Diffuser):

    # ------------------------------------------------------------------------------------------
    # -------------------------------- ATTRIBUTES :: Attributes --------------------------------
    # ------------------------------------------------------------------------------------------
    transforms_forward : transforms.Compose
    transforms_reverse : transforms.Compose

    scheduler : scheduling.LRScheduler
    optimizer : optimizing.Optimizer
    unet      : networks.Module
    device    : torch.device
    paths     : list[str]

    resolution  : int   # maximum image resolution
    timesteps   : int   # maximum denoising steps
    max_beta    : float # maximum noise
    min_beta    : float # minimum noise
    max_lr      : float
    min_lr      : float
    t_channels  : int   # t-embedding dimension
    s_channels  : int   # s-embedding dimension (spatial embedding)
    epochs      : int
    restarts    : int
    batch_size  : int
    dataset_len : int

    t: torch.Tensor # precomputed temporal embeddings
    s: torch.Tensor # precomputed spatial embeddings
    a: torch.Tensor # alpha schedule
    b: torch.Tensor # beta schedule
    ā: torch.Tensor # cumulative alpha schedule
    b̄: torch.Tensor # cumulative beta schedule
    ã̄: torch.Tensor # square-root of ā
    ã: torch.Tensor # square-root of a
    b̃̄: torch.Tensor # square-root of b̄
    b̃: torch.Tensor # square-root of b

    # ------------------------------------------------------------------------------------------
    # ------------------- OPERATORS :: Sampler and Data-Loader Functionality -------------------
    # ------------------------------------------------------------------------------------------
    def __getitem__(self, path: str) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(self.transforms_forward(Image.open(path)).to(self.device))

    def __iter__(self) -> typing.Iterator[str]:
        return iter(random.sample(self.paths, len(self.paths)))

    def __len__(self) -> int:
        return len(self.paths)

    # ------------------------------------------------------------------------------------------
    # ------------------------------- CONSTRUCTOR :: Constructor -------------------------------
    # ------------------------------------------------------------------------------------------
    def __init__(self, **hyperparameters: int | float) -> None:
        super().__init__()

        self.resolution  = hyperparameters.get("resolution",  256) # max resolution
        self.timesteps   = hyperparameters.get("timesteps",  1000) # max timesteps
        self.max_lr      = hyperparameters.get("max_lr",     1e-2)
        self.min_lr      = hyperparameters.get("min_lr",     1e-5)
        self.max_beta    = hyperparameters.get("max_beta",   2e-2) # vestigial
        self.min_beta    = hyperparameters.get("min_beta",   1e-4) # vestigial
        self.t_channels  = hyperparameters.get("t_channels",   32) # timestep embed dimension
        self.s_channels  = hyperparameters.get("s_channels",   32) # position embed dimension
        self.epochs      = hyperparameters.get("epochs",      128)
        self.restarts    = hyperparameters.get("restarts",      8) # number of lr-restarts
        self.batch_size  = hyperparameters.get("batch_size",    8)
        self.dataset_len = hyperparameters.get("dataset_len",   0) # mandatory

        self.optimizer = None
        self.scheduler = None
        self.unet = None
        self.device = None

        self.t = None
        self.s = None
        self.a = None
        self.b = None
        self.ā = None
        self.b̄ = None
        self.ã̄ = None
        self.ã = None
        self.b̃̄ = None
        self.b̃ = None

        self.configure_transforms()
        self.configure_components()
        self.configure_scheduling()
        self.configure_embeddings()


    # ------------------------------------------------------------------------------------------
    # -------------------- CONFIGURATION :: Configure Child Components, Etc --------------------
    # ------------------------------------------------------------------------------------------
    def configure_components(self) -> None:

        # cuda if possible else mps (apple silicon)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")

        # paths of all images in the dataset
        self.paths = list(pathlib.Path('Dataset').glob("*.JPG"))

        # the unet
        self.unet = UNet(self.t_channels, self.s_channels)
        self.unet = self.unet.to(self.device)

        # the unet's optimizer
        self.optimizer = optimizing.Adam(self.unet.parameters(), lr=self.max_lr)

        # number of batches until lr-restart
        batches_per_epoch = math.ceil(len(self.paths) / self.batch_size)
        epochs_per_restart = self.epochs / self.restarts
        T_0 = epochs_per_restart * batches_per_epoch

        # cosine-annealing-warm-restarts scheduler
        self.scheduler = CAWRScheduler(self.optimizer, T_0=50, eta_min=0)


    # ------------------------------------------------------------------------------------------
    # ------------- CONFIGURATION :: Configure Forward and Reverse Transformations -------------
    # ------------------------------------------------------------------------------------------
    def configure_transforms(self) -> None:

        BILINEAR = transforms.InterpolationMode.BILINEAR

        # forward transforms - spatial and color transforms, to tensor, and normalize
        self.transforms_forward = transforms.Compose([
            # transforms.RandomAffine(
            #     degrees=(-45, 45),  # Rotation between -15 to +15 degrees
            #     translate=(0.50, 0.50),  # Translation up to 10% horizontally and vertically
            #     scale=(0.50, 1.50),  # Scaling between 90% to 110%
            #     fill=0xFFFFFF,
            #     interpolation=BILINEAR  # Use bilinear interpolation
            # ),
            # transforms.ColorJitter(
            #     brightness=0.25, # ±10% brightness
            #     contrast=0.25, # ±10% contrast
            #     saturation=0.25, # ±10% saturation
            #     hue=0.25 # ±10% hue
            # ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize(256, interpolation=BILINEAR),
            transforms.RandomCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        # reverse transforms - from tensor and denormalize
        self.transforms_reverse = transforms.Compose([
            transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0)),
            transforms.ToPILImage(),
        ])

    # ------------------------------------------------------------------------------------------
    # ----------------------- CONFIGURATION :: Configure Noise Injection -----------------------
    # ------------------------------------------------------------------------------------------
    def configure_scheduling(self, ε: float = 0, π: float = torch.pi) -> None:

        # cumulative cosine signal-to-noise
        ā = torch.linspace(ε, 1 + ε, self.timesteps + 1) / (1 + ε)
        ā = torch.cos(ā * π * 0.50) ** 2
        ā = ā / ā[0]

        # stepwise signal-to-noise
        a = torch.cat([torch.Tensor([1.00]), ā[1:] / ā[:-1]], dim=0)

        # cumulative and stepwise noise-to-signal
        b = 1.00 - a
        b̄ = 1.00 - ā

        # square-roots of relevant tensors for cleanliness
        ã̄ = torch.sqrt(ā)
        ã = torch.sqrt(a)
        b̃̄ = torch.sqrt(b̄)
        b̃ = torch.sqrt(b)

        self.a = a.to(self.device)
        self.b = b.to(self.device)
        self.ā = ā.to(self.device)
        self.b̄ = b̄.to(self.device)
        self.ã = ã.to(self.device)
        self.ã̄ = ã̄.to(self.device)
        self.b̃ = b̃.to(self.device)
        self.b̃̄ = b̃̄.to(self.device)

    # ------------------------------------------------------------------------------------------
    # --------------- CONFIGURATION :: Configure Temporal and Spatial Embeddings ---------------
    # ------------------------------------------------------------------------------------------
    def configure_embeddings(self, π: float = torch.pi) -> None:

        # sinusoidal frequencies for timestep embeddings (radians/unit-time)
        ω = torch.linspace(math.log(1.0), math.log(10000.0), self.t_channels // 2)
        ω = torch.exp(ω)

        # mapping from timestep to percentage of maximum time
        u = torch.linspace(0.00, 1.00, self.timesteps + 1).unsqueeze(1)

        # precomputed sinusoidal timestep embeddings for all steps
        sin = torch.sin(u * ω) # (θ * ω) is in radians
        cos = torch.cos(u * ω) # (θ * ω) is in radians
        t = torch.cat([sin, cos], dim=1)

        # sinusoidal frequencies for spatial embeddings (radians/unit-displacement)
        ω = torch.linspace(math.log(1.0), math.log(10000.0), self.s_channels // 2)
        ω = torch.exp(ω)

        # mapping from timestep to percentage of maximum time
        u = torch.linspace(0.00, 1.00, self.resolution).unsqueeze(1)

        # Precomputed sinusoidal spatial embedding
        sin = torch.sin(u * ω)  # (u * ω) is in radians
        cos = torch.cos(u * ω)  # (u * ω) is in radians
        s = torch.cat([sin, cos], dim=1)
        s = s.unsqueeze(1) + s.unsqueeze(0)
        s = s.swapaxes(0, 2)

        # spatial and temporal embeddings, moved to the device
        self.t = t.to(self.device)
        self.s = s.to(self.device)

    # ------------------------------------------------------------------------------------------
    # ------------------------ FORWARD :: The Forward Diffusion Process ------------------------
    # ------------------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: int = 0) -> tuple[torch.Tensor, ...]:

        # start training on just the first few timestep
        t = t or random.randint(1, 100)

        ã̄ₜ = self.ã̄[t].float() # signal-to-noise
        b̃̄ₜ = self.b̃̄[t].float() # noise-to-signal

        # the true noise
        eₜ = torch.randn_like(x).to(self.device).float()

        # math
        xₜ = (x * ã̄ₜ) + (eₜ * b̃̄ₜ)

        # the temporal and spatial embeddings
        t = self.t[t]
        s = self.s

        return xₜ, eₜ, t, s

    # ------------------------------------------------------------------------------------------
    # ------------------------ REVERSE :: The Reverse Diffusion Process ------------------------
    # ------------------------------------------------------------------------------------------
    def reverse(self, n: int = 5, t: int = 0) -> torch.Tensor:

        xₜ = self.transforms_forward(Image.open(random.choice(self.paths))).to(self.device)
        self.transforms_reverse(xₜ).save(f'Outputs/before.png')
        xₜ, _, _, _ = self.forward(xₜ, t=100)
        self.transforms_reverse(xₜ).save(f'Outputs/after.png')

        xₜ = xₜ.unsqueeze(0)

        self.unet.eval()

        with torch.no_grad():

            # skip t = timesteps due to a quirk in the unet simulation logic
            for t in reversed(range(1, 101)):

                time  = self.t[t].unsqueeze(0)
                space = self.s.unsqueeze(0)

                eₜ = self.unet(xₜ, time, space)

                ãₜ = self.ã[t].float()
                b̃ₜ = self.b̃[t].float()

                xₜ = (xₜ - (eₜ * b̃ₜ)) / ãₜ

                self.transforms_reverse(xₜ[0]).save(f'Outputs/reverse_{t}.png')

        self.unet.train()

        return xₜ


    # ------------------------------------------------------------------------------------------
    # ------------------------------- METHOD :: Learn from Error -------------------------------
    # ------------------------------------------------------------------------------------------
    def learn(self, error: torch.Tensor) -> None:

        error.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()


    # ------------------------------------------------------------------------------------------
    # --------------------------------- METHOD :: Sanity Check ---------------------------------
    # ------------------------------------------------------------------------------------------
    def test(self, t: int = 0) -> torch.Tensor:

        xₜ = self.transforms_forward(Image.open('Dataset/Pkmn_img1.JPG')).to(self.device)
        xₜ = xₜ.float()

        epsilons = []
        forwards = []
        reverses = []

        for t in range(1, self.timesteps):

            forwards.append(xₜ)

            eₜ = torch.randn_like(xₜ).float()

            ãₜ = self.ã[t].float()
            b̃ₜ = self.b̃[t].float()

            xₜ = (ãₜ * xₜ) + (b̃ₜ * eₜ)

            epsilons.append(eₜ)

            self.transforms_reverse(xₜ).save(f'Outputs/forward_{t}.png')

        for t in reversed(range(1, self.timesteps)):

            eₜ = epsilons.pop()

            ãₜ = self.ã[t].float()
            b̃ₜ = self.b̃[t].float()

            xₜ = (xₜ - (eₜ * b̃ₜ)) / ãₜ

            reverses.append(xₜ)

            self.transforms_reverse(xₜ).save(f'Outputs/reverse_{t}.png')