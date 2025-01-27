# --------------------------------------------------------------------------------------------------
# ----------------------------------- DIFFUSERS :: DDPM Diffuser -----------------------------------
# --------------------------------------------------------------------------------------------------
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as scheduling
import torch.optim as optimizing
import torch.nn as networks
import torch
import os
import math
import shutil

from . Diffuser import Diffuser

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

    resolution : int   # maximum image resolution
    timesteps  : int   # maximum denoising steps
    max_beta   : float # maximum noise
    min_beta   : float # minimum noise
    max_lr     : float
    min_lr     : float
    t_channels : int   # t-embedding dimension
    s_channels : int   # s-embedding dimension (spatial embedding)
    scales     : int   # number of distinct resolutions
    epochs     : int
    restarts   : int
    batch_size : int

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
    # ------------------------------- CONSTRUCTOR :: Constructor -------------------------------
    # ------------------------------------------------------------------------------------------
    def __init__(self, **hyperparameters: int | float) -> None:
        super().__init__()

        self.resolution = hyperparameters.get("resolution",  256)
        self.timesteps  = hyperparameters.get("timesteps",   100)
        self.max_lr     = hyperparameters.get("max_lr",     1e-2)
        self.min_lr     = hyperparameters.get("min_lr",     1e-5)
        self.max_beta   = hyperparameters.get("max_beta",   2e-2)
        self.min_beta   = hyperparameters.get("min_beta",   1e-4)
        self.t_channels = hyperparameters.get("t_channels",   32)
        self.s_channels = hyperparameters.get("s_channels",   32)
        self.scales     = hyperparameters.get("scales",        4)
        self.epochs     = hyperparameters.get("epochs",      128)
        self.restarts   = hyperparameters.get("restarts",      8)
        self.batch_size = hyperparameters.get("batch_size",    8)

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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")


    # ------------------------------------------------------------------------------------------
    # ------------- CONFIGURATION :: Configure Forward and Reverse Transformations -------------
    # ------------------------------------------------------------------------------------------
    def configure_transforms(self) -> None:

        BILINEAR = transforms.InterpolationMode.BILINEAR

        # forward transforms - spatial and color transforms, to tensor, and normalize
        self.transforms_forward = transforms.Compose([
            transforms.RandomAffine(
                degrees=(-15, 15),  # Rotation between -15 to +15 degrees
                translate=(0.10, 0.10),  # Translation up to 10% horizontally and vertically
                scale=(0.90, 1.10),  # Scaling between 90% to 110%
                fill=0xFFFFFF,
                interpolation=BILINEAR  # Use bilinear interpolation
            ),
            transforms.ColorJitter(
                brightness=0.10, # ±10% brightness
                contrast=0.10, # ±10% contrast
                saturation=0.10, # ±10% saturation
                hue=0.10 # ±10% hue
            ),
            transforms.RandomHorizontalFlip(),
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
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:

        ã̄ₜ = self.ã̄[self.timesteps] # signal-to-noise
        b̃̄ₜ = self.ã̄[self.timesteps] # noise-to-signal

        # the first noise prediction as well as the noise injection
        eₜ = torch.randn_like(x)

        # the noisy image
        xₜ = (x * ã̄ₜ) + (eₜ * b̃̄ₜ)

        return x, xₜ, eₜ

    # ------------------------------------------------------------------------------------------
    # ------------------------ REVERSE :: The Reverse Diffusion Process ------------------------
    # ------------------------------------------------------------------------------------------
    def reverse(self, x0: torch.Tensor, xₜ: torch.Tensor, eₜ: torch.Tensor) -> torch.Tensor:

        # skip t = timesteps due to a quirk in the unet simulation logic
        for t in reversed(range(1, self.timesteps)):

            ã̄ = self.ã̄[t]
            ã = self.ã[t]
            b = self.b[t]
            b̃̄ = self.b̃̄[t]

            xₜ = (1 / ã) * (xₜ - (eₜ * (b / b̃̄))) # the denoised image

            # simulate noise prediction by deriving it directly from the clean image
            eₜ = (xₜ - (x0 * ã̄)) / b̃̄

            self.transforms_reverse(xₜ).save(f"Outputs/{t}_reversed.png")

        return xₜ