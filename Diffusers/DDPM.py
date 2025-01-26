# --------------------------------------------------------------------------------------------------
# ----------------------------------- Diffusion :: DDPM Diffuser -----------------------------------
# --------------------------------------------------------------------------------------------------
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts as CAWRScheduler

import torchvision.transforms as transforms
import torch.optim as optimizers
import torch.nn as networks
import torch

import uuid
import math
import random

from Modules.UNet import UNet


# --------------------------------------------------------------------------------------------------
# ------------------------------------- CLASS :: DDPM Diffuser -------------------------------------
# --------------------------------------------------------------------------------------------------
class DDPM(object):

    def __init__(self, **parameters: int | float) -> None:

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")

        # tensor-to-image procedure
        self.transform = transforms.Compose([
            transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0)),
            transforms.ToPILImage(),
        ])

        self.timesteps  : int   = parameters.get('timesteps', 1000) # total denoising steps
        self.resolution : int   = parameters.get('resolution', 256) # maximum image resolution
        self.max_beta   : float = parameters.get('max_beta',  2e-2) # minimum noise
        self.min_beta   : float = parameters.get('min_beta',  1e-4) # maximum noise
        self.max_lr     : float = parameters.get('max_lr',    5e-2)
        self.min_lr     : float = parameters.get('min_lr',    5e-3)
        self.t_channels : int   = parameters.get('t_channels',  32) # timestep embedding dimension
        self.s_channels : int   = parameters.get('s_channels',  32) # position embedding dimension
        self.cycles     : int   = parameters.get('cycles',      32) # epochs before restart

        # initialize the unet
        self.unet = UNet(self.t_channels, self.s_channels)
        self.unet = self.unet.to(self.device)
        self.unet = self.unet.train()

        # initialize the optimizer and scheduler
        self.optimizer = optimizers.Adam(self.unet.parameters(), lr=self.min_lr)
        self.scheduler = CAWRScheduler(self.optimizer, T_0=self.cycles, eta_min=self.min_lr)

        # declare remaining tensor attributes
        self.a : torch.Tensor
        self.b : torch.Tensor
        self.ā : torch.Tensor
        self.b̄ : torch.Tensor
        self.t : torch.Tensor
        self.s : torch.Tensor

        self.compute_embeddings()
        self.compute_scheduling()

    def compute_embeddings(self) -> None:

        # sinusoidal embedding along the height and width axes (used by
        # the self-attention mechanisms)
        ω = torch.linspace(math.log(1.0), math.log(10000.0), self.t_channels // 2).exp()
        d = torch.linspace(0.00, math.pi * 2, self.resolution).unsqueeze(1)
        sin = torch.sin(d * ω)
        cos = torch.cos(d * ω)
        s = torch.cat([sin, cos], dim=1)
        s = s.unsqueeze(1) + s.unsqueeze(0)
        s = s.swapaxes(0, 2)

        # precomputed sinusoidal embedding along the time axis
        ω = torch.linspace(math.log(1.0), math.log(10000.0), self.t_channels // 2).exp()
        t = torch.linspace(0.00, math.pi * 2, self.timesteps + 1).unsqueeze(1)
        sin = torch.sin(t * ω)
        cos = torch.cos(t * ω)
        t = torch.cat([sin, cos], dim=1)

        # store relevant tensors on CPU
        self.t = t
        self.s = s

    def compute_scheduling(self) -> None:

        # a cosine-squared beta schedule from min_beta to max_beta
        b = torch.linspace(0, torch.pi / 2.00, self.timesteps + 1)
        b = torch.cos(b) ** 2
        b = b * (self.max_beta - self.min_beta) + self.min_beta

        # alpha schedule
        a = 1.00 - b

        # cumulative alpha and beta schedules
        ā = torch.cumprod(a, dim=0)
        b̄ = 1.00 - ā

        # store relevant tensors on CPU
        self.a = a
        self.b = b
        self.ā = ā
        self.b̄ = b̄

    def forward(self, signal: torch.Tensor, t: int = 0) -> tuple[torch.Tensor, ...]:

        t = t or random.randint(1, self.timesteps)

        t_embedding = self.t[t]
        s_embedding = self.s

        āₜ = self.ā[t]
        b̄ₜ = self.b̄[t]

        signal = torch.sqrt(āₜ) * signal
        noise  = torch.sqrt(b̄ₜ) * torch.randn_like(signal)

        return signal + noise, noise, t_embedding, s_embedding

    def reverse(self, n: int = 1) -> None:

        signal = torch.randn(n, 3, 256, 256).to(self.device)

        self.unet.eval()
        with torch.no_grad():
            for t in reversed(range(1, self.timesteps + 1)):

                t_embedding = self.t[t]
                s_embedding = self.s

                t_embedding = t_embedding.to(self.device)
                s_embedding = s_embedding.to(self.device)

                t_embedding = t_embedding.unsqueeze(0)
                s_embedding = s_embedding.unsqueeze(0)

                incoming = torch.randn_like(signal) if t != 1 else torch.zeros_like(signal)
                outgoing = self.unet(signal, t_embedding, s_embedding)

                bₜ = self.b[t]
                b̄ₜ = self.b̄[t]
                aₜ = self.a[t]

                incoming = incoming * (torch.sqrt(bₜ))
                outgoing = outgoing * (bₜ / torch.sqrt(b̄ₜ))

                signal = signal - outgoing
                signal = signal / torch.sqrt(aₜ)
                signal = signal + incoming

        self.unet.train()

        self.transform(signal[0]).save(f"Outputs/{uuid.uuid4()}.jpg")

    def learn(self, error: torch.Tensor) -> None:

        error.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()