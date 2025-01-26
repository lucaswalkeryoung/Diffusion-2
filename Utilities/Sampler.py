# --------------------------------------------------------------------------------------------------
# ----------------------------------- Diffusion :: Batch Sampler -----------------------------------
# --------------------------------------------------------------------------------------------------
from Diffusers.DDIM import DDIM
from Diffusers.DDPM import DDPM

import pathlib
import torch
import random
import typing


# --------------------------------------------------------------------------------------------------
# ------------------------------------- CLASS :: Batch Sampler -------------------------------------
# --------------------------------------------------------------------------------------------------
class Sampler(torch.utils.data.Sampler):

    def __init__(self, diffuser: DDIM | DDPM) -> None:
        super().__init__()

        self.images = list(pathlib.Path('Dataset').rglob('*.JPG'))

    def __len__(self) -> int:
        return len(self.images)

    def __iter__(self) -> typing.Iterator[torch.Tensor]:
        return iter(random.sample(self.images, len(self.images)))