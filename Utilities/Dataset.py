# --------------------------------------------------------------------------------------------------
# ------------------------------------ Diffusion :: Data Loader ------------------------------------
# --------------------------------------------------------------------------------------------------
import torchvision.transforms as transforms
import torch
import pathlib

from PIL import Image

from Diffusers.DDIM import DDIM
from Diffusers.DDPM import DDPM

bilinear = transforms.InterpolationMode.BILINEAR


# --------------------------------------------------------------------------------------------------
# -------------------------------------- CLASS :: Data Loader --------------------------------------
# --------------------------------------------------------------------------------------------------
class Dataset(torch.utils.data.Dataset):

    def __init__(self, diffuser: DDIM | DDPM) -> None:
        super().__init__()

        self.transforms = transforms.Compose([
            transforms.RandomAffine(
                degrees=(-15, 15),  # Rotation between -15 to +15 degrees
                translate=(0.10, 0.10),  # Translation up to 10% horizontally and vertically
                scale=(0.90, 1.10),  # Scaling between 90% to 110%
                fill=0xFFFFFF,
                interpolation=bilinear  # Use bilinear interpolation
            ),
            transforms.ColorJitter(
                brightness=0.10,  # ±5% brightness
                contrast=0.10,  # ±5% contrast
                saturation=0.10,  # ±5% saturation
                hue=0.10  # ±5% hue
            ),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(256, interpolation=bilinear),
            transforms.RandomCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.diffuser = diffuser

    def __getitem__(self, path: pathlib.Path) -> tuple[torch.Tensor, ...]:
        return self.diffuser.forward(self.transforms(Image.open(path)))

