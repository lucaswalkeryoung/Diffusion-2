# -------------------------------------------------------------------------------------------------
# ---------------------------------- Diffusion :: Encoder Blocks ----------------------------------
# -------------------------------------------------------------------------------------------------
import torch.nn.utils.parametrizations as parametrizations
import torch.nn.functional as functional
import torch.nn as networks

import torch
import math

Tensor = torch.Tensor


# -------------------------------------------------------------------------------------------------
# ------------------------------------ BLOCK :: Upsample Block ------------------------------------
# -------------------------------------------------------------------------------------------------
class DecoderScale(networks.Module):

    def __init__(self, i_channels: int, o_channels: int = 32) -> None:
        super().__init__()

        self.conv = networks.Conv2d(i_channels, o_channels, kernel_size=3, padding=1)
        self.conv = parametrizations.weight_norm(self.conv)
        self.norm = networks.InstanceNorm2d(o_channels)
        self.silu = networks.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = functional.interpolate(x, scale_factor=2.00, mode='bilinear')

        x = self.conv(x)
        x = self.norm(x)
        x = self.silu(x)

        return x


# -------------------------------------------------------------------------------------------------
# -------------------------------- BLOCK :: Residual Decoder Block --------------------------------
# -------------------------------------------------------------------------------------------------
class DecoderInner(networks.Module):

    def __init__(self, x_channels: int, t_channels: int = 32) -> None:
        super().__init__()

        a_channels = x_channels + t_channels
        b_channels = x_channels + x_channels

        self.conv1 = networks.Conv2d(a_channels, x_channels, kernel_size=3, padding=1)
        self.conv1 = parametrizations.weight_norm(self.conv1)
        self.norm1 = networks.InstanceNorm2d(x_channels)
        self.conv2 = networks.Conv2d(b_channels, x_channels, kernel_size=3, padding=1)
        self.conv2 = parametrizations.weight_norm(self.conv2)
        self.norm2 = networks.InstanceNorm2d(x_channels)
        self.swish = networks.SiLU()
        self.noise = networks.Dropout(0.10)

    def forward(self, t: torch.Tensor, r: torch.Tensor, s: torch.Tensor) -> torch.Tensor:

        t = t.unsqueeze(2).unsqueeze(3).expand(-1, -1, r.size(2), r.size(3))

        x = self.conv1(torch.cat([r, t], dim=1))
        x = self.norm1(x)
        x = self.swish(x)
        x = self.noise(x)
        x = self.conv2(torch.cat([x, s], dim=1))
        x = self.norm2(x)
        x = self.swish(x)

        return x + r


# -------------------------------------------------------------------------------------------------
# ---------------------------------- BLOCK :: Decoder Exit Block ----------------------------------
# -------------------------------------------------------------------------------------------------
class DecoderFinal(networks.Module):

    def __init__(self, i_channels: int, o_channels: int) -> None:
        super().__init__()

        self.conv = networks.Conv2d(i_channels, o_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# -------------------------------------------------------------------------------------------------
# -------------------------------- BLOCK :: Standard Decoder Block --------------------------------
# -------------------------------------------------------------------------------------------------
class DecoderBlock(networks.Module):

    def __init__(self, i_channels: int, o_channels: int, t_channels: int = 32) -> None:
        super().__init__()

        self.sample = DecoderScale(i_channels, o_channels)
        self.block1 = DecoderInner(o_channels, t_channels)
        self.block2 = DecoderInner(o_channels, t_channels)

    def forward(self, t: Tensor, x: Tensor, y: Tensor, z: Tensor) -> Tensor:

        x = self.sample(x)
        x = self.block1(t, x, y)
        x = self.block2(t, x, z)

        return x