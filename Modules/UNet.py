# -------------------------------------------------------------------------------------------------
# ----------------------- Diffusion :: The U-Net and Assorted Common Blocks -----------------------
# -------------------------------------------------------------------------------------------------
import torch.nn.utils.parametrizations as parametrizations
import torch.nn.functional as functional
import torch.nn.init as initialization
import torch.nn as networks
import torch

import math

from . Encoder import EncoderEntry
from . Encoder import EncoderBlock
from . Decoder import DecoderBlock
from . Decoder import DecoderFinal


# -------------------------------------------------------------------------------------------------
# --------------------------------- BLOCK :: Self-Attention Block ---------------------------------
# -------------------------------------------------------------------------------------------------
class SelfAttentionBlock(networks.Module):

    def __init__(self, x_channels: int, s_channels: int = 32) -> None:
        super().__init__()

        i_channels = x_channels + s_channels
        heads = 4

        self.attention = networks.MultiheadAttention(i_channels, heads)

        self.line1 = networks.Linear(i_channels, x_channels, bias=False)
        self.line1 = parametrizations.weight_norm(self.line1)
        self.norm1 = networks.LayerNorm(x_channels)
        self.line2 = networks.Linear(x_channels, x_channels, bias=False)
        self.line2 = parametrizations.weight_norm(self.line2)
        self.norm2 = networks.LayerNorm(x_channels)
        self.swish = networks.SiLU()

    def forward(self, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

        # store original dimensions
        batch, channels, height, width = x.shape

        # scale the spatial embedding to match the input
        s = functional.interpolate(s, size=(height, width), mode='bilinear')

        # concatenate with spatial embedding and reshape for mha
        x = torch.cat([s, x], dim=1)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.permute(2, 0, 1)

        # apply attention
        x, _ = self.attention(x, x, x)

        # small feed-forward to refine discovered features
        x = self.line1(x)
        x = self.norm1(x)
        x = self.swish(x)
        x = self.line2(x)
        x = self.norm2(x)
        x = self.swish(x)

        # reshape for convolution and return
        return x.view(batch, channels, height, width)


# -------------------------------------------------------------------------------------------------
# ----------------------------------- BLOCK :: Bottleneck Inner -----------------------------------
# -------------------------------------------------------------------------------------------------
class BottleneckInner(networks.Module):

    def __init__(self, x_channels: int, t_channels: int = 32) -> None:
        super().__init__()

        i_channels = x_channels + t_channels

        self.conv1 = networks.Conv2d(i_channels, x_channels, kernel_size=3, padding=1)
        self.conv1 = parametrizations.weight_norm(self.conv1)
        self.norm1 = networks.InstanceNorm2d(x_channels)
        self.conv2 = networks.Conv2d(x_channels, x_channels, kernel_size=3, padding=1)
        self.conv2 = parametrizations.weight_norm(self.conv2)
        self.norm2 = networks.InstanceNorm2d(x_channels)
        self.swish = networks.SiLU()
        self.noise = networks.Dropout(0.10)

    def forward(self, t: torch.Tensor, r: torch.Tensor) -> torch.Tensor:

        t = t.unsqueeze(2).unsqueeze(3).expand(-1, -1, r.size(2), r.size(3))

        x = self.conv1(torch.cat([r, t], dim=1))
        x = self.norm1(x)
        x = self.swish(x)
        x = self.noise(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.swish(x)

        return x + r


# -------------------------------------------------------------------------------------------------
# ----------------------------------- BLOCK :: Bottleneck Block -----------------------------------
# -------------------------------------------------------------------------------------------------
class BottleneckBlock(networks.Module):

    def __init__(self, x_channels: int, t_channels: int = 32) -> None:
        super().__init__()

        self.block1 = BottleneckInner(x_channels, t_channels)
        self.block2 = BottleneckInner(x_channels, t_channels)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

        x = self.block1(t, x)
        x = self.block2(t, x)

        return x


# -------------------------------------------------------------------------------------------------
# -------------------------------------- MODULE :: The U-Net --------------------------------------
# -------------------------------------------------------------------------------------------------
class UNet(networks.Module):

    def __init__(self, t_channels: int = 32, s_channels: int = 32) -> None:
        super().__init__()

        self.encoder0 = EncoderEntry(3,    32)
        self.encoder1 = EncoderBlock(32,   64, t_channels)
        self.encoder2 = EncoderBlock(64,  128, t_channels)
        self.encoder3 = EncoderBlock(128, 256, t_channels)

        self.bottleneck1 = BottleneckBlock(256, t_channels)
        self.bottleneck2 = BottleneckBlock(256, t_channels)
        self.bottleneck3 = BottleneckBlock(256, t_channels)

        self.decoder3 = DecoderBlock(256, 128, t_channels)
        self.decoder2 = DecoderBlock(128,  64, t_channels)
        self.decoder1 = DecoderBlock(64,   32, t_channels)
        self.decoder0 = DecoderFinal(32,    3)

        self.attention1 = SelfAttentionBlock(64,  s_channels)
        self.attention2 = SelfAttentionBlock(128, s_channels)
        self.attention3 = SelfAttentionBlock(256, s_channels)
        self.attention4 = SelfAttentionBlock(256, s_channels)
        self.attention5 = SelfAttentionBlock(128, s_channels)
        self.attention6 = SelfAttentionBlock(64,  s_channels)

        for layer in self.modules():

            if isinstance(layer, networks.ConvTranspose2d):
                initialization.kaiming_normal_(layer.weight, a=math.sqrt(5))
                initialization.zeros_(layer.bias)

            if isinstance(layer, networks.Conv2d):
                initialization.kaiming_normal_(layer.weight, a=math.sqrt(5))
                initialization.zeros_(layer.bias)

            if isinstance(layer, networks.Linear):
                initialization.kaiming_normal_(layer.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:

        x = self.encoder0(x)

        x, a2, a1 = self.encoder1(t, x)
        x, b2, b1 = self.encoder2(t, x)
        x, c2, c1 = self.encoder3(t, x)

        x = self.bottleneck1(t, x)
        x = self.bottleneck2(t, x)
        x = self.bottleneck3(t, x)

        x = self.decoder3(t, x, c2, c1)
        x = self.decoder2(t, x, b2, b1)
        x = self.decoder1(t, x, a2, a1)

        x = self.decoder0(x)

        return x