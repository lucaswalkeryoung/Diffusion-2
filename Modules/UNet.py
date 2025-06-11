# -------------------------------------------------------------------------------------------------
# ----------------------- Diffusion :: The U-Net and Assorted Common Blocks -----------------------
# -------------------------------------------------------------------------------------------------
import torch.nn.init as initialization
import torch.nn as networks
import torch

from . Encoder import EncoderEntry
from . Encoder import EncoderBlock
from . Decoder import DecoderBlock
from . Decoder import DecoderFinal


# ---------------------------------------------------------------------------------------------
# ------------------------------------ MODULE :: The U-Net ------------------------------------
# ---------------------------------------------------------------------------------------------
class UNet(networks.Module):

    def __init__(self, t_channels: int = 32, s_channels: int = 32) -> None:
        super().__init__()

        self.encoder1 = EncoderEntry(3,   16)
        self.encoder2 = EncoderBlock(16,  32, t_channels)
        self.encoder3 = EncoderBlock(32,  64, t_channels)
        self.encoder4 = EncoderBlock(64, 128, t_channels)

        self.decoder4 = DecoderBlock(128, 64, t_channels)
        self.decoder3 = DecoderBlock(64,  32, t_channels)
        self.decoder2 = DecoderBlock(32,  16, t_channels)
        self.decoder1 = DecoderFinal(16,   3)

        for layer in self.modules():

            if isinstance(layer, networks.ConvTranspose2d):
                initialization.kaiming_uniform_(layer.weight)
                initialization.zeros_(layer.bias)

            if isinstance(layer, networks.Conv2d):
                initialization.kaiming_uniform_(layer.weight)
                initialization.zeros_(layer.bias)

            if isinstance(layer, networks.Linear):
                initialization.kaiming_uniform_(layer.weight)
                initialization.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:

        x = self.encoder1(x)
        x, a2, a1 = self.encoder2(t, x)
        x, b2, b1 = self.encoder3(t, x)
        x, c2, c1 = self.encoder4(t, x)

        x = self.decoder4(t, x, c2, c1)
        x = self.decoder3(t, x, b2, b1)
        x = self.decoder2(t, x, a2, a1)
        x = self.decoder1(x)

        return x