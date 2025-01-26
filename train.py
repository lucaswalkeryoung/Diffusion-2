# -------------------------------------------------------------------------------------------------
# --------------------------------------- Diffusion :: Main ---------------------------------------
# -------------------------------------------------------------------------------------------------
import torch


# -------------------------------------------------------------------------------------------------
# ------------------------ Environment, Initialization, and Infrastructure ------------------------
# -------------------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

t_channels =  32 # timestep embedding size/dimension
s_channels =  32 # spatial  embedding size/dimension
timesteps = 1000 # total denoising steps
max_beta  = 1e-5
min_beta  = 1e-5
max_lr    = 1e-5
min_lr    = 1e-5

CYCLES =  8 # number of epochs before restart
EPOCHS = 32
BATCH_SIZE = 64
