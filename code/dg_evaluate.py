import sys
import os
import pandas as pd
import numpy as np 
import time

import torch



from neuralop.models.fno import TFNO, FNO
from neuralop.training.trainer import Trainer
from neuralop.utils import count_model_params
from neuralop.losses.data_losses import LpLoss, H1Loss
from utils.brinkman_amitex import load_stokesbrinkman
from utils.helpers import (
    calc_pressure_grad_2d,
    recon_pressure_grad_2d_fd,
)

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# import evaluate

device = "cuda"

# Define path to saved model
saved_model_path = "./save_dir/2026-04-07T143406.659283Z_last.pt"

# Load the saved model
#data_processor = data_processor.to(device)
model = torch.load(saved_model_path, map_location=device, weights_only=False)
model.eval()
