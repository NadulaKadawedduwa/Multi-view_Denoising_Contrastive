import torch
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as  dist
import torch.multiprocessing as mp
import os
import argparse
import numpy as np

from dataset import CustomDataset
from smaller_model import UNet


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    max_pixel = np.max(img1)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr