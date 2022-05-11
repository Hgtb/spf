import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import xarray as xr
# import d2l.torch as d2l
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from collections import OrderedDict
from tqdm import tqdm


class Reshape(nn.Module):
    def __init__(self, *args):
        """
        Reshape层，可以在nn.Sequential中进行Tensor变形
        used like Reshape(1, 2, 3, 4)
        """
        super(Reshape, self).__init__()
        self.shape = list(args)

    def forward(self, x:torch.Tensor):
        # shape = self.shape
        # if self.shape[0] < x.shape[0]:
        #     shape[0] = x.shape[0]
        return x.view(self.shape)


class Permute(nn.Module):
    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)


def device():
    """获取可用设备"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

