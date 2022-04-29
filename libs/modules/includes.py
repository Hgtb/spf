import math
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from collections import OrderedDict


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
