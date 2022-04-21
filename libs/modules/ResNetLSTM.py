import winnt

from includes import *


class ResidualBlock(nn.Module):
    """
    ResNet's block
    -> Copy -+-> Conv -> ReLU -> Conv -+-> Plus ->
             |_________________________|
    """
    def __init__(self, in_channels, out_channels, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.ReLU = nn.ReLU()
        self.downsample = downsample

    def forward(self, x: torch.Tensor):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.ReLU(x)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.ReLU(out)
        return out


class BottleneckBlock(nn.Module):
    """
    ResNet's bottleneck block
    -> Copy -+-> Conv1*1 -> ReLU -> Conv3*3 -> ReLU -> Conv1*1 -+-> Plus ->
             |__________________________________________________|
    """
    def __init__(self, in_channels, hidden_channels, out_channels, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.ReLU = nn.ReLU()
        self.downsample = downsample

    def forward(self, x: torch.Tensor):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.ReLU(x)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ReLU(x)

        out = self.conv3(x)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.ReLU(out)
        return out


class GoogLeBlock(nn.Module):
    """
    GoogLeNet's Block

           +- conv1*1 ->--------- -+
           +- conv1*1 ->- conv3*3 -+
    ->-+->-+- conv1*1 ->- conv5*5 -+->-+->-
       |   +- maxP1*1 ->- conv1*1 -+   |
       +->---------residual---------->-+
    """
    def __init__(self, in_channels, hidden_channels, out_channels, downsample=None):
        """
        通常out_channels = in_channels
        :param in_channels:
        :param hidden_channels:
        :param out_channels:
        :param downsample:
        """
        super(GoogLeBlock, self).__init__()
        # self.

    def forward(self, x):
        residual = x






class ResNetLSTM(nn.Module):
    def __init__(self):
        super(ResNetLSTM, self).__init__()


    def foeward(self, x):
        pass




