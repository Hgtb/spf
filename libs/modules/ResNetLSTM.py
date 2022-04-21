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


class InceptionV1(nn.Module):
    """
       +- conv1*1 ->--------- -+
       +- conv1*1 ->- conv3*3 -+
    ->-+- conv1*1 ->- conv5*5 -+->-
       +- maxP3*3 ->- conv1*1 -+
    """
    def __init__(self, in_channels, out_channels, downsample=None):
        super(InceptionV1, self).__init__()
        # self.
        self.ReLU = nn.ReLU()
        pass

    def forward(self, x):

        pass


class InceptionV1ResiduaBlock(nn.Module):
    """
    结合InceptionV1模块和残差设计的模块
    ->-+->- InceptionV1 ->-+->-
       |                   |
       +->--- Conv1*1 --->-+
    """

    def __init__(self, in_channels, out_channels, downsample=None):
        """
        中间层自动生成
        """
        super(InceptionV1ResiduaBlock, self).__init__()
        self.inceptionV1 = InceptionV1(in_channels=in_channels, out_channels=out_channels)

        self.Conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.Conv_bn = nn.BatchNorm2d()

        self.ReLU = nn.ReLU()

    def forward(self, x):
        residual = x

        out = self.inceptionV1(x)
        out = self.inceptionV1_bn(out)


class ResNetLSTM(nn.Module):
    def __init__(self):
        super(ResNetLSTM, self).__init__()

    def foeward(self, x):
        pass
