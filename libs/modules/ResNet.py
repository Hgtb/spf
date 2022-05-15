from libs.modules.includes import *


"""
ResNet use ResNet50
"""


class BasicBlock(nn.Module):
    """
    ResNet 18 or 34  block
    -> Copy -+-> Conv -> ReLU -> Conv -+-> Plus ->
             |_________________________|
    """

    def __init__(self, in_channels, out_channels, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.ReLU = nn.ReLU()
        self.downsample = downsample

    def forward(self, x: torch.Tensor):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.ReLU(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.ReLU(out)
        return out


class BottleneckBlock(nn.Module):
    """
    ResNet 50 or 101 or 152  block
    --+-> Conv1*1 -> BN ->  ReLU -> Conv3*3 -> BN -> ReLU -> Conv1*1 -> BN -+-> ReLU ->
      +--------------------------(Conv1*1 -> BN)----------------------------+
                                  自动选择是否卷积x
    """

    def __init__(self, in_channels, hidden_channels, out_channels, stride=1, downsample=None):
        """
        当 in_channels != out_channels 时，自动对x进行卷积升维。
        当 stride > 1 时，若没有传入 downsample ，则使用 conv1*1 stride=2 来对x进行下采样，统一尺寸
        """
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(1, 1), stride=(1, 1),
                      padding=0, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=(3, 3),
                      stride=(stride, stride), padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1),
                      padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # ToDo(Alex Han) 在自动网络生成功能完成时，需要将这里的 downsample 模块放到自动网络生成模块中，并从 downSample 传入
        self.downsample = downsample
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                          stride=(stride, stride), padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.ReLU = nn.ReLU()

    def forward(self, x: torch.Tensor):
        residual = x
        out = self.conv3(self.conv2(self.conv1(x)))

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = out + residual
        out = self.ReLU(out)
        return out


class Conv1X(nn.Module):
    def __init__(self, use_checkpoint):
        super(Conv1X, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.convBlock = nn.Sequential(  # 1*120*120 -> 64*116*116 -> 64*112*112 -> 64*56*565
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        # print("Conv1X : ")
        # print("conv1 : ", self.convBlock[0].state_dict())
        # print("conv2 : ", self.convBlock[3].state_dict())
        if self.use_checkpoint:
            x = checkpoint_sequential(self.convBlock, 2, x)
            # print("Conv1X DONE")
            return x
        else:
            x = self.convBlock(x)
            return x


class Conv2X(nn.Module):
    def __init__(self, use_checkpoint):
        super(Conv2X, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.convBlock = nn.Sequential(
            BottleneckBlock(in_channels=64, hidden_channels=64, out_channels=256),
            BottleneckBlock(in_channels=256, hidden_channels=64, out_channels=256),
            # BottleneckBlock(in_channels=256, hidden_channels=64, out_channels=256)
        )

    def forward(self, x):
        if self.use_checkpoint:
            x = checkpoint_sequential(self.convBlock, 2, x)
            return x
        else:
            x = self.convBlock(x)
            return x


class Conv3X(nn.Module):
    def __init__(self, use_checkpoint):
        super(Conv3X, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.convBlock = nn.Sequential(
            BottleneckBlock(in_channels=256, hidden_channels=128, out_channels=512, stride=2),
            BottleneckBlock(in_channels=512, hidden_channels=128, out_channels=512),
            # BottleneckBlock(in_channels=512, hidden_channels=128, out_channels=512),
            # BottleneckBlock(in_channels=512, hidden_channels=128, out_channels=512)
        )

    def forward(self, x):
        if self.use_checkpoint:
            x = checkpoint_sequential(self.convBlock, 2, x)
            return x
        else:
            x = self.convBlock(x)
            return x


class Conv4X(nn.Module):
    def __init__(self, use_checkpoint):
        super(Conv4X, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.convBlock = nn.Sequential(
            BottleneckBlock(in_channels=512, hidden_channels=256, out_channels=1024, stride=2),
            BottleneckBlock(in_channels=1024, hidden_channels=256, out_channels=1024),
            # BottleneckBlock(in_channels=1024, hidden_channels=256, out_channels=1024),
            # BottleneckBlock(in_channels=1024, hidden_channels=256, out_channels=1024),
            # BottleneckBlock(in_channels=1024, hidden_channels=256, out_channels=1024),
            # BottleneckBlock(in_channels=1024, hidden_channels=256, out_channels=1024)
        )

    def forward(self, x):
        if self.use_checkpoint:
            x = checkpoint_sequential(self.convBlock, 2, x)
            return x
        else:
            x = self.convBlock(x)
            return x


class Conv5X(nn.Module):
    def __init__(self, use_checkpoint):
        super(Conv5X, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.convBlock = nn.Sequential(
            BottleneckBlock(in_channels=1024, hidden_channels=512, out_channels=2048, stride=2),
            BottleneckBlock(in_channels=2048, hidden_channels=512, out_channels=2048),
            # BottleneckBlock(in_channels=2048, hidden_channels=512, out_channels=2048)
        )

    def forward(self, x):
        if self.use_checkpoint:
            x = checkpoint_sequential(self.convBlock, 2, x)
            return x
        else:
            x = self.convBlock(x)
            return x


# ToDo(Alex Han) 添加自动网络生成模块，根据传入参数自动生成四个动态网络模块(conv2_x, conv3_x, conv4_x, conv5_x)
class ResNet(nn.Module):
    def __init__(self, layer_nums: list, use_checkpoint=False):
        # layer_nums = [conv2_x_layer_nums, conv3_x_layer_nums, conv4_x_layer_nums, conv5_x_layer_nums]
        # In ResNet50, layer_nums = [3, 4, 6, 3]
        # layer_nums will be used in future dynamic network generation features
        super(ResNet, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.layer_nums = layer_nums
        self.conv1_x = Conv1X(use_checkpoint=use_checkpoint)
        self.conv2_x = Conv2X(use_checkpoint=use_checkpoint)
        self.conv3_x = Conv3X(use_checkpoint=use_checkpoint)
        self.conv4_x = Conv4X(use_checkpoint=use_checkpoint)
        self.conv5_x = Conv5X(use_checkpoint=use_checkpoint)
        self.avgPool = nn.AvgPool2d(kernel_size=7, stride=1)

    def forward(self, x):
        # x : torch.Size([360, 1, 120, 120])
        # print("ResNet : ")
        x = self.conv1_x(x)
        # print("conv1_x : ", x[0])
        x = self.conv2_x(x)
        # print("conv1_x : ", x[0])
        x = self.conv3_x(x)
        # print("conv1_x : ", x[0])
        x = self.conv4_x(x)
        # print("conv1_x : ", x[0])
        x = self.conv5_x(x)
        # print("conv1_x : ", x[0])
        x = self.avgPool(x)
        # print("avgPool : ", x[0])
        # print("ResNet DONE")
        # x : torch.Size([360, 2048, 1, 1])
        return x



