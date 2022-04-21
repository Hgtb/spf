from includes import *


# ToDo(Alex Han) 使用json或config文件来保存/生成模型

# def calculateSize(size: int) -> int:
#     return int((size - 2.0) / 2.0)


# class Spp2d(nn.Module):
#     def __init__(self):
#         super(Spp2d, self).__init__()
#
#         pass
#
#     def forward(self, x):
#
#         pass


# class CNN1DLSTM(nn.Module):
#     def __init__(self, inputDates, outputDates, parametersNum, stocksNum):
#         """
#         :param inputDates: 输入的天数
#         :param outputDates: 预测的天数
#         :param parametersNum: 每只股票的参数个数
#         :param stocksNum: 输入的股票数量
#         """
#         super(CNN1DLSTM, self).__init__()
#
#         # [batchSize, parametersNum, stocksNum]
#         self.conv_layer = nn.Sequential(OrderedDict([
#             ("conv0", nn.Conv1d(in_channels=1, out_channels=64, kernel_size=parametersNum)),
#             ("Sigmoid", nn.Sigmoid()),
#             ("conv1", nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)),
#         ]))
#         self.lstm_layer = torch.lstm()
#         self.linear_layer = nn.Sequential(OrderedDict([
#             ("Linear0", nn.Linear(in_features=, out_features=stocksNum * outputDates))
#         ]))
#
#     def forward(self, data):
#         data = self.conv_layer(data)
#         data = data.view([360, -1])
#         data = self.lstm_layer(data)
#         data = self.linear_layer(data)
#         return data


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





class Conv2dBlock(nn.Module):
    def __init__(self, channel_size:list, kernel_size:tuple):
        """
        channel_size : [in_channels, hidden_channels, output_channels]
        kernel_size : [H, W]
        padding: [Conv1_padding, Conv2_padding]
        Block : Conv2d -> ReLU -> Conv2d ->AvgPool2d -> ReLU
        """
        super(Conv2dBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=channel_size[0], out_channels=channel_size[1], kernel_size=kernel_size, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel_size[1], out_channels=channel_size[2], kernel_size=kernel_size, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )

    def forward(self, x):
        return self.block(x)


class CNN2dLSTM(nn.Module):
    """
    Todo(Alex Han) 需要动态初始化网络大小，目前固定为1570支股票
    """
    def __init__(self, inputDates, outputDates, parametersNum, stocksNum):
        """
        :param inputDates: 输入的天数
        :param outputDates: 预测的天数
        :param parametersNum: 每只股票的参数个数
        :param stocksNum: 输入的股票数量
        """
        super(CNN2dLSTM, self).__init__()
        # 使用[parametersNum, 3]大小的卷积核在股票维度上平移卷积
        # [batchSize, channel, parametersNum, stocksNum]
        self.conv_layer = nn.Sequential(OrderedDict([
            # 1570 --> 1568 -> 784 --> 782 -> 391 -> 389 -> ...
            # 1570 --> 1568 -> 1566 --> 783 -> 781 -> 779 -> 390 -> 388 -> 386 -> 193 -> 191 -> 189 -> 95 -> 93 -> 91 -> 46 -> 44 -> 42 -> 40 -> 20 -> 18 -> 16 -> 8 -> 6 -> 4 -> 2 -> 1
            ("convBlock0", Conv2dBlock(channel_size=[1, 32, 64], kernel_size=(parametersNum, 3))), # 1570 -> 783
            ("convBlock1", Conv2dBlock(channel_size=[64, 128, 256], kernel_size=(parametersNum, 3))), # 783 -> 390
            ("convBlock2", Conv2dBlock(channel_size=[256, 512, 1024], kernel_size=(parametersNum, 3))), # 390 -> 193
            ("convBlock3", Conv2dBlock(channel_size=[64, 128, 256], kernel_size=(parametersNum, 3))), # 193 -> 783
            ("convBlock4", Conv2dBlock(channel_size=[64, 128, 256], kernel_size=(parametersNum, 3))), # 1570 -> 783
            ("convBlock5", Conv2dBlock(channel_size=[64, 128, 256], kernel_size=(parametersNum, 3))) # 1570 -> 783
        ]))
        self.lstm_layer = torch.lstm()
        self.linear_layer = nn.Sequential(OrderedDict([
            ("Linear0", nn.Linear(in_features=, out_features=stocksNum * outputDates))
        ]))

    def forward(self, data):
        data = self.conv_layer(data)
        data = data.view([360, -1])
        data, (h_n, h_c) = self.lstm_layer(data, None)
        data = self.linear_layer(data)
        return data


# ToDo(Alex Han) CNNLSTM新网络结构


# ToDo(Alex Han) 强化学习








