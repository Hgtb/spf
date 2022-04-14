import math
import torch
import torch.nn as nn
import torch.functional as func
import torch.optim as optim
from collections import OrderedDict

# ToDo(Alex Han) 使用json或config文件来保存/生成模型

def calculateSize(size: int) -> int:
    return int((size - 2.0) / 2.0)


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
    def __init__(self, channel_size:list, kernel_size:tuple, padding:list):
        """
        channel_size : [in_channels, hidden_channels, output_channels]
        kernel_size : [H, W]
        padding: [Conv1_padding, Conv2_padding]
        Block : Conv2d -> ReLU -> Conv2d ->AvgPool2d -> ReLU
        """
        super(Conv2dBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=channel_size[0], out_channels=channel_size[1], kernel_size=kernel_size, stride=1, padding=padding[0]),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel_size[1], out_channels=channel_size[2], kernel_size=kernel_size, stride=1, padding=padding[1]),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.ReLU()
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
            # 1570 --> 1568 -> 1566 --> 783 -> 781 -> 779 -> 390 -> 388 -> 386 -> 193 -> 97 -> 95 -> 93 -> 47 -> 45 -> 43 -> 22 -> 20 -> 18 -> 9 -> 7 -> 5 -> 3 -> 1...
            ("conv0", nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(parametersNum, 3))), # kernel_size = (9, 3)
            ("Sigmoid", nn.Sigmoid()),
            ("conv1", nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(parametersNum, 3))),
            ("AvgPool", nn.AvgPool2d(kernel_size=2, stride=2)),
            ("Sigmoid", nn.ReLU),
            ("Block", nn.ReLU),
            ("Sigmoid", nn.ReLU),
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


class LSTM(nn.Module):
    """
    单股票LSTM模型
    输入：tensor.shape = [batch_size, -1, input_size]
    """
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm_layer = nn.LSTM(input_size=9, num_layers=3, hidden_size=1024, batch_first=True, dropout=0.2)
        self.linear_layer = nn.Linear(in_features=1024, out_features=30)

    def forward(self, x):
        x, (h_n, h_c) = self.lstm_layer(x, None)
        x = self.linear_layer(x)
        return x





