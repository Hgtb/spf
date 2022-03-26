import torch
import torch.nn as nn
import torch.functional as func
import torch.optim as optim
from collections import OrderedDict


class CNNLSTM(nn.Module):
    def __init__(self, inputDates, outputDates, parametersNum, stocksNum):
        """
        :param inputDates: 输入的天数
        :param outputDates: 预测的天数
        :param parametersNum: 每只股票的参数个数
        :param stocksNum: 输入的股票数量
        """
        super(CNNLSTM, self).__init__()

        # [batchSize, parametersNum, stocksNum]
        self.conv_layer = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv1d(in_channels=1, out_channels=64, kernel_size=parametersNum)),
            ("Sigmoid", nn.Sigmoid()),
            ("conv1", nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)),
        ]))
        self.lstm_layer = torch.lstm()
        self.linear_layer = nn.Sequential(OrderedDict([
            ("Linear0", nn.Linear(in_features=, out_features=stocksNum * outputDates))
        ]))

    def forward(self, data):
        data = self.conv_layer(data)
        data = data.view([360, -1])
        data = self.lstm_layer(data)
        data = self.linear_layer(data)
        return data


class CNN2DLSTM(nn.Module):
    def __init__(self, inputDates, outputDates, parametersNum, stocksNum):
        """
        :param inputDates: 输入的天数
        :param outputDates: 预测的天数
        :param parametersNum: 每只股票的参数个数
        :param stocksNum: 输入的股票数量
        """
        super(CNN2DLSTM, self).__init__()

        # [batchSize, parametersNum, stocksNum]
        self.conv_layer = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(parametersNum, 3))), # kernel_size = [9, 3]
            ("Sigmoid", nn.Sigmoid()),
            ("", nn.)
            ("conv1", nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)),
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





