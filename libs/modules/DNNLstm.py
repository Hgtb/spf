import torch
import torch.nn as nn
import torch.functional as func
import torch.optim as optim
from collections import OrderedDict


# ToDo(Alex Han) 最简单的模型，由全连接层和LSTM组成
class DNNLstm(nn.Module):
    def __init__(self):
        super(DNNLstm, self).__init__()
        self.linear_layer1 = nn.Linear()
        self.linear_layer2 = nn.Linear()
        self.linear_layer3 = nn.Linear()
        self.lstm_layer1 = nn.LSTM(input_size=1570 * 9, hidden_size=28260, dropout=0.1)
        self.lstm_layer2 = nn.LSTM(input_size=28260, hidden_size=32768,  dropout=0.1)
        self.lstm_layer3 = nn.LSTM(input_size=32768, hidden_size=1570 * 30,  dropout=0.1)
        self.linear_layer = nn.Linear(in_features=)