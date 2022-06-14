from includes import *


# 单股LSTM模型
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

def trainLstm():

