import torch
import xarray as xr
import matplotlib.pyplot as plt
import plotly as plty
import plotly.graph_objects as go
import plotly_express as px
from plotly.subplots import make_subplots
import numpy as np
import plotly.offline as py


def tensorToArray(data: torch.Tensor):
    return data.detach().cpu().numpy()


def toArray(data):
    if type(data) == torch.Tensor:
        return tensorToArray(data)
    else:
        return data


def drawLoss(loss: list):
    """绘制loss曲线， 输入必须为list[float]格式"""
    trace = dict(
        y=np.array(loss),
        mode="lines",
        name="loss"
    )
    fig = go.Figure(trace)
    fig.show()


def drawTargetsAndPredicts(predict_data, history_data, shifter: int, stock_index: int):
    # y = history_data.numpy().reshape(-1)
    predicts_ = predict_data.permute(0, 2, 1)
    targets_ = predict_data.permute(0, 2, 1)
    data_seq = []
    for batch in range(len(predict_data)):
        trace = go.Scatter(
            x=np.linspace(shifter + 5 + batch, shifter + 2 * 5 + batch, 5),
            y=predicts_[batch][stock_index].numpy(),
            mode="lines",
            name=f"预测数据{batch}",
        )
        data_seq.append(trace)
    for batch in range(len(history_data)):
        trace = go.Scatter(
            x=np.linspace(shifter + 5 + batch, shifter + 2 * 5 + batch, 5),
            y=targets_[batch][stock_index].numpy(),
            mode="lines",
            name=f"历史数据{batch}",
        )
        data_seq.append(trace)
    py.iplot(data_seq)


def drawTargetAndPredicts(predicts: torch.Tensor, targets: torch.Tensor,
                          batch_index: int, stock_index: int):
    """
    :param predicts: (356, 5, 1440)
    :param targets: (356, 5, 1440)
    :param batch_index:
    :param stock_index: in set [0, 1440)
    :return:
    """
    predicts_ = predicts[batch_index].permute(1, 0)
    targets_ = targets[batch_index].permute(1, 0)
    trace0 = go.Scatter(
        # x = np.linspace(0, 1, 30),
        y=predicts_[stock_index].numpy(),
        mode="lines",
        name="predict"
    )
    trace1 = go.Scatter(
        # x = np.linspace(0, 1, 30),
        y=targets_[stock_index].numpy(),
        mode="lines",
        name="target"
    )
    data = [trace0, trace1]
    py.iplot(data)


def showPolyline(predict, target, facet_column_wrap):
    """
    :param predict:  [batch_size, N]
    :param target:   [batch_size. N]  The shape of target must be the same as the shape of predict
    :param facet_column_wrap: graphs in a line
    """
    predict = toArray(predict)
    target = target(target)
    fig = px.line()
