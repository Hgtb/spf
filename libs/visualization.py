import torch
import xarray as xr
import matplotlib.pyplot as plt
import plotly as plty
import plotly.graph_objects as go
import plotly_express as px
from plotly.subplots import make_subplots


def tensorToArray(data: torch.Tensor):
    return data.detach().cpu().numpy()


def toArray(data):
    if type(data) == torch.Tensor:
        return tensorToArray(data)
    else:
        return data


# def


def showPolyline(predict, target, facet_column_wrap):
    """
    :param predict:  [batch_size, N]
    :param target:   [batch_size. N]  The shape of target must be the same as the shape of predict
    :param facet_column_wrap: graphs in a line
    """
    predict = toArray(predict)
    target = target(target)
    fig = px.line()


