from os.path import join
from libs.functions import *
from libs.dataProcess import DataProcess
import xarray as xr
import torch
import numpy as np


class DataSet:
    def __init__(self, data: xr.DataArray = None, dataSetPath: str = None):
        self.data: xr.DataArray = xr.DataArray([])
        if data is not None:
            self.data = data
        elif dataSetPath is not None:
            self.data = xr.load_dataarray(dataSetPath)
        else:
            raise "data not found"
        self.data.load()
        self.data = self.data.astype(np.float32)
        self.indexList = [i for i in range(len(self.data.Date))]  # 不知道怎么从xr.DataArray中获取一段时间的切片，用这种方法代替
        self.trainDays: int = 360
        self.targetDays: int = 30

    def setLength(self, trainDays, targetDays):
        self.trainDays = trainDays
        self.targetDays = targetDays

    def getTrainData(self, item):
        return torch.Tensor(self.data.isel(Date=self.indexList[item: item + self.trainDays]).values)
        # torch.Size([trainDays, stocksNum, parametersNum]) like torch.Size([360, 1570, 9])

    def getTargetData(self, item):
        return torch.Tensor(self.data
                            .isel(Date=self.indexList[item + self.trainDays: item + self.trainDays + self.targetDays])
                            .sel(Parameter="close")
                            .values)
        # torch.Size([targetDays, stocksNum]) like torch.Size([30, 1570])

    def __getitem__(self, item):
        return self.getTrainData(item), self.getTargetData(item)

    def __len__(self):
        return len(self.data.Date)


class DataLoader:
    def __init__(self, dataSet: DataSet):
        self.dataSet: DataSet = dataSet
        self.shifter: int = -1
        self.len = len(self.dataSet) - (360 + 30) + 1

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __next__(self):
        self.shifter += 1
        if self.shifter < self.len:
            return self.dataSet[self.shifter]
        else:
            raise StopIteration
