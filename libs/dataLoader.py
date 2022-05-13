from os.path import join
from libs.functions import *
from libs.dataProcess import DataProcess
from typing import List
import xarray as xr
import torch
import numpy as np


def DataLength(dataPath):
    data = xr.load_dataarray(dataPath)
    return len(data)


def findSameNum(target, check_list: list):
    times = 0
    for item in check_list:
        if item == target:
            times += 1
    return times


class DataSet:
    r"""
    生成滑动窗口数据，每步返回两个值: trainData 和 targetData
    trainData.shape = torch.Size([trainDays, stocksNum, parametersNum])
    targetData.shape = torch.Size([targetDays, stocksNum]) 均为收盘价
    trainDays 与 targetDays 可使用 setLength 函数更改

    isel 函数：对数据进行切片，需要输入切片的开始和结束索引，以及进行切片操作的维度
    """

    def __init__(self, data: xr.DataArray = None, dataPath: str = None,
                 trainDays: int = 360, targetDays: int = 30,
                 isel: List[int] = None,  # [start_index, end_index]
                 device=None):
        self.device = device  # 自动将tensor转移device，device==None则不转移(默认使用CPU)
        self.data: xr.DataArray = xr.DataArray([])

        if data is not None:
            self.data = data
        elif dataPath is not None:
            self.data = xr.load_dataarray(dataPath)
        else:
            raise "data not found"

        self.indexList = [i for i in range(len(self.data.Date))]  # 不知道怎么从xr.DataArray中获取一段时间的切片，用这种方法代替
        if isel is not None:
            if isel[0] < 0:
                isel[0] = 0
            if isel[1] > len(self.data.Date):
                isel[1] = len(self.data.Date)
            self.data = self.data.isel(Date=self.indexList[isel[0]:isel[1]])  # 左闭右开
        self.isel_index = isel

        self.data.load()
        self.data = self.data.astype(np.float32)
        self.trainDays: int = trainDays
        self.targetDays: int = targetDays

    def isel(self, startIndex: int, endIndex: int, dim: str = "Date", inplace: bool = True):
        """
        use like   dataSet.sel(index=[0: 3000])
        or         trainDataSet = dataSet.sel(index=[0: 3000], inplace=False)
        """
        if inplace:
            self.data = eval("self.data.isel(" + dim + "=self.indexList[startIndex: endIndex])")  # 有bug
            return self
        else:
            return DataSet(eval("self.data.isel(" + dim + "=self.indexList[startIndex: endIndex])"))

    def setLength(self, trainDays, targetDays):
        self.trainDays = trainDays
        self.targetDays = targetDays

    def to_device(self, device: torch.device):
        self.device = device

    def getTrainData(self, item: int):
        r"""
        data.nc中获取的是[-1, 1440, 10]的数据，需要转换为[-1, 120, 120]的数据
        在使用返回的数据时，需要使用 trainData.reshape(360, 1, 120, 120)
            reshape的工作方式:
                                            0, 1, 2, 3
               0, 1, 2, 3, 4, 5, 6, 7  ->   4, 5, 6, 7
               9, 8, 7, 6, 5, 4, 3, 2  ->   9, 8, 7, 6
                                            5, 4, 3, 2
        :return: torch.Size([trainDays, stocksNum, parametersNum]) like torch.Size([360, 1570, 9])
        """
        if self.device is None:
            return torch.Tensor(self.data.isel(Date=self.indexList[item: item + self.trainDays]).values)
        else:
            return torch.Tensor(self.data.isel(Date=self.indexList[item: item + self.trainDays]).values).to(self.device)

    def getTargetData(self, item: int, Parameter: str = "close"):
        r"""
        :return: torch.Size([targetDays, stocksNum]) like torch.Size([30, 1440])
        """
        target_data = torch.Tensor(self.data
                                   .isel(
                                    Date=self.indexList[item + self.trainDays: item + self.trainDays + self.targetDays])
                                   .sel(Parameter=Parameter)
                                   .values)
        times = findSameNum(target=Parameter, check_list=list(self.data.coords["Parameter"].data))
        if times == 0:
            raise "Parameter Error, can't find " + "'" + Parameter + "'" + " in " \
                  + str(list(self.data.coords["Parameter"].data))
        elif times > 1:
            # print("WARNING : Found " + str(times) + " '" + Parameter + "' in data")
            target_data = target_data[:, :, 1]
            target_data = target_data.squeeze(dim=-1)

        if self.device is None:
            return target_data
        else:
            return target_data.to(self.device)

    def __getitem__(self, item):
        return self.getTrainData(item), self.getTargetData(item)

    def __len__(self):
        return len(self.data.Date)


class DataLoader:
    def __init__(self, dataSet: DataSet, device: torch.device = None):
        self.device: torch.device = device
        self.dataSet: DataSet = dataSet
        if device is not None:
            self.dataSet.to_device(device)
        self.shifter: int = -1
        self._len: int = 0
        self._calculate_len()

    def _calculate_len(self):
        self._len = len(self.dataSet) - (self.dataSet.trainDays + self.dataSet.targetDays) + 1
        if self._len <= 0:
            raise Exception(f"The length of DataSet is less than 0, the length of DataSet is {len(self.dataSet)}, need longer DataSet")

    def resetShifter(self, shift:int = 0):
        """DataLoader will start from `shift`"""
        if (shift < 0) or (shift > self._len):
            raise Exception(f"shift must in [0, {self._len}], but got {shift}")

    def to_device(self, device: torch.device):
        self.device = device
        self.dataSet.to_device(device)

    def isel(self, startIndex: int, endIndex: int):
        self.dataSet.isel(startIndex=startIndex, endIndex=startIndex + (endIndex + (self.dataSet.trainDays + self.dataSet.targetDays) - 1), inplace=True)
        self._calculate_len()

    def __len__(self):
        return self._len

    def __iter__(self):
        return self

    def __next__(self):
        self.shifter += 1
        if self.shifter < self._len:
            return self.dataSet[self.shifter]
        else:
            raise StopIteration
