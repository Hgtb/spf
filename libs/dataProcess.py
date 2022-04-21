import dask
import pandas as pd
from tqdm import tqdm
from os.path import join
from libs.functions import detectFolder
import numpy as np
import xarray as xr
import json


# ToDo(Alex Han) 在完成因子下载函数后将因子与行情数据合并
# ToDo(Alex Han) 从行情数据中获得股票数量最多或最少的日数据的股票列表作为筛选用的股票列表，将数据处理从数据下载中抽取出来

# class DataProcessLog:
#     """
#     DataProcess类的Log类，能够记录DataProcess的数据读取、输出路径，并记录DataProcess的操作
#     """
#     def __init__(self):
#         self.loadPath = ""
#         self.storePath = ""
#         self.trackBack = []
#         pass
#
#     def __str__(self):
#         pass
#
#     def store(self, path=""):
#
#         pass

# ToDo(Alex Han) 使用xarray实现数据的加载、保存、修改。
# ToDo(Alex Han) 统一数据保存格式为netCDF
class DataProcess:
    def __init__(self, dataSetPath: str = "../dataSet/"):
        self.rawBasicDataPath = join(dataSetPath, "rawData/basicData/")  #
        self.rawMarketDataPath = join(dataSetPath, "rawData/marketData/")  #

        self.basicDataPath = join(dataSetPath, "data/basicData/")  #
        self.marketDataPath = join(dataSetPath, "data/marketData/")  #

        self.__detectFolders()

        # dailyData的三个数据轴
        self.dims = ["Date", "Stock", "Parameter"]
        self.tradeCal = []
        self.stockList = []
        self.parameters = []
        self.dailyData = xr.DataArray()

        self.trackBack = []

    def __detectFolders(self):
        detectFolder(self.rawBasicDataPath)
        detectFolder(self.rawMarketDataPath)
        detectFolder(self.basicDataPath)
        detectFolder(self.marketDataPath)

    def __loadTradeCal(self):
        self.tradeCal = pd.read_csv(self.rawBasicDataPath + "tradeCal.csv")
        self.tradeCal = list(self.tradeCal["trade_date"])
        return self.tradeCal

    def __loadStockList(self):
        self.stockList = pd.read_csv(self.rawBasicDataPath + "stockList.csv")
        self.stockList = list(self.stockList["ts_code"])
        return self.stockList

    def __loadParameters(self):
        self.parameters = pd.read_csv(self.rawBasicDataPath + "parameters.csv")
        self.parameters = list(self.parameters["Parameters"])
        return self.parameters

    def __loadDailyData(self) -> np.ndarray:
        dailyData = []
        for date in tqdm(self.tradeCal, desc="Load daily data"):
            data = pd.read_csv(join(self.rawMarketDataPath, str(date) + ".csv"))
            dailyData.append(data.values)
        return np.array(dailyData)  # shape = [date, stock, parameters]

    def loadData(self):
        tradeCal = self.__loadTradeCal()
        stockList = self.__loadStockList()
        parameters = self.__loadParameters()
        dailyData = self.__loadDailyData()
        coordinates = {"Date": tradeCal, "Stock": stockList, "Parameter": parameters}
        self.dailyData = xr.DataArray(data=dailyData, dims=self.dims, coords=coordinates)

    def storeData(self, path: str = None):
        """
        必须删掉"ts_code项才能正常保存dailyData"
        """
        if path is None:
            path = self.basicDataPath
        with open(join(path, "basicData.json"), "w") as fp:
            json.dump(obj={
                "dims": self.dims,
                "Date": self.tradeCal,
                "Stock": self.stockList,
                "Parameter": self.parameters
            }, fp=fp)
        self.dailyData.to_netcdf(join(self.marketDataPath, "data.nc"))

    def getData(self):
        return self.dims, self.stockList, self.tradeCal, self.parameters, self.dailyData

    # 下面是操作函数
    def dimSpeculate(self, labels: list):
        """
        推导label所在维度，若找不到则返回None，若找到一个维度则返回该维度名称，若找到两个及以上维度则无法判断维度返回False
        """
        _dimCount = 0
        dim = None
        labels = set(labels)
        if labels <= set(self.dailyData):
            _dimCount += 1
            dim = self.dims[0]
        if labels <= set(self.stockList):
            _dimCount += 1
            dim = self.dims[1]
        if labels <= set(self.parameters):
            _dimCount += 1
            dim = self.dims[2]
        if _dimCount <= 1:
            return dim
        else:
            return False

    def dropDataFrame(self, labels: list, dim: str) -> None:  # exp: dropDataFrame("ts_code", "trade_date")
        self.dailyData = self.dailyData.drop(labels=labels, dim=dim)  # 有提示但是能正常运行

    def minMaxNormal(self) -> None:
        """
        min-max归一化，将数据转换为[0, 1]区间内的值
        """
        # self.dailyData.max("Date").sel(Parameter="close").to_netcdf(join(self.marketDataPath, "maxClose.nc"))
        # self.dailyData.min("Date").sel(Parameter="close").to_netcdf(join(self.marketDataPath, "minClose.nc"))
        self.dailyData.max("Date").to_netcdf(join(self.marketDataPath, "max.nc"))
        self.dailyData.min("Date").to_netcdf(join(self.marketDataPath, "min.nc"))

        def max_min_normal(ds: xr.DataArray, _dim):
            return (ds - ds.min(_dim)) / (ds.max(_dim) - ds.min(_dim))

        self.dailyData = self.dailyData.groupby("Stock").apply(max_min_normal, args=("Date",))


if __name__ == "__main__":
    dataProcess = DataProcess()
    dataProcess.loadData()
    dataProcess.dropDataFrame(labels=["ts_code", "trade_date"], dim="Parameter")
    dataProcess.storeData()
