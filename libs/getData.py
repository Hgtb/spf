from libs.functions import *
import numpy as np
from os.path import join
import pandas as pd
import tushare as ts
from tqdm import tqdm
import xarray as xr


def dateNoGreaterThan(date: int, tradeCal: list) -> int:
    for index in range(len(tradeCal) - 1, -1, -1):  # index = 3, 2, 1, 0
        if int(tradeCal[index]) <= date:
            return int(tradeCal[index])  # 20100103


def dateNotLessThan(date: int, tradeCal: list) -> int:
    for index in range(len(tradeCal)):
        if int(tradeCal[index]) >= date:
            return int(tradeCal[index])


def getCurrentDate() -> int:
    return int(datetime.datetime.now().strftime('%Y%m%d'))


def getYesterday() -> int:
    return int((datetime.date.today() + datetime.timedelta(days=-1)).strftime("%Y%m%d"))


def getCurrentTime() -> int:
    return int(datetime.datetime.now().strftime('%H'))


def getLatestDataDate(tradeCal: list) -> int:
    """
    根据当前时间判断最新数据的日期
    """
    date = 0
    if getCurrentTime() > 16:
        date = getCurrentDate()
    else:
        date = getYesterday()
    return dateNoGreaterThan(date, tradeCal)


class DownloadDataInterface:
    def __init__(self):
        pass

    def saveTradeCal(self):
        pass

    def saveStockList(self):
        pass

    def saveDailyData(self):
        pass


# ToDo(Alex Han) 直接将数据存储为netCDF
class DownloadData(DownloadDataInterface):
    def __init__(self, Token: str, dataSetPath: str = "../dataSet/"):
        """
        需要Tushare账号积分至少600
        调用复权行情,每日指标
        在没有特殊要求时，可以直接调用storeData函数获取数据
        """
        super().__init__()
        ts.set_token(Token)
        self.pro = ts.pro_api()
        self.wait = FrequencyLimitation()

        self.rawDataPath = join(dataSetPath, "rawData/")
        self.basicDataStoreFolder = join(dataSetPath, "rawData/basicData/")
        self.marketDataStorePath = join(dataSetPath, "rawData/marketData/")
        self.detectFolder()

        self.stockList = pd.DataFrame([])  # 从第一个获取的dailyData中获取，用于统一所有数据的股票及股票顺序
        self.dataStartDate = 20100104
        self.downloadStartDate = 20091208  # 遍历找出的日期，这一天与20220429的股票列表的交集有1440支
        self.endDate = 0
        self.tradeCal = []
        self.hfq_columns = ["open", "high", "low", "close", "pre_close", "change"]   # 要复权的日行情数据
        self.parameters = ["ts_code", "trade_date", "open", "high", "low", "close",
                           "pre_close", "change", "pct_chg", "vol", "amount"]

    def setStartDate(self, dataStartDate):
        self.dataStartDate = dataStartDate  # need a check

    def detectFolder(self):
        detectFolder(self.basicDataStoreFolder)
        detectFolder(self.marketDataStorePath)

    def setStorePath(self, BasicDataStoreFolder, MarketDataStorePath):
        self.basicDataStoreFolder = BasicDataStoreFolder
        self.marketDataStorePath = MarketDataStorePath
        self.detectFolder()

    def storeTradeCal(self):
        self.wait()
        tradeCal = self.pro.trade_cal(**{"start_date": self.downloadStartDate, "is_open": 1}, fields=["cal_date"])
        tradeCal = list(tradeCal["cal_date"])
        self.endDate = getLatestDataDate(tradeCal)
        self.tradeCal = [i for i in tradeCal if (int(i) <= self.endDate)]  # 从downloadStartDate开始，用于提前于dataStartDate下载数据，以补全数据
        tradeCal_store = [i for i in self.tradeCal if (int(i) >= self.dataStartDate)]  # 从dataStartDate开始，用于对保存的数据进行索引
        tradeCal_store = pd.DataFrame(data=tradeCal_store, columns=["trade_date"])
        storeAsCsv(tradeCal_store, self.basicDataStoreFolder + "tradeCal.csv")

    def storeStockList(self):
        # ToDo(Alex Han) 需要加入自动找1440支股票的功能(downloadStartDate的股票列表与剔除ST股的最新股票列表的交集大于等于1440)， 需要支持传入股票池
        if self.stockList.empty:
            self.wait()
            # 可使用 pro.stock_basic(fields=["ts_code"]) 获取最新所有股票的列表，但是未来股票数量会超过单次数据调取上限
            latestStockList = pd.concat([self.pro.stock_basic(**{"exchange": "SZSE"}, fields=["ts_code", "name"]),
                                         self.pro.stock_basic(**{"exchange": "SSE"}, fields=["ts_code", "name"])],
                                        ignore_index=True)  # 最近交易日股票列表

            # 删去ST股
            latestStockList = latestStockList[~latestStockList.name.str.contains('ST')].reset_index(drop=True)

            latestStockList = pd.DataFrame(latestStockList["ts_code"])

            self.wait()
            daily = self.pro.daily(**{"trade_date": self.downloadStartDate}, fields=["ts_code"])
            # 最后一个与最近交易日共有1440支股票的日期是20091208

            # self.wait()
            # dailyStart = self.pro.daily(**{"trade_date": self.startDate}, fields=["ts_code"])
            # # 训练数据开始日的股票列表

            # 取开始日期的股票列表与开始日、最近交易日股票列表的交集
            stockList = pd.merge(latestStockList, daily["ts_code"], how="inner")["ts_code"]  # type(stockList) = pd.Series
            # stockList = pd.merge(stockList, dailyStart["ts_code"], how="inner")["ts_code"]
            stockList = stockList.sort_values()
            stockList = stockList.reset_index(drop=True)
            self.stockList = stockList

        storeAsCsv(self.stockList, join(self.basicDataStoreFolder, "stockList.csv"))

    def setStockList(self, stockListPath: str):
        """读取csv文件，从columns中选取‘ts_code’列"""
        stockList = pd.read_csv(stockListPath)["ts_code"]
        storeAsCsv(stockList, join(self.basicDataStoreFolder, "stockList.csv"))
        self.stockList = stockList

    def storeParameters(self):
        df = pd.DataFrame({"Parameters": self.parameters})
        df.to_csv(join(self.basicDataStoreFolder, "parameters.csv"), index=False)

    def storeDailyData(self):
        """
        保存后复权股票日行情数据
        """
        if self.stockList.empty:
            self.storeStockList()
        self.endDate = getLatestDataDate(self.tradeCal)

        self.showDownloadInfo()  # 显示要下载的目标数据的参数

        rem = pd.DataFrame()  # 存储前一天的行情数据，用于补全数据

        log = readLog(logPath=self.rawDataPath + "storeLog.json")  # 读取数据存储日志，根据日志记录的最后一天继续下载数据

        tradeCal = []
        if int(log["stockNum"]) != len(self.stockList):
            # 股票数量改变，则每天的数据都需要重新下载
            tradeCal = self.tradeCal
        else:
            # 获取 大于等于log中的endDate的日期 至 最新数据的日期 的切片
            tradeCal = getSliceFromValues(self.tradeCal,
                                          val1=dateNotLessThan(log["endDate"], self.tradeCal),
                                          val2=self.endDate)

        for date in tqdm(tradeCal):
            date = int(date)
            # 获取日行情数据和复权因子，并根据ts_code排序
            self.wait()
            daily = self.pro.daily(**{"trade_date": date, }, fields=self.parameters)
            daily = daily.sort_values(by="ts_code")
            self.wait()
            daily_basic = self.pro.adj_factor(trade_date=date)
            daily_basic = daily_basic.sort_values(by="ts_code")

            # 根据self.stockList筛选股票
            daily = pd.merge(daily, self.stockList, how="right")
            daily_basic = pd.merge(daily_basic, self.stockList, how="right")

            # 计算后复权数据
            for column in self.hfq_columns:
                daily[column] = daily[column] * daily_basic["adj_factor"]
            daily = daily.round(4)

            # 根据前一天的数据填补当天缺失值
            try:
                if not rem.empty:
                    daily[daily.T.isnull().any()] = rem.loc[daily[daily.T.isnull().any()].index.values]
            except:
                e_rem = rem
                e_daily = daily
                e_index = daily[daily.T.isnull().any()].index.values
                from functions import printf
                printf(e_rem, loc=locals())
                printf(e_daily, loc=locals())
                printf(e_index, loc=locals())
                storeAsCsv(e_rem, "e_rem.csv")
                storeAsCsv(e_daily, "e_daily.csv")
                e_index = pd.DataFrame(e_index)
                storeAsCsv(e_index, "e_index.csv")
                raise "Data filling error"

            if date >= self.dataStartDate:
                storeAsCsv(daily, join(self.marketDataStorePath, str(date) + ".csv"))  # e.p. */20100104.csv
            rem = daily.copy(deep=True)
        saveLog(logPath=self.rawDataPath + "storeLog.json",
                dataStartDate=self.dataStartDate, endDate=self.endDate, stockNum=len(self.stockList))

    def storeDailyBasic(self):
        """
        保存每日指标
        暂且没有功能，不知道获取的因子有什么用处
        """
        pass

    def showDownloadInfo(self):
        print("DownLoad Information : ")
        print("    Download Dates : ", len(self.tradeCal), "  Days")
        print("    Stock quantity : ", len(self.stockList), "  Stocks")
        print("    Parameters :     ", len(self.parameters), "  Parameters")

    def storeData(self):  # 在storeStockList修复前不推荐使用
        self.storeTradeCal()
        self.storeStockList()
        self.storeParameters()
        self.storeDailyData()
        self.storeDailyBasic()


