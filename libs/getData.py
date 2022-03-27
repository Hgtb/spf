from tqdm import tqdm
import tushare as ts
import pandas as pd
from functions import *


def dateNoGreaterThan(date: int, tradeCal: list) -> int:
    for index in range(len(tradeCal) - 1, -1, -1):  # index = 3, 2, 1, 0
        if int(tradeCal[index]) <= date:
            return int(tradeCal[index])


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


class DownloadDataBasic(DownloadDataInterface):  # Abandon
    def __init__(self):
        super().__init__()
        self.token = "b54bfb5fc70a78e4962b8c55911b93a0a4ddd4c764115aeee3c301a3"
        ts.set_token(self.token)
        self.pro = ts.pro_api()
        self.wait = FrequencyLimitation()
        self.exchangeList = ["SSE", "SZSE"]  # 交易所代码列表
        self.dailyDataSavePath = "../dataSet/dailyData/"
        self.basicDataSavePath = "../rawData/"
        detectFolder(self.dailyDataSavePath)

    def saveTradeCal(self, startDate=20100101):
        """
        获取深圳交易所的tradeCal
        (深交所和上交所的tradeCal应当一样)
        :return:
        """
        for index in tqdm(range(1), desc="Save TradeCal"):
            self.wait()
            df_SZSE = self.pro.trade_cal(**{
                "exchange": "SSE",
                "cal_date": "",
                "start_date": startDate,
                "end_date": "",
                "is_open": "",
                "limit": "",
                "offset": ""
            }, fields=["cal_date", "is_open"])
            df_SZSE.to_csv(self.basicDataSavePath + "tradeCal.csv", index=None)

    def saveStockList(self):
        """
        分别获取深交所和上交所当前的股票列表
        :return:
        """
        for exchange in tqdm(self.exchangeList, desc="Save StockList"):
            self.wait()
            _stockList = self.pro.stock_basic(**{
                "ts_code": "",
                "name": "",
                "exchange": exchange,
                "market": "",
                "is_hs": "",
                "list_status": "L",
                "limit": 5000,
                "offset": ""
            }, fields=[
                "ts_code",
                "symbol",
                "name",
                "industry",
                "market",
                "list_date",
                "exchange"])
            _stockList.to_csv(self.basicDataSavePath + "stockList_" + exchange + ".csv", index=None)

    def saveDailyData(self, StockList=None):
        """
        分别获取深交所和上交所的如数据
        如果没有输入StockList则根据saveStockList()获取的股票列表下载数据
        如果输入StockList则根据StockList获取股票数据
        :param StockList:
        :return:
        """
        if StockList is None:
            StockList = []
            for exchange in self.exchangeList:
                df = pd.read_csv(self.basicDataSavePath + "stockList_" + exchange + ".csv")
                df = list(df["ts_code"])
                StockList.extend(df)
        assert type(StockList) == list, "StockList type error"
        for index in tqdm(range(len(StockList)), desc="Save DailyData"):  # 不直接遍历列表是为了tqdm进度条正常显示
            stock_name = StockList[index]
            self.wait()
            file_path = self.dailyDataSavePath + stock_name + ".csv"
            df_daily = self.pro.daily(**{
                "ts_code": stock_name,
                "trade_date": "",
                "start_date": 20100101,
                "end_date": "",
                "offset": "",
                "limit": ""
            }, fields=[
                "ts_code",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "pre_close",
                "change",
                "pct_chg",
                "vol",
                "amount"
            ])
            df_daily = df_daily.iloc[:: -1]  # 反序
            df_daily.to_csv(file_path, index=None)

    def saveData(self, startDate=20100101):
        log = readLog("../rawData/save_log.json")
        if log == {}:
            self.saveTradeCal(startDate)
            self.saveStockList()
            self.saveDailyData()
        else:
            print("Data already saved")
            pass


class DownloadDataPro(DownloadDataInterface):
    def __init__(self, Token):
        """
        需要Tushare账号积分至少600
        调用复权行情,每日指标
        在没有特殊要求时，可以直接调用storeData函数获取数据
        """
        super().__init__()
        ts.set_token(Token)
        self.pro = ts.pro_api()
        self.wait = FrequencyLimitation()

        self.rawDataPath = "../dataSet/rawData/"
        self.basicDataStoreFolder = "../dataSet/rawData/basicData/"
        self.marketDataStorePath = "../dataSet/rawData/marketData/"
        self.__initFolder()

        self.stockList = pd.DataFrame([])  # 从第一个获取的dailyData中获取，用于统一所有数据的股票及股票顺序
        self.startDate = 20100104
        self.endDate = 0
        self.tradeCal = []
        self.hfq_columns = ["open", "high", "low", "close", "pre_close", "change"]
        self.detectFolder()

    def __initFolder(self):
        detectFolder(self.basicDataStoreFolder)
        detectFolder(self.marketDataStorePath)

    def setStartDate(self, startDate):
        self.startDate = startDate  # need a check

    def detectFolder(self):
        detectFolder(self.basicDataStoreFolder)
        detectFolder(self.marketDataStorePath)

    def setStorePath(self, BasicDataStoreFolder, MarketDataStorePath):
        self.basicDataStoreFolder = BasicDataStoreFolder
        self.marketDataStorePath = MarketDataStorePath
        self.detectFolder()

    def storeTradeCal(self):
        self.wait()
        self.tradeCal = self.pro.trade_cal(**{"start_date": self.startDate, "is_open": 1}, fields=["cal_date"])
        self.tradeCal.set_axis(["trade_date"], axis='columns', inplace=True)
        storeAsCsv(self.tradeCal, self.basicDataStoreFolder + "tradeCal.csv")
        self.tradeCal = list(self.tradeCal["trade_date"])
        self.startDate = dateNotLessThan(self.startDate, self.tradeCal)

    def storeStockList(self):
        self.wait()
        # 可使用 pro.stock_basic(fields=["ts_code"]) 获取最新所有股票的列表，但是未来股票数量会超过单次数据调取上限
        latestStockList = pd.concat([self.pro.stock_basic(**{"exchange": "SZSE"}, fields=["ts_code"]),
                                     self.pro.stock_basic(**{"exchange": "SSE"}, fields=["ts_code"])],
                                    ignore_index=True)
        self.wait()
        daily = self.pro.daily(**{"trade_date": self.startDate, }, fields=["ts_code"])

        # 取开始日期的股票列表与最新股票列表的交集，删去退市股票
        stockList = pd.merge(latestStockList, daily["ts_code"], how="inner")["ts_code"]  # type(stockList) = pd.Series
        stockList = stockList.sort_values()
        stockList = stockList.reset_index(drop=True)
        storeAsCsv(stockList, "../dataSet/rawData/basicData/stockList.csv")
        self.stockList = stockList

    def storeDailyData(self):
        """
        保存后复权股票日行情数据
        """
        if self.stockList.empty:
            self.storeStockList()
        self.endDate = getLatestDataDate(self.tradeCal)
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
            # 获取日行情数据和复权因子，并根据ts_code排序
            self.wait()
            daily = self.pro.daily(**{"trade_date": date, }, fields=[
                "ts_code", "trade_date", "open", "high", "low", "close",
                "pre_close", "change", "pct_chg", "vol", "amount"])
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

            storeAsCsv(daily, "../dataSet/rawData/marketData/" + str(date) + ".csv")
            rem = daily.copy(deep=True)
        saveLog(logPath=self.rawDataPath + "storeLog.json",
                startDate=self.startDate, endDate=self.endDate, stockNum=len(self.stockList))

    def storeDailyBasic(self):
        """
        保存每日指标
        暂且没有功能，不知道获取的因子有什么用处
        """
        pass

    def storeData(self):
        self.storeTradeCal()
        self.storeStockList()
        self.storeDailyData()
        self.storeDailyBasic()


if __name__ == "__main__":
    downloadData = DownloadDataPro("b54bfb5fc70a78e4962b8c55911b93a0a4ddd4c764115aeee3c301a3")
    downloadData.storeData()
