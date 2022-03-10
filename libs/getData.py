from tqdm import tqdm
import tushare as ts
import pandas as pd
from functions import *


class DownloadData:
    def __init__(self):
        self.token = "b54bfb5fc70a78e4962b8c55911b93a0a4ddd4c764115aeee3c301a3"
        ts.set_token(self.token)
        self.pro = ts.pro_api()
        self.wait = FrequencyLimitation()
        self.basicDataSavePath = "../rawData/"
        self.dailyDataSavePath = "../rawData/daily/"
        self.exchangeList = ["SSE", "SZSE"]  # 交易所代码列表

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


if __name__ == "__main__":
    downloadData = DownloadData()
    downloadData.saveData(startDate=20100101)
