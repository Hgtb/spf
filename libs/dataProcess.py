import pandas as pd
from tqdm import tqdm
from os.path import join
from libs.functions import storeAsCsv, detectFolder


# ToDo(Alex Han) 在完成因子下载函数后将因子与行情数据合并
# ToDo(Alex Han) 从行情数据中获得股票数量最多或最少的日数据的股票列表作为筛选用的股票列表，将数据处理从数据下载中抽取出来


class DataProcess:
    def __init__(self, dataSetPath: str = "../dataSet/"):
        self.rawBasicDataPath = join(dataSetPath, "rawData/basicData/")    #
        self.rawMarketDataPath = join(dataSetPath, "rawData/marketData/")  #

        self.basicDataPath = join(dataSetPath, "data/basicData/")    #
        self.marketDataPath = join(dataSetPath, "data/marketData/")  #

        self.__detectFolders()

        self.stockList = pd.DataFrame([])
        self.tradeCal = pd.DataFrame([])
        self.dailyData = []

    def __detectFolders(self):
        # print(self.rawBasicDataPath)
        # print(self.rawMarketDataPath)
        # print(self.basicDataPath)
        # print(self.marketDataPath)
        detectFolder(self.rawBasicDataPath)
        detectFolder(self.rawMarketDataPath)
        detectFolder(self.basicDataPath)
        detectFolder(self.marketDataPath)

    def __loadTradeCal(self):
        self.tradeCal = pd.read_csv(self.rawBasicDataPath + "tradeCal.csv")

    def __loadStockList(self):
        self.stockList = pd.read_csv(self.rawBasicDataPath + "stockList.csv")

    def __loadDailyData(self):
        for date in tqdm(list(self.tradeCal["trade_date"]), desc="Load daily data"):
            data = pd.read_csv(self.rawMarketDataPath + str(date) + ".csv")
            self.dailyData.append(data)

    def loadData(self):
        self.__loadTradeCal()
        self.__loadStockList()
        self.__loadDailyData()

    def storeData(self):
        storeAsCsv(self.tradeCal, self.basicDataPath + "tradeCal.csv")
        storeAsCsv(self.stockList, self.basicDataPath + "stockList.csv")
        tradeCal = list(self.tradeCal["trade_date"])
        for index in tqdm(range(len(tradeCal)), desc="Store daily data"):
            storeAsCsv(self.dailyData[index], self.marketDataPath + str(tradeCal[index]) + ".csv")

    # 下面是操作函数
    def __dropColumns(self, DataFrame: pd.DataFrame, *ColumnNames) -> pd.DataFrame:
        assert ColumnNames != ()
        if type(ColumnNames[0]) == tuple:
            ColumnNames = ColumnNames[0]
        for col in ColumnNames:
            assert type(col) == str
            DataFrame.drop(columns=col, inplace=True)
        return DataFrame

    def dropColumns(self, *ColumnNames) -> None:  # exp: dropColumns("ts_code", "trade_date")
        assert ColumnNames != ()
        for index in tqdm(range(len(self.dailyData)), desc="Drop columns"):
            self.dailyData[index] = self.__dropColumns(self.dailyData[index], ColumnNames)

    def doSomethingElse(self):

        pass


if __name__ == "__main__":
    dataProcess = DataProcess()
    dataProcess.loadData()
    dataProcess.dropColumns("ts_code", "trade_date")
    dataProcess.storeData()
