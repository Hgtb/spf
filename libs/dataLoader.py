import pandas as pd

from functions import *


cal_path = "../dataSet/tradeCal.csv"
stock_list_path = "../dataSet/stockList.csv"
data_folder_path = "../dataSet/daily/"
log_path = "../dataSet/dailyData/save_log.json"


class DataLoaderInterface:
    """
    DataLoader类用于获取数据，来训练神经网络或数据调取
    """
    def __init__(self):
        self.accumulator = 0
        pass

    def getData(self, TrainDataDates, TargetDataDates):
        pass

    def reset(self):
        self.accumulator = 0


class DataLoaderBasic(DataLoaderInterface):  # Abandon
    def __init__(self, calenderPath=cal_path, stockListPath=stock_list_path, dataFolderPath=data_folder_path):
        super().__init__()
        self.data = []
        self.max_data = []  # 顺序与trade_cal中股票顺序一致
        self.calender = pd.read_csv(calenderPath)
        self.trade_date_list = getTradeDateList(cal_path)
        self.stock_list = getStockList(stockListPath)
        self.data_folder_path = dataFolderPath
        self.trade_days_num = len(self.calender)
        self.stocks_num = len(self.stock_list)

        self.daily_data = []
        self.pt = 0  # 标记trainData输出的最后一天的数据的index

    def __daily_data_init(self):
        for i in range(self.trade_days_num):
            self.daily_data.append(pd.DataFrame([]))

    def loadData(self):
        """
        This function can be merged into __init__() function.
        But for functional clarity, we separate it out.
        :return:
        """
        for stock in tqdm(self.stock_list, desc="Load data"):
            buf = getData(data_folder_path, stock)
            self.max_data.append(buf.max(axis=0))
            self.data.append(buf.div(buf.max(axis=0)))  # normalize

    def __loadDailyDataFromRawData(self):  # return type : torch.Tensor
        """
        very slowly!
        :return:
        """
        self.__daily_data_init()
        for stock in tqdm(range(self.stocks_num), desc="Convert to time data"):
            for date in range(self.trade_days_num):
                self.daily_data[date] = pd.concat([self.daily_data[date], self.data[stock][date:date + 1]],
                                                  ignore_index=True)

    def __loadDailyDataFromFile(self):
        for date in tqdm(self.trade_date_list):
            save_path = "../dataSet/dailyData/" + str(date) + ".csv"
            daily_data = pd.read_csv(save_path)
            self.daily_data.append(daily_data)

    def loadDailyData(self):
        if readLog(log_path) == {}:  # 简单判断，待完善
            self.__loadDailyDataFromRawData()
        else:
            self.__loadDailyDataFromFile()

    def saveDailyData(self):
        # begin, end = readLog(log_path)  # 需加入判别daily_data的columns是否改变等等， 由于目前不需要，所以暂且不加入新功能
        # start = 0
        # # begin, end 为日期， start为index
        # if (begin == self.trade_date_list[0]) & (end < self.trade_date_list[-1]):
        #     start = self.trade_date_list.index(end) + 1
        # if (begin == self.trade_date_list[0]) & (end >= self.trade_date_list[-1]):
        #     start = len(self.trade_date_list) - 1
        # if end < self.trade_date_list[0]:  #
        #     start = 0
        for index, daily_data in enumerate(self.daily_data[:-1]):
            save_path = "../dataSet/dailyData/" + str(self.trade_date_list[index]) + ".csv"
            daily_data.to_csv(save_path, index=None)
        saveLog(log_path, self.trade_date_list[0], self.trade_date_list[-1], self.stocks_num)

    def __getTrainData(self, trainDatesDuration):  # 未完成
        list_tensor = self.daily_data[self.accumulator:self.accumulator + trainDatesDuration]  # [0:360]
        for i in range(trainDatesDuration):
            list_tensor[i] = torch.tensor(list_tensor[i].values).unsqueeze(dim=1)
        return torch.stack(list_tensor, dim=1)  # [360, 1, 3142, 9] 在卷积层中，360为batch，不会卷积

    def __getTestData(self, trainDatesDuration, testDatesDuration):  # 未完成
        """
        返回二维tensor
        返回值为每天收盘价
        :param testDatesDuration:
        :return:
        """
        list_tensor = self.daily_data[trainDatesDuration + self.accumulator:
                                      trainDatesDuration + testDatesDuration + self.accumulator]
        print(self.trade_date_list[trainDatesDuration + self.accumulator],
              self.trade_date_list[trainDatesDuration + testDatesDuration + self.accumulator])
        for i in range(testDatesDuration):
            list_tensor[i] = torch.tensor((list_tensor[i])["close"])
        return torch.stack(list_tensor, dim=0)

    def getData(self, TrainDataDates=360, TargetDataDates=30):
        """
        :param TrainDataDates:
        :param TargetDataDates:
        :return tensor[360, 1, 1571, 9], tensor[1571, -1] or tensor[-1, 1571]:
        """
        self.accumulator += 1
        return self.__getTrainData(TrainDataDates), self.__getTestData(trainDatesDuration=TrainDataDates,
                                                                           testDatesDuration=TargetDataDates)


class DataLoaderPro(DataLoaderInterface):  # 最新的dataloader类
    def __init__(self):
        super(DataLoaderPro, self).__init__()
        self.basicDataPath = "../dataSet/rawData/basicData/"    #
        self.marketDataPath = "../dataSet/rawData/marketData/"  #

        self.stockList = []
        self.tradeCal = []
        self.dailyData = []
        self.dailyMaxData = []

    def __loadTradeCal(self):
        self.tradeCal = list((pd.read_csv(self.basicDataPath + "tradeCal.csv"))["cal_date"])

    def __loadStockList(self):
        self.stockList = list((pd.read_csv(self.basicDataPath + "stockList.csv"))["ts_code"])

    def __loadDailyData(self):
        for date in self.tradeCal:
            data = pd.read_csv(self.marketDataPath + str(date) + ".csv")
            # 用列表存储每日最大数据   type(self.dailyData) = list[pd.DataFrame]
            self.dailyMaxData.append(data.max(axis=0))
            # 用列表存储归一化行情数据   type(self.dailyData) = list[pd.DataFrame]
            self.dailyData.append(data.div(data.max(axis=0)))

    def loadData(self):
        self.__loadTradeCal()
        self.__loadStockList()
        self.__loadDailyData()

    def getData(self, TrainDataDates=360, TargetDataDates=30):

        pass


if __name__ == "__main__":
    data_loader = DataLoaderBasic()
    data_loader.loadData()
    data_loader.loadDailyData()
    data_loader.saveDailyData()
    train_data, test_data = data_loader.getData()
    print(train_data)
