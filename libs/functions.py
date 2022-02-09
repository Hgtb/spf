import pandas as pd
from tqdm import tqdm
import json
import torch

cal_path = "../dataSet/tradeCal.csv"
stock_list_path = "../dataSet/stockList.csv"
data_folder_path = "../dataSet/daily/"
log_path = "../dataSet/dailyData/save_log.json"


def filePath(dataFolderPath, stockName):
    if dataFolderPath[-1] != "/":  # 检测路径最后一项是否有/
        return dataFolderPath + "/" + stockName + ".csv"
    else:
        return dataFolderPath + stockName + ".csv"


def getData(dataFolderPath, stockName):
    """
    :param dataFolderPath:
    :param stockName:
    :return pandas.DataFrame:
    """
    return pd.read_csv(filePath(dataFolderPath, stockName))


def getStockList(stockListPath=stock_list_path):
    stock_list = pd.read_csv(stockListPath)
    stock_list = list(stock_list["ts_code"])
    return stock_list


def getTradeDateList(tradeDatePath=cal_path):
    trade_date_list = pd.read_csv(tradeDatePath)
    trade_date_list = list(trade_date_list["cal_date"])
    return trade_date_list


def getMax(dataFrame):
    return list(dataFrame.max(axis=0))


# def listDataframeToListTensor(listDataframe:list):
#     list_tensor = []
#     for dataframe in listDataframe:
#         list_tensor.append(torch.tensor(dataframe.values))


def saveLog(logPath, begin, end, stockNum):
    import datetime
    # 写入完成后，记录写入范围，用于加速下次loadDailyData和saveDailyData
    # 使用package.json文件保存
    save_log = {"begin": begin,
                "end": end,
                "stock_num": stockNum,
                "save_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 2021-02-10 1:53:03
                }
    save_log = json.dumps(save_log)  # dic -> str
    log_file = open(logPath, "w")
    log_file.write(save_log)
    log_file.close()


def readLog(logPath):
    log_file = open(logPath, "r")
    save_log = json.loads(log_file.read())
    log_file.close()
    return save_log["begin"], save_log["end"]


class DataLoader:
    def __init__(self, calenderPath=cal_path, stockListPath=stock_list_path, dataFolderPath=data_folder_path):
        self.data = []
        self.max_data = []  # 顺序与trade_cal中股票顺序一致
        self.calender = pd.read_csv(calenderPath)
        self.trade_date_list = getTradeDateList()
        self.stock_list = getStockList(stockListPath)
        self.data_folder_path = dataFolderPath
        self.trade_days_num = len(self.calender)
        self.stocks_num = len(self.stock_list)

        self.daily_data = []
        self.__daily_data_init()

        self.accumulator = 0

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
            self.data.append(buf)
            self.max_data.append(buf.max(axis=0))

    def loadDailyData(self, save=False):  # return type : torch.Tensor
        for stock in tqdm(range(self.stocks_num), desc="Convert to time data"):
            for date in range(self.trade_days_num):
                self.daily_data[date] = pd.concat([self.daily_data[date], self.data[stock][date:date + 1]],
                                                  ignore_index=True)

    def saveDailyData(self):
        begin, end = readLog(log_path) # 需加入判别daily_data的columns是否改变等等， 由于目前不需要，所以暂且不加入新功能
        start = 0
        if begin == (self.trade_date_list[0]) & (end < self.trade_date_list[-1]):
            start = self.trade_date_list.index(end) + 1
        for index, daily_data in enumerate(self.daily_data[:-1]):
            save_path = "../dataSet/dailyData/" + str(self.trade_date_list[index]) + ".csv"
            daily_data.to_csv(save_path, index=None)
        saveLog(log_path, self.trade_date_list[0], self.trade_date_list[-1], self.stocks_num)

    def getTrainData(self, dates: int = 360):
        list_tensor = self.daily_data[self.accumulator:self.accumulator + dates]  # [0:360]
        self.accumulator += 1
        for i in range(dates):
            list_tensor[i] = torch.tensor(self.daily_data[i].values).unsqueeze(dim=0)
        return list_tensor

    def getTestData(self, dates: int = 30):

        pass


if __name__ == "__main__":
    data_loader = DataLoader()
    data_loader.loadData()
    data_loader.loadDailyData()
    data_loader.saveDailyData()
    train_data = data_loader.getTrainData()
    print(train_data)
    print(train_data[359].shape)
