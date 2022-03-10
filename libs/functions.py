import pandas as pd
from tqdm import tqdm
import json
import torch
import os

import time
import timeit


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


def getStockList(stockListPath):
    stock_list = pd.read_csv(stockListPath)
    stock_list = list(stock_list["ts_code"])
    return stock_list


def getTradeDateList(tradeDatePath):
    trade_date_list = pd.read_csv(tradeDatePath)
    trade_date_list = list(trade_date_list["cal_date"])
    return trade_date_list


def getMax(dataFrame):
    return list(dataFrame.max(axis=0))


def saveLog(logPath, begin, end, stockNum):
    import datetime
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
    try:
        log_file = open(logPath, "r")
        save_log = json.loads(log_file.read())
        log_file.close()
        return save_log
    except FileNotFoundError:
        return {}


def detectFolder(path):
    if not os.path.exists(path):
        os.makedirs(path)


class FrequencyLimitation:  # Tushare 调取数据频率上限：1min 500次 每次5000条
    def __init__(self, time_interval=0.12):
        self.dTime = 0.0
        self.timeStamp = 0.0
        self.timeInterval = time_interval  # 单位 ： 1s
        pass

    def __call__(self):  # 在请求前使用
        # self.dTime = timeit.default_timer() - self.timeStamp
        self.timeStamp += self.timeInterval
        # print(self.timeStamp, self.dTime)
        if timeit.default_timer() < self.timeStamp:
            time.sleep(self.timeStamp - timeit.default_timer())
        self.timeStamp = timeit.default_timer()
        pass

