import pandas as pd
from tqdm import tqdm
from functions import getStockList, getData

wish_start_date = 20100101
raw_cal_path = "../rawData/tradeCal.csv"
cal_path = "../dataSet/tradeCal.csv"
raw_stockList_SSE = "../rawData/stockList_SSE.csv"
raw_stockList_SZSE = "../rawData/stockList_SZSE.csv"
raw_data_path = "../rawData/daily/"
stock_list_path = "../dataSet/stockList.csv"


def cleanTradeCal(tradeCalenderPath=raw_cal_path):
    trade_cal = pd.read_csv(tradeCalenderPath)
    trade_cal.drop(trade_cal[trade_cal["is_open"] == 0].index, inplace=True)
    trade_cal.drop(columns="is_open", inplace=True)
    trade_cal.to_csv(cal_path, index=None)


def getInFactStartDay(date):  # such as date=20100101 return=20100104
    calender = pd.read_csv(cal_path)
    date_list = list(calender["cal_date"])
    for date in date_list:
        if date >= wish_start_date:
            return date
    raise ValueError("Can't find start date")


start_date = getInFactStartDay(wish_start_date)


def __cleanStockList(stockListPath, tradeDataPath):
    stock_list_pd = pd.read_csv(stockListPath)
    stock_list_pd.drop(columns="Unnamed: 0", inplace=True)
    stock_list_pd.set_index("ts_code", inplace=True)
    stock_list = getStockList(stockListPath)
    for stock in tqdm(stock_list, desc="Clean stock list"):
        buf = getData(raw_data_path, stock)
        if buf.empty:
            stock_list_pd.drop(index=stock, inplace=True)
            continue
        if buf["trade_date"][0] > start_date:
            stock_list_pd.drop(index=stock, inplace=True)
    # stock_list_pd.to_csv("test_stockList.csv")
    return stock_list_pd


def cleanStockList(stockListPath_SZSE, stockListPath_SSE, tradeDataPath):
    stock_list_SZSE = __cleanStockList(stockListPath_SZSE, tradeDataPath)
    stock_list_SSE = __cleanStockList(stockListPath_SSE, tradeDataPath)
    stock_list = stock_list_SZSE.append(stock_list_SSE)
    stock_list_SZSE.to_csv("../dataSet/stockList_SZSE.csv")
    stock_list_SSE.to_csv("../dataSet/stockList_SSE.csv")
    stock_list.to_csv(stock_list_path)


def cleanStockData(stockListPath=stock_list_path, rawDataPath=raw_data_path, outputPath="../dataSet/daily/"):
    stock_list = getStockList(stock_list_path)
    for stock in tqdm(stock_list, desc="Clean stock data"):
        _raw_data_path = rawDataPath + str(stock) + ".csv"
        _output_path = outputPath + str(stock) + ".csv"
        _data = pd.read_csv(_raw_data_path)
        _data.drop(columns="ts_code", inplace=True)
        _data.drop(columns="trade_date", inplace=True)
        _data.to_csv(_output_path, index=None)


if __name__ == "__main__":
    cleanTradeCal()
    cleanStockList(stockListPath_SZSE=raw_stockList_SZSE, stockListPath_SSE=raw_stockList_SSE, tradeDataPath=raw_data_path)
    cleanStockData()