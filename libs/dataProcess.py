import pandas as pd


# ToDo(Alex Han) 在完成因子下载函数后将因子与行情数据合并
# ToDo(Alex Han) 从行情数据中获得股票数量最多或最少的日数据的股票列表作为筛选用的股票列表，将数据处理从数据下载中抽取出来


def dropColumns(DataFrame: pd.DataFrame, *ColumnNames) -> pd.DataFrame:
    assert ColumnNames != ()
    if type(ColumnNames[0]) == tuple:
        ColumnNames = ColumnNames[0]
    for col in ColumnNames:
        assert type(col) == str
        DataFrame.drop(columns=col, inplace=True)
    return DataFrame


def dropData(dataList: list, *ColumnNames) -> list:
    """目前没用"""
    assert ColumnNames != ()
    for index in range(len(dataList)):
        dataList[index] = dropColumns(dataList[index], ColumnNames)
    return dataList


