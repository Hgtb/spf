# **StockPriceForecast 股票价格预测**
## **1. 项目目标**
>    通过Tushare平台获取数据，进行数据清洗、数据处理，并训练神经网络，使用训练完成的模型对未来30天的股价进行预测。
## **2. 使用方法**
>   完整的代码可以使用 `main.py` 执行或者使用 `jupyter notebook` 调用模块快速使用。
## **3. 项目结构**
> ### **`dataSet\ ` 目录**
> >目录层级：
> >dataSet：  
> >     * 
> >     *
> >
> ### **`libs\ ` 目录**
> >在`libs\ `目录下有多个`.py`文件，包括`cleanData.py`，`dataLoader.py`，`dataProcess.py`，`funtions.py`，`getData.py`，
> >`module.py`。
> > >### functions.py  
> > >`funtions.py`内为用到的简单工具函数和类，用于简化一些功能的实现。
> >
> > >### getData.py
> > >`getData.py`内为两个版本的`DownloadData`类，用于从Tushare平台获取数据。分别为`DownloadDataBasic`和`DownloadDataPro`。
> > >这两个类有统一的接口类`DownloadDataInterface`。   
> > > ~~DownloadDataBasic~~ 已弃用，未来将只使用 `DownloadDataPro`  
> >
> > >### ~~cleanData.py~~   ***已弃用***
> > >`cleanData.py`中的函数用于`getData.py`文件中的`DownloadDataBasic`类，由`DownloadDataBasic`类获取的数据需要使用这些函数进行清洗。
> >
> > >### dataProcess.py
> > >
> >
> > >### dataLoader.py
> > >读取处理好的最终数据（未归一化），转化为适合模型输入的`shape`的`torch.Tensor`类型，并在模型训练与测试时给出便于使用的数据。
> >
> > >### module.py
> > >

