# **StockPriceForecast 股票价格预测**
## **1. 项目目标**
>    通过Tushare平台获取数据，进行数据清洗、数据处理，并训练神经网络，使用训练完成的模型对未来30天的股价进行预测。
## **2. 使用方法**
>   完整的代码可以使用 `main.py` 执行或者使用 `jupyter notebook` 调用模块快速使用。
## **2. 依赖模块**
>   `tushare`, `numpy`, `pandas`, `xarray`, `dask`, `proplot`, `tqdm`
## **3. 项目结构**
> ### **`dataSet\ ` 目录**
> >###########目录层级  
> > |--- readme.md &emsp;&emsp;&emsp;&emsp;# Help   
> > |--- main.py  
> > |--- main.ipynb  
> > |--- dataSet      &emsp; &emsp; &emsp; &emsp; &emsp; # 存储数据  
> > |&emsp; |--- data &emsp; &emsp; &emsp; &emsp; &emsp; # 存储处理完的数据   
> > |&emsp; |&emsp; |--- basicData  &emsp;&emsp; # 存储股票列表、交易日历   
> > |&emsp; |&emsp; |&emsp; |--- `basicData.json`            
> > |&emsp; |&emsp; |--- marketData&emsp;&emsp;# 存储行情数据（日）  
> > |&emsp; |&emsp;  &emsp; |--- `data.nc`    
> > |&emsp; |--- rawData     &emsp;&emsp;&emsp; &emsp; # 存储原始数据  
> > |&emsp; |&emsp; |--- basicData  &emsp;&emsp; # 存储行情数据（日）  
> > |&emsp; |&emsp; |&emsp; |--- `parameters.csv`   
> > |&emsp; |&emsp; |&emsp; |--- `stockList.csv`   
> > |&emsp; |&emsp; |&emsp; |--- `tradeCal.csv`   
> > |&emsp; |&emsp; |--- marketData&emsp;&emsp;# 存储股票列表、交易日历   
> > |&emsp; |&emsp; &emsp; |--- `20100104.csv`   
> > |&emsp; |&emsp; &emsp; |--- `20100105.csv`   
> > |&emsp; |&emsp; &emsp; |--- `20100106.csv`   
> > |&emsp; |&emsp; &emsp; |--- `...`   
> > |&emsp; |--- `storeLog.json`     # 存储下载记录          
> > |--- forecastData  
> > |--- libs  
> > |&emsp; |--- `__init__.py`   
> > |&emsp; |--- `dataLoader.py` &emsp; # 从dataSet/中加载数据   
> > |&emsp; |--- `dataProcess.py`&emsp;# 处理dataSet/中的数据  
> > |&emsp; |--- `functions.py`&emsp;&emsp;# 工具函数文件  
> > |&emsp; |--- `getData.py`&emsp;&emsp;&emsp;# 从Tushare平台下载数据   
> > |&emsp; |--- `module.py`&emsp;&emsp; &emsp; # 保存torch模型   
> > |&emsp; |--- `quant.py` &emsp; &emsp; &emsp; # 量化   
> > |&emsp; |--- `visualization.py`   # 可视化   
> > |--- models   # 保存已训练的模型   
> >



## **4. 待办**
* DataLoader模块
* 模型训练框架
* 数据可视化框架
* 信号生成框架
* 回测框架（没用）

