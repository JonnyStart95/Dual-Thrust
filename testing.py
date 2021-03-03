# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 14:14:13 2021

@author: w
"""
import pandas as pd
import numpy as np
from DualThrust import Dual_Thrust

# 读取训练数据集
#train_data = pd.read_csv('文件1.样本训练数据.csv',encoding='gbk')
#val_data = pd.read_csv('文件2.回测数据.csv',encoding='gbk')

data_2018 = pd.read_csv('C:/Users/w/Dropbox/BTC_Trading/hubo_btcusdt_2018.csv', index_col=1, header = None, names = ['ID', '开盘价', '收盘价', '最高价', '最低价','成交量', '成交额', '成交笔数'])
data_2019 = pd.read_csv('C:/Users/w/Dropbox/BTC_Trading/huobi_btcusdt_2019.csv', index_col=1, header = None, names = ['ID', '开盘价', '收盘价', '最高价', '最低价','成交量', '成交额', '成交笔数'])
data_2020 = pd.read_csv('C:/Users/w/Dropbox/BTC_Trading/huobi_btcusdt_2020.csv', index_col=1, header = None, names = ['ID', '开盘价', '收盘价', '最高价', '最低价','成交量', '成交额', '成交笔数'])
data_2021 = pd.read_csv('C:/Users/w/Dropbox/BTC_Trading/huobi_btcusdt_2021.csv', index_col=1, header = None, names = ['ID', '开盘价', '收盘价', '最高价', '最低价','成交量', '成交额', '成交笔数'])
train_data = [data_2018,data_2020]

test_data = data_2019.append(data_2020)

full_data = data_2018.append(data_2019)
full_data = full_data.append(data_2020)
full_data = full_data.append(data_2021)

#**----------------------------------调试策略参数-------------------------------**
# k1 should be much bigger if you are bearish on the market.
        
params_testing_2018_hourly = {}
params_testing_2019_hourly = {}
params_testing_2020_hourly = {}
params_testing = [params_testing_2018_hourly,params_testing_2020_hourly]

n_list = list(range(25,32))

k1_list = [0.1,0.2,0.3,0.4,0.5,1.0,1.5,2.0,2.5]
k2_list = [1.5]

for k in range(2):
    for n in n_list:
        for i in k1_list:
            for j in k2_list:
                dual = Dual_Thrust(N=n, K1=i, K2=j, Freq='4H')
                dual.Input(Data = train_data[k], Cap = 10000, Tran_fee = 0.001)
                dual.trading()
                #dual.evaluate_stats()
                params_testing[k][str([n,i,j])] = dual.Return()
                print('One test finished')
              
# Transfer dictionary into dataframes           
df_params_testing_2018_hourly = pd.DataFrame(list(params_testing_2018_hourly.items()),columns = ['param_settings','return']) 
#df_params_testing_2019_hourly = pd.DataFrame(list(params_testing_2019_hourly.items()),columns = ['param_settings','return']) 
df_params_testing_2020_hourly = pd.DataFrame(list(params_testing_2020_hourly.items()),columns = ['param_settings','return']) 

# Export dataframes into csv files
df_params_testing_2018_hourly.to_csv('params_testing_2020_4hourly_N_2_to_7.csv', index=False)
#df_params_testing_2019_hourly.to_csv('params_testing_2019_hourly.csv', index=False)
df_params_testing_2020_hourly.to_csv('params_testing_2018_4hourly_N_2_to_7.csv', index=False)

# Test on 2019 data

#df_params_testing_2018 = pd.read_csv('params_testing_2018.csv')
#df_params_testing_2019 = pd.read_csv('params_testing_2019.csv')
#df_params_testing_2020 = pd.read_csv('params_testing_2020.csv')
#
## The best params setting for 2018 is N=2, k1=1.2, k2=0.01~1.5
#df_params_testing_2018[df_params_testing_2018['return']==df_params_testing_2018['return'].max()]
#
## The best params setting for 2019 is N=8, k1=0.8, k2=0.01~1.5
#df_params_testing_2019[df_params_testing_2019['return']==df_params_testing_2019['return'].max()]
#
## The best params setting for 2020 is N=17~20, k1=0.7, k2=0.01~1.5
#df_params_testing_2020[df_params_testing_2020['return']==df_params_testing_2020['return'].max()]
#

dual = Dual_Thrust(N=3, K1=3, K2=1.5, Freq='4H')
dual.Input(Data = data_2018, Cap = 10000, Tran_fee = 0.001)
dual.trading()
dual.evaluate_stats()
dual.show_return()
sample = dual.return_data()
# sample.tail()

start = 333
end = 364
print(sample[start:end]['Short_signal'] * sample[start:end]['Trading_action'])
print('Transaction No. is: %s ' % (sum(sample[start:end]['Short_signal'] * sample[start:end]['Trading_action'])))



dual.MaxDrawdown(sample['BenchMark'])

sharp_mean = np.mean(sample['BenchMark'])
sharp_std = np.std(sample['BenchMark']) # Standard Deviation of returns 
Sharp_Ratio = (sharp_mean - 0.04) / sharp_std
print(Sharp_Ratio)





# Get maximum value from a dictionary
# 1Day
print(max(params_testing_2018_hourly, key=params_testing_2018_hourly.get))
print(params_testing_2018_hourly[max(params_testing_2018_hourly, key=params_testing_2018_hourly.get)])
print(max(params_testing_2020_hourly, key=params_testing_2020_hourly.get))
print(params_testing_2020_hourly[max(params_testing_2020_hourly, key=params_testing_2020_hourly.get)])
# 1Day
# 2018: [2, 2.0, 1.5] = 0.34983909634674565
# 2019: [2, 0.5, 1.5] = 2.501846072095831
# 4hour
# 2018: [4, 2.5, 1.5] = 0.5730103232646645
# 2020: [3, 3.0, 1.5] = 3.0603314583624837
