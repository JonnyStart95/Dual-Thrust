# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt

# N为我们考虑历史交易的天数；K1,K2为Dual_Thrust的两个可调参数; freq为交易的频率('D'以天为交易单位，'min'为分钟)
class Dual_Thrust(object):
    def __init__(self, N, K1, K2, Freq):
        self.n = N
        self.k1 = K1
        self.k2 = K2
        self.freq = Freq
        self.holding = 0
        self.trading_times = 0
        self.profit = []
        
    def Input(self, Data, Cap,Tran_fee):
    # Data为交易数据; Cap为初始资金
        self.pre_data = Data
        self.cap = 0
        self.init_cap = Cap
        self.cash = Cap
        self.fee = Tran_fee
        self.Data_Process()
        
        
    def Data_Process(self):
        # 分别对于开盘价、收盘价、最高价、最低价和成交量进行处理
        self.pre_data.index = pd.to_datetime(self.pre_data.index)
        self.data = self.pre_data.resample(self.freq).last()
        
        self.data['open'] = self.pre_data['开盘价'].resample(self.freq).first()
        self.data['close'] = self.pre_data['收盘价'].resample(self.freq).last()
        # 处理最高价和最低价
        self.data['high'] = self.pre_data['最高价'].resample(self.freq).max()
        self.data['low'] = self.pre_data['最低价'].resample(self.freq).min()
        # 成交额和成交笔数的转化
        self.data['volume'] = self.pre_data['成交额'].resample(self.freq).sum()
        self.data['transaction_numbers'] = self.pre_data['成交笔数'].resample(self.freq).sum()
        
        # 去除数据集中的无效数据
        self.data.dropna(axis=0)
        self.data = self.data.drop(self.data.columns[[1,2,3,4,5,6,7]], axis=1)
        self.data['Date'] = self.data.index
        #self.data = self.data.rename(columns={'时间': 'Date'})
        
    def parameters_cal(self):
        # 参数计算：N日High的最高价HH, N日Close的最低价LC，N日Close的最高价HC，N日Low的最低价LL
        # Range = Max(HH-LC,HC-LL) 用来描述震荡区间的大小
        self.data['HH'] = self.data['high'].rolling(window=self.n).max().shift(1).fillna(0)
        self.data['LC'] = self.data['close'].rolling(window=self.n).min().shift(1).fillna(0)
        self.data['HC'] = self.data['close'].rolling(window=self.n).max().shift(1).fillna(0)
        self.data['LL'] = self.data['low'].rolling(window=self.n).min().shift(1).fillna(0)
        self.data['Range'] = np.where((self.data['HH']-self.data['LC']) > (self.data['HC']-self.data['LL']),
                 self.data['HH']-self.data['LC'], self.data['HC']-self.data['LL'])
        
        # 计算当时交易参考线 BuyLine 和 SellLine
        # BuyLine = Open + K1*Range
        # SellLine = Open - K2*Range
        self.data['BuyLine'] = self.data['open'] + self.k1 * self.data['Range']
        self.data['SellLine'] = self.data['open'] - self.k1 * self.data['Range']
        
        # 移除前N行没有参考历史数据的信号
        self.data = self.data.iloc[self.n:]
        
    def trading_signal(self):
        # 当价格向上突破上轨时，Long_signal=1
        # 当价格向下突破下轨时, Short_signal=1
        self.data['Long_signal'] = 0
        self.data['Short_signal'] = 0
        self.data['Long_signal'] = np.where(self.data['high'] > self.data['BuyLine'], 1, 0)
        self.data['Short_signal'] = np.where(self.data['low'] < self.data['SellLine'], 1, 0)
        self.data['Trading_action'] = 0
        
    def buy(self,row):
        # 发出交易动作，BTC按照当前交易价格结算，清空现金
        # 减去吃单扣除手续费
        self.cash = self.cash - self.cash * self.fee
        self.profit.append(self.cash)
        self.holding = self.cash/self.data.iloc[row,self.data.columns.get_loc('close')]
        self.cash = 0
        self.data.iloc[row,self.data.columns.get_loc('Trading_action')] = 1
        print('Trading Date is: %s' % self.data.iloc[row,self.data.columns.get_loc('Date')])
        print('Buying, account status after trading is: Cash:%s, BTC holding:%s' % (self.cash, self.holding))
        
    def sell(self,row):
        # 发出卖出动作，按照当前low价格结算为对应BTC
        self.cash += self.holding * self.data.iloc[row,self.data.columns.get_loc('close')]
        # 减去挂单扣除手续费
        self.cash = self.cash - self.cash * self.fee
        self.holding = 0
        self.trading_times += 1
        self.profit.append(self.cash)
        self.data.iloc[row,self.data.columns.get_loc('Trading_action')] = 1
        print('Trading Date is: %s' % self.data.iloc[row,self.data.columns.get_loc('Date')])
        print('Selling, account status after trading is: Cash:%s, BTC holding:%s' % (self.cash, self.holding))        
        
    def update_cap(self,row):
        #计算当前资本
        self.data.iloc[row,self.data.columns.get_loc('cash')] = self.cash
        self.data.iloc[row,self.data.columns.get_loc('BTC_quantity')] = self.holding
        self.data.iloc[row,self.data.columns.get_loc('Capital')] = self.cash + self.holding * self.data.iloc[row, self.data.columns.get_loc('close')]
        self.data.iloc[row,self.data.columns.get_loc('BenchMark')] = (self.init_cap / self.data.iloc[0, self.data.columns.get_loc('open')] * self.data.iloc[row, self.data.columns.get_loc('close')] - self.init_cap) / self.init_cap 
        #print('Total Capital now is %s' % self.data.iloc[row,self.data.columns.get_loc('Capital')])
        
    def show_return(self):
        # 画出收益率曲线
        fig = plt.figure(figsize=(12,8))   
        fig.suptitle('Arbitrage Performance Curve 2020', fontsize=20)
        plt.xlabel('Date', fontsize=10)
        plt.ylabel('Yield', fontsize=10)
        plt.plot(self.data['yield'],'#1f77b4',label = 'Dual_Thrust')
        plt.plot(self.data['BenchMark'],'#FF8C00', label = 'BenchMark')  
        plt.legend(loc='upper left')
        fig.savefig('Yield_Curve_N_%s_K1_%s_K2_%s_Freq_%s.jpg' % (self.n, self.k1, self.k2, self.freq))
        
    def trading(self):
        self.parameters_cal()
        self.trading_signal()
        self.data['cash'] = 0
        self.data['BTC_quantity'] = 0
        self.data['Capital'] = 0
        self.data['BenchMark'] = 0
        for i in range(0,len(self.data)):
            if self.data.iloc[i,self.data.columns.get_loc('Long_signal')] == 1 & self.data.iloc[i,self.data.columns.get_loc('Short_signal')] == 1:
                if self.holding:
                    self.sell(i)
            elif self.data.iloc[i,self.data.columns.get_loc('Long_signal')] == 1:
                if not self.holding:
                    self.buy(i)
            elif self.data.iloc[i,self.data.columns.get_loc('Short_signal')] == 1:
                if self.holding:
                    self.sell(i)
            else:
                pass
            self.update_cap(i)
            
        self.data['yield'] = (self.data['Capital'] - self.init_cap)/ self.init_cap
        #self.show_return()
        
    def MaxDrawdown(self,return_list):
        # 计算虽大回撤率
        capital = return_list
        highwatermarks = capital.cummax()
        drawdowns = 1 - (1 + capital) / (1 + highwatermarks)
        max_drawdown = max(drawdowns)

        return max_drawdown
    
    def Return(self):
        
        return self.data['yield'][-1]

    def evaluate_stats(self):
        #计算Sharp Ratio,这里我们将无风险利率按照国债利率设置为4%
        # Sharp Ratio = (E(Yield) - rf)/sigma
        sharp_mean = np.mean(self.data['yield'])
        sharp_std = np.std(self.data['yield']) # Standard Deviation of returns 
        
        Sharp_Ratio = (sharp_mean - 0.04) / sharp_std
        
        #时间段内波动率计算 sigma_T = sigma * T^1/2
        
        sigma_T  = sharp_std * (len(self.data['yield']) ** 0.5)      
        
        # 最大回撤率
        Max_drawdown = self.MaxDrawdown(self.data['Capital']) 
        
        # 年/月总回报率统计
        print("Sharp率为 :%.6s，波动率为: %.4s，最大回撤为 :%.5s%%，总收益率为 :%.5s%%" % (Sharp_Ratio, sigma_T, Max_drawdown * 100, self.Return() * 100))
        
        # 计算交易频次
        Trading_freq = sum(self.data['Trading_action']) / len(self.data)
        Trading_avg_profit = (self.Return() * self.init_cap) / sum(self.data['Trading_action'])
        self.profit = [x - self.init_cap for x in self.profit]
        print("交易次频率为: %.8s, 每笔交易的平均盈利/亏损为: %.6s, 其中最大盈利为：%.7s, 最大亏损为: %.7s。" % (Trading_freq, Trading_avg_profit, np.nanmax(self.profit), np.nanmin(self.profit)))        
        
        # 交易信号准确率
        self.data['Trading_correctness'] = (self.data['Long_signal'] - self.data['Short_signal']) * (self.data['close'].shift(-1) - self.data['close'])
        self.data['Trading_correctness']= self.data['Trading_correctness'].fillna(0)
        print("Dual Thrust信号判断的正确率为: %.5s%%。" % (100 * sum(self.data['Trading_correctness'].gt(0))/sum(self.data['Trading_correctness']!=0)))        

        
    def return_data(self):
        
        return self.data
    
    def return_profit(self):

        return self.profit