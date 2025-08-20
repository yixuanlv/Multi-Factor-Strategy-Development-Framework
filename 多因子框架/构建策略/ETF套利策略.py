# -*- coding: utf-8 -*-
"""
持续空IF主力策略 - 持续持有IF主力合约的空头头寸
"""

import sys
import os
from datetime import time,date,datetime
import pandas as pd
import numpy as np
# 在导入rqalpha之前，先添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class ETFSpread:
    def __init__(self,symbol="IF",index="000300.XSHG",etf="510300.XSHG",multiplier=300):  
        self.underlying_symbol = symbol
        self.underlying_index = index
        self.etf = etf    
        self.multiplier = multiplier    
        self.basis_20d = []
        self.time_index = 0
        index_df = pd.read_parquet(r'D:/代码mod整合/bundle_mod_data/index_mins.parquet')
        index_df = index_df[index_df['order_book_id']==self.underlying_index]
        self.index_data  = index_df.sort_values(by='datetime')
        self.maturity_dates = []
        #设置仓位比例，这里设置为满仓空头
        self.position_ratio = 1.0
        #记录当前持有的主力合约和上期持有数量（方便平仓）
        self.current_contract = None
        self.last_keep = 0
        #etf交易判断
        self.etf_trade_pending = False
        self.etf_target_value = 0
        #记录过去的基差
        self.basis_20d = np.full(4800, np.nan)

    def show_run_time(self,start_time,end_time,fun):
        run_time = (end_time - start_time).seconds
        print("{} 代码运行时间为:{} 秒".format(fun,run_time))

    def before_trading(self,context): 
        self.time_index = 0
        now = context.now
        self.now_date = date(now.year, now.month, now.day)
        self.curr_day = now.strftime("%Y%m%d") 
        print(self.curr_day)
        self.skip_rest_of_day = False        
        ft_lst = get_future_contracts(self.underlying_symbol)   
        self.contract_id = ft_lst[0]
        maturity_date = instruments(self.contract_id).maturity_date.date()
        if maturity_date not in self.maturity_dates:
            self.maturity_dates.append(maturity_date)
        subscribe(self.contract_id)
        
    def get_basis_and_adjust(self,bar_dict, context):
        try:
            futures_price = bar_dict[self.contract_id].close
            spot_price = self.index_data.loc[self.index_data['datetime'] == context.now, 'close']
            spot_price = spot_price.iloc[0] if not spot_price.empty else None
             # 如果期货价格或现货价格无效，跳过处理并更新基差为NaN
            if pd.isna(futures_price) or pd.isna(spot_price) or spot_price <= 0:
                self.basis_20d = np.append(self.basis_20d[1:], np.nan)  # 将当前基差位置填充为 NaN                
                return None  # 返回 None，表示数据无效，跳过后续处理
            # 计算基差
            basis = futures_price - spot_price
            #更新20日基差数据，保留最新4800条
            self.basis_20d = np.append(self.basis_20d[1:], basis)  # 移除最早的基差并添加新的基差
            # 返回当前计算的基差值
            return basis
        except Exception as e:
            logger.warning(f"获取 {self.contract_id} 或 {self.underlying_index} 价格失败: {e}")
            return None  # 返回 None，表示计算失败

    def calculate_recent_20d_avg_std(self):
        """计算最近20条基差数据的平均值和标准差"""
        # 计算有效值的数量
        valid_values = self.basis_20d[~np.isnan(self.basis_20d)]  # 过滤掉 NaN 值
        valid_count = len(valid_values)    
        if valid_count < 20:
            #如果有效值少于20个，返回默认值或 None
            #logger.warning(f"有效基差值不足20个，当前有效值数量为: {valid_count}")
            return None, None  # 或者返回默认值如 (0, 0)
        # 计算平均值
        mean_20d = np.nanmean(self.basis_20d)  
        # 计算标准差
        std_20d = np.nanstd(self.basis_20d) 
        return mean_20d, std_20d

    def handle_bar(self,context, bar_dict):
        self.time_index+=1
        #portfolio = context.portfolio
        #logger.info(portfolio.total_value)       
        basis = self.get_basis_and_adjust(bar_dict, context)
        if self.etf_trade_pending:
            actual_value = context.stock_account.positions.get(self.etf, None)
            actual_value = actual_value.market_value if actual_value else 0
            try:
                if self.etf_target_value_future_id == self.contract_id:
                    futures_price = bar_dict[self.etf_target_value_future_id].close
                    self.etf_target_value =  self.etf_target_value_future * futures_price * self.multiplier
                else:
                    self.etf_target_value = 0
                # 判断是否已经接近目标仓位
                if abs(actual_value - self.etf_target_value) /(actual_value + 10000) <= 0.002:
                    #logger.info(f"✅ ETF调仓完成：目标={context.etf_target_value:.2f}, 实际={actual_value:.2f}")
                    self.etf_trade_pending = False
                    self.etf_target_value = 0
                elif (actual_value < self.etf_target_value) and (self.now_date not in self.maturity_dates):
                    #logger.info(f"📌 检测到未完成ETF调仓，再次尝试买入下单。目标={context.etf_target_value:.2f}, 当前={actual_value:.2f}")
                    order_target_value(self.etf, self.etf_target_value)
                    self.skip_rest_of_day = True                
                elif actual_value > self.etf_target_value:
                    #logger.info(f"📌 检测到未完成ETF调仓，再次尝试卖出下单。目标={context.etf_target_value:.2f}, 当前={actual_value:.2f}")
                    order_target_value(self.etf, self.etf_target_value)
                    self.etf_trade_pending = True
            except:
                order_target_value(self.etf,0)
                self.etf_trade_pending = True
                self.etf_target_value = 0
        if self.skip_rest_of_day:
            #logger.info("今天买过了")
            return
        if (self.now_date in self.maturity_dates) and (context.now.time() == time(15, 0)):
            #logger.info(f"📆 今天是交割日 {now_date} 且当前时间为 15:00，清仓所有 ETF")
            for self.etf, position in context.stock_account.positions.items():
                order_target_value(self.etf, 0)
                self.etf_trade_pending = True
                self.etf_target_value_future = 0
            # 清除这一天，避免重复清仓
            self.maturity_dates.remove(self.now_date)
            a = float(context.stock_account.total_value)
            b = float(context.future_account.total_value)
            target_total_value_each = int((a+b) / 2)
            if a <= b:
                deposit(account_type="STOCK",amount=target_total_value_each-a)
                withdraw(account_type="FUTURE",amount=b - target_total_value_each)
            else:
                withdraw(account_type="STOCK",amount=a- target_total_value_each)
                deposit(account_type="FUTURE",amount=target_total_value_each-b)

        # 判断合约是否需要交易
        futures_price = bar_dict[self.contract_id].close      
        position = context.future_account.positions.get(self.contract_id)
        current_qty = position.sell_quantity if position else 0
        mean_20d, std_20d = self.calculate_recent_20d_avg_std()
        if mean_20d is None or std_20d is None or basis is None:
            #logger.warning("基差均值或标准差为 None，跳过交易判断")
            return
        if basis < mean_20d -std_20d:
            # 如果要开空，并且当前没有持仓，就卖出（开空）
            if (current_qty == 0) and (self.now_date not in self.maturity_dates):
                #logger.info(f"🟥 卖出期货 {context.contract_id}，进行开空操作")
                keep = context.future_account.total_value// (futures_price * self.multiplier)
                sell_open(self.contract_id, keep)  
                order_target_value(self.etf,keep*futures_price * self.multiplier)
                self.etf_trade_pending = True
                self.etf_target_value_future = keep
                self.etf_target_value_future_id = self.contract_id    
                self.last_keep = keep
                self.skip_rest_of_day = True   
            else:
                #logger.info(f"📌 已持有 {context.contract_id} 的仓位或今天是交割日，跳过开空")
                pass                 
        elif  basis >= mean_20d + std_20d :
            # 如果要买入（平空），且当前持有空仓，就平仓
            if current_qty > 0:
                #logger.info(f"✅ 平仓期货 {self.contract_id}，买入平空")
                logger.info(self.last_keep)
                buy_close(self.contract_id,self.last_keep) 
                order_target_value(self.etf,0) # 平仓
                self.etf_trade_pending = True
                self.etf_target_value_future = 0
                self.etf_target_value_future_id = self.contract_id
                a = float(context.stock_account.total_value)
                b = float(context.future_account.total_value)
                target_total_value_each = int((a+b) / 2)
                if a <= b:
                    deposit(account_type="STOCK",amount=target_total_value_each-a)
                    withdraw(account_type="FUTURE",amount=b - target_total_value_each)
                else:
                    withdraw(account_type="STOCK",amount=a- target_total_value_each)
                    deposit(account_type="FUTURE",amount=target_total_value_each-b)
            else:
                pass
                #logger.info(f"📌 当前没有空仓，跳过平仓")

def init(context):
    context.stg1 = ETFSpread("IF","000300.XSHG","510300.XSHG",300) 

def before_trading(context):
    if context.stg1:  
        context.stg1.before_trading(context)

def handle_bar(context, bar_dict):
    if context.stg1:  
        context.stg1.handle_bar(context,bar_dict)
    
def after_trading(context):
    if context.stg1:  
        context.stg1.after_trading(context)

if __name__ == "__main__":
    __config__ = {
        "base": {
            "frequency": "1m",
            "start_date": "2023-08-15",
            "end_date": "2024-08-18",
            #"end_date": "2022-02-10",
            "accounts": {"future": 200_0000,
                         "stock": 200_0000},
            "benchmark": "000300.XSHG",
            # "benchmark": None,
        },
        "mod": {
            "sys_analyser": {
                "enabled": True,
                "plot": True,
                "benchmark": None,
                "output_file": r"D:/代码mod整合/rqalpha/examples/中性策略1.pkl",
                "plot_save_file": "持续空IF主力_回测结果.png"
            },
            "merged_minute_mod": {
                "enabled": True,
                "lib": "rqalpha.mod.rqalpha_mod_merged_minute",
                "ft_minute_mod": {
                    "parquet_path": r"D:/代码mod整合/bundle_mod_data/ft_minute.parquet"
                },
                "etf_minute_mod": {
                    "parquet_path": r"D:/代码mod整合/bundle_mod_data/etf_mins.parquet"

                },
            
            }
        }
    }
    from rqalpha import run_func
    run_func(init=init,handle_bar=handle_bar,config=__config__,before_trading = before_trading)