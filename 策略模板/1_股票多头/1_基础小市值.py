from rqalpha.api import *
import pandas as pd
import os

def init(context):
    # 每周第一个交易日（周一）早上开盘后立即调仓
    # scheduler.run_monthly(rebalance, tradingday=1)   # 月调仓
    # scheduler.run_weekly(rebalance, tradingday=5)      # 周五收盘调仓
    scheduler.run_daily(rebalance)                   # 日调仓
    # 选取市值最小的前100只股票
    context.stock_num = 100
    # 预加载 外部数据
    context.factor_data = pd.read_pickle( r"C:\Users\14717\Desktop\rq本地化\rqalpha-localization\data\factor_data.pkl")
    # 确保order_book_id为字符串
    context.factor_data['order_book_id'] = context.factor_data['order_book_id'].astype(str)


def handle_bar(context, bar_dict):
    pass

def rebalance(context, bar_dict):
    # 更新当日股票池，股票池为A股所有股票,补充新票
    stocks_poor = all_instruments(type='CS', date=None)
    context.stocks = stocks_poor['order_book_id'].tolist()
    # 获取当前日期
    current_date = context.now.strftime("%Y-%m-%d")
    # 过滤出当前日期的市值数据
    df = context.factor_data[context.factor_data['date'] == current_date]
    # 只保留有补充数据的股票
    df = df[df['order_book_id'].isin(context.stocks)]
    # ====前置过滤，过滤st 退，涨停
    normal_set = set(stocks_poor[stocks_poor['special_type'] == 'Normal']['order_book_id'])  # st与退市与暂停交易
    not_limit_set = set(df[df['close'] != df['limit_up']]['order_book_id'])   # 涨停不买入 ，不合理 get price
    post_filler_keep = normal_set & not_limit_set
    df = df[df['order_book_id'].isin(post_filler_keep)] # 前置过滤
    # ====选股
    target_stocks = set(df.sort_values('market_cap').head(context.stock_num)['order_book_id']) # 目标要买进的股票
    # ====后置过滤
    # 。。。。
    # ==================rebalance交易
    pos_stocks = set(context.portfolio.positions.keys())  # 当前仓位 有的股票
    # 卖旧
    for stock in (pos_stocks-target_stocks):
        print(stock,'开始清零')
        order_target_percent(stock, 0)
    # 调新
    for stock in target_stocks:
        weight = 1.0 / context.stock_num
        print(stock, '开始买入:',weight)
        order_target_percent(stock, weight)


if __name__ == '__main__':
    __config__ = {
        "base": {
            "start_date": "2014-01-01",
            "end_date": "2025-07-22",
            "stock_commission_multiplier": 0.125,
            # "frequency": "1m",
            "accounts": {
                "stock": 10000000
            },
            "benchmark": "000300.XSHG"
        },
        "mod": {
            "sys_analyser": {
                "enabled": True,
                "plot": True ,
                "output_file": r"测试策略_1.pkl"
            }
        }
    }
    from rqalpha import run_func  # 相对导入当前包内的模块
    run_func(init=init,handle_bar=handle_bar,config=__config__)