from rqalpha.api import *
import pandas as pd
import os
import rqdatac as rq
rq.init('license','FTp2uHaRsHQhI9XoFCwKkJNyYdaZ9f6oMHT4kPr1Pq95n2VHHH_KnxinjLrktuwQ1AuW72X5vSrS01MGqOw8OWRD8W3B_EWHNBWGz5vE2A3cLKxEz25vNYeXIbzDbt5v8crTY1OOkjjypfOnfnItH5r95_8C3ck0QmGHgMyMbTw=U_T-NcVYHMdYw9LovsISVLqco69JY8b6093uhl6PUG4S2gEXQOO_RAuTwjbFEK-GXnzutyBAP-s-3JpXpkjdnfQ96Ypjzl6J1DULxSGzqWQo6LWJwVw8YB125DfT5oSHqk9lUsgriFojSQqG92uKh8HKpZK4fVvNhp3Lv2U410c=')  # 初始化

def init(context):
    # 每周调仓，默认每周一开盘前执行
    scheduler.run_weekly(rebalance, tradingday=1)  # 每周一开盘调仓
    # 股票池为A股所有股票
    context.stocks = all_instruments(type='CS', date=None)['order_book_id'].tolist()
    # 选取市值最小的前100只股票
    context.stock_num = 500
    # 预加载factor_data.pkl
    context.factor_data = pd.read_pickle(r"C:\Users\9shao\Desktop\本地化\rqalpha-localization\测试代码\因子数据\multivariate_rolling_120_复合因子_长数据.pkl")
    # 确保order_book_id为字符串
    context.factor_data['order_book_id'] = context.factor_data['order_book_id'].astype(str)

def before_trading(context):
    pass

def handle_bar(context, bar_dict):
    # 打印当前持仓股数、累计收益、当日收益等信息
    portfolio = context.portfolio
    持仓股数 = len([p for p in portfolio.positions.values() if p.quantity > 0])
    # 修正：Portfolio对象没有'returns'属性，使用'total_returns'和'daily_returns'
    累计收益 = getattr(portfolio, "total_returns", None)
    当日收益 = getattr(portfolio, "daily_returns", None)
    总资产 = portfolio.total_value
    可用资金 = portfolio.cash
    # 兼容属性不存在的情况
    if 累计收益 is not None and 当日收益 is not None:
        print(f"日期: {context.now.strftime('%Y-%m-%d')}, 持仓股数: {持仓股数}, 累计收益: {累计收益:.4%}, 当日收益: {当日收益:.4%}, 总资产: {总资产:.2f}, 可用资金: {可用资金:.2f}")
    else:
        print(f"日期: {context.now.strftime('%Y-%m-%d')}, 持仓股数: {持仓股数}, 总资产: {总资产:.2f}, 可用资金: {可用资金:.2f}")

def rebalance(context, bar_dict):
    # 获取当前日期
    current_date = context.now.strftime("%Y-%m-%d")
    # 过滤出当前日期的市值数据
    df = context.factor_data[context.factor_data['date'] == current_date]
    # 过滤涨跌停和ST股票
    df_target = df[(df['limit_up_flag'] == False) & (df['limit_down_flag'] == False) & (df['ST'] == False)]
    
    # 只保留股票池内的股票
    df_target = df_target[df_target['order_book_id'].isin(context.stocks)]

    # 过滤ST、停牌、涨跌停股票
    out_stocks = []
    for obid in df_target['order_book_id']:
        # ST或停牌
        if is_st_stock(obid) or is_suspended(obid):
            out_stocks.append(obid)
            continue
    # 保留df['order_book_id']不在out_stocks的部分
    df_target = df_target[~df_target['order_book_id'].isin(out_stocks)]
    small_cap_stocks = df_target.sort_values('factor', ascending=False).head(context.stock_num)['order_book_id'].tolist()

    # 卖出不在小市值池中的股票
    for stock in list(context.portfolio.positions.keys()):
        if is_suspended(stock):
            continue
        if df[df['order_book_id'] == stock]['limit_down_flag'].values[0]:
            continue
        if stock not in small_cap_stocks:
            order_target_percent(stock, 0)
    # 平均分配资金买入小市值股票
    weight = 1.0 / context.stock_num if context.stock_num > 0 else 0
    for stock in small_cap_stocks:
        order_target_percent(stock, weight)
