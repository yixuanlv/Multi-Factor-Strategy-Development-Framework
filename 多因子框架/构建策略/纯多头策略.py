from rqalpha.api import *
import pandas as pd
import os
import rqdatac as rq
from rqalpha import run_func

rq.init('license', 'B1T4WrPGQ0YBin6JPZm_DlLj3JGxAiuGzi9-SuUNqOUce6MrZ7yLejN2O9OWDPBJ3U6cVO-6uaK8Wn29JTxgNRHrqWJgGHTtf483vOI3bFPOMknL3dEAgQZJTLHJ7LyMZcalsTdqlVOyhT0mlNm_9iNEZBgTxhQW0X_DHOzxLBg=efJUVSAH7ub2Gg33v_nzcsj25LD0caxMdEYz93JazlzGzbk5tG6kxZKqGx9LcGX58GmZmd2IxNETyXC1jdWnnH7u97TeyhpkfP1lfJ5h5sp-1vSqT0tXvJH0SOGoIY4spxMnMeEprvKck7cwl1As3xh8y068HH8MApBtIrP7aOA=')

config_0 = {
    "base": {
        "start_date": "2019-01-01",
        "end_date": "2025-07-22",
        "stock_commission_multiplier": 0.125,
        "frequency": "1d",
        "accounts": {
            "stock": 10000000
        },
        "benchmark": "000300.XSHG"
    },
    "mod": {
        "sys_analyser": {
            "enabled": True,
            "plot": True,
            "output_file": r"中性策略_1.pkl"
        }
    }
}
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
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from rqalpha.api import *
    from rqalpha import run_func
    run_func(
        init=init,
        before_trading=before_trading,
        handle_bar=handle_bar,
        config=config_0
    )