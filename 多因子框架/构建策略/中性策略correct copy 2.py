from rqalpha.api import *
import pandas as pd
import os
import rqdatac as rq
from rqalpha import run_func

rq.init('license', 'B1T4WrPGQ0YBin6JPZm_DlLj3JGxAiuGzi9-SuUNqOUce6MrZ7yLejN2O9OWDPBJ3U6cVO-6uaK8Wn29JTxgNRHrqWJgGHTtf483vOI3bFPOMknL3dEAgQZJTLHJ7LyMZcalsTdqlVOyhT0mlNm_9iNEZBgTxhQW0X_DHOzxLBg=efJUVSAH7ub2Gg33v_nzcsj25LD0caxMdEYz93JazlzGzbk5tG6kxZKqGx9LcGX58GmZmd2IxNETyXC1jdWnnH7u97TeyhpkfP1lfJ5h5sp-1vSqT0tXvJH0SOGoIY4spxMnMeEprvKck7cwl1As3xh8y068HH8MApBtIrP7aOA=')

config_0 = {
    "base": {
        "start_date": "2015-01-01",
        "end_date": "2025-07-22",
        "stock_commission_multiplier": 0.125,
        "frequency": "1d",
        "accounts": {
            "stock": 10000000,
            "future": 10000000
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
    scheduler.run_weekly(rebalance1, tradingday=1)  # 每周一开盘调仓
    # 股票池为A股所有股票
    context.stocks = all_instruments(type='CS', date=None)['order_book_id'].tolist()
    # context.stock_num 不再在init中设置，改为在rebalance中动态计算
    # 预加载复合因子数据
    context.factor_data = pd.read_pickle(r"C:\Users\9shao\Desktop\github公开项目\Multi-Factor-Strategy-Development-Framework\多因子框架\因子库\multivariate_rolling_120_复合因子.pkl")
    # 预加载行情数据用于过滤
    context.market_data = pd.read_pickle(r"C:\Users\9shao\Desktop\github公开项目\Multi-Factor-Strategy-Development-Framework\多因子框架\行情数据库\data.pkl")
    

def before_trading(context):
    pass

def handle_bar(context, bar_dict):
    # 打印当前持仓股数、累计收益、当日收益等信息
    portfolio = context.portfolio
    # 修正：期货持仓使用long_quantity，股票持仓使用quantity
    持仓股数 = len([p for p in portfolio.positions.values() 
                if hasattr(p, 'quantity') and p.quantity > 0 or 
                   hasattr(p, 'long_quantity') and p.long_quantity > 0])
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

def stock_available():
    pass

def select_future_type(type_list,today,next_month):
    spread_dict = {}
    for contract_prefix in type_list:
        far_contract = f"{contract_prefix}{next_month}"
        try:
            basis_df = rq.futures.get_basis(far_contract,today, today)
            if basis_df.empty:
                continue
            spread = basis_df["basis"].iloc[0]
            price = basis_df["close"].iloc[0]
            if pd.isna(spread) or pd.isna(price) or price <= 0:
                continue
            spread_dict[(contract_prefix, far_contract)] = (spread, price)
        except Exception:
            continue
    if not spread_dict:
        return
    sorted_spreads = sorted(spread_dict.items(), key=lambda x: x[1][0], reverse=True)
    return sorted_spreads[0]

def rebalance1(context, bar_dict):
    today = context.now
    # ✅ 进度提示行
    print(f"📅 正在处理回测日期：{today.strftime('%Y-%m-%d')}")  # ✅ 进度提示行
    
    # 获取当前日期
    date_str = today.strftime("%Y-%m-%d")
    current_date = f"{today.year}{today.month:02d}{today.day:02d}"
    next_month = (today.month % 12) + 1
    next_year = today.year + (1 if next_month == 1 else 0)
    #次月合约数字部分
    next_code = f"{next_year % 100:02d}{next_month:02d}"
    
    # 获取当前日期的因子数据（宽格式）
    if date_str in context.factor_data.index:
        factor_row = context.factor_data.loc[date_str]
        # 转换为长格式
        factor_long = factor_row.reset_index()
        factor_long.columns = ['order_book_id', 'factor']
        # 过滤掉NaN值
        factor_long = factor_long.dropna()
    else:
        print(f"日期 {date_str} 在因子数据中不存在，跳过本次调仓")
        return
    
    # 获取当前日期的行情数据用于过滤
    market_df = context.market_data[context.market_data['date'] == date_str]
    
    # 过滤涨跌停和ST股票
    market_filtered = market_df[(market_df['limit_up_flag'] == False) & 
                               (market_df['limit_down_flag'] == False) & 
                               (market_df['ST'] == False)]
    
    # 只保留股票池内的股票
    market_filtered = market_filtered[market_filtered['order_book_id'].isin(context.stocks)]
    
    # 剔除所有不需要的股票（非正常）
    out_stocks = []
    stocks1 = all_instruments(type='CS',date = current_date).set_index("order_book_id")
    for obid in market_filtered['order_book_id']:
        try:
            special_type = stocks1.loc[obid]["special_type"]
            if special_type != "Normal":
                out_stocks.append(obid)
        except KeyError:
            out_stocks.append(obid)  # 没查到的，也剔除
    
    # 过滤ST、停牌、涨跌停股票
    for obid in market_filtered['order_book_id']:
        # ST或停牌
        if is_st_stock(obid) or is_suspended(obid):
            out_stocks.append(obid)
            continue
    
    # 保留不在out_stocks的股票
    valid_stocks = market_filtered[~market_filtered['order_book_id'].isin(out_stocks)]['order_book_id'].tolist()
    
    # 合并因子数据和有效股票列表
    df_target = factor_long[factor_long['order_book_id'].isin(valid_stocks)]
    
    # 动态计算目标持仓股票数：当前未停牌未退市未ST股票的10%（仿照纯多头策略）
    valid_stock_num = len(df_target)
    context.stock_num = max(1, int(valid_stock_num * 0.1))  # 至少持有1只
    
    # 筛选操作的股票（使用因子排序，仿照纯多头策略）
    small_cap_stocks = df_target.sort_values('factor', ascending=False).head(context.stock_num)['order_book_id'].tolist()
    
    # 期货合约选择
    future_info = select_future_type(["IF"],current_date,next_code)
    if not future_info:
        print(f"无法获取期货合约信息，跳过本次调仓")
        return
        
    contract_prefix, far_contract = future_info[0]
    price = future_info[1][1]
    multiplier = 200 if contract_prefix == "IC" else 300
    
    # 卖出不在目标池中的股票
    for stock in list(context.stock_account.positions.keys()):
        if is_suspended(stock):
            continue
        if stock not in small_cap_stocks:
            order_target_percent(stock, 0)
    
    a = float(context.stock_account.total_value)
    b = float(context.future_account.total_value)
    
    target_total_value_each = int((a+b) / 2)
    
    if a <= b:
        # 计算应该持有的合约数量
        keep = target_total_value_each // (price * multiplier)-1
        deposit(account_type="STOCK",amount=target_total_value_each-a)
        
        # 平均分配资金买入目标股票（仿照纯多头策略）
        weight = 1.0 / context.stock_num if context.stock_num > 0 else 0
        for stock in small_cap_stocks:
            order_target_percent(stock, weight)
            
        # 期货空头逻辑
        if not hasattr(context, "last_far_contract"):
            sell_open(far_contract, keep)
        else:
            if context.last_far_contract == far_contract:
                if keep < context.last_keep:
                    buy_close(far_contract, context.last_keep - keep)
                elif keep > context.last_keep:
                    sell_open(far_contract, keep - context.last_keep)
            else:
                buy_close(context.last_far_contract, context.last_keep)  # 平掉旧合约
                sell_open(far_contract, keep)  # 开新合约
        c = float(context.future_account.total_value)
        withdraw(account_type="FUTURE",amount=c - target_total_value_each)
            
    else:
        keep = (target_total_value_each // (price * multiplier)) -2
        print(keep*(price * multiplier))
        print(context.stock_account.cash,context.stock_account.total_value)
        
        # 平均分配资金买入目标股票（仿照纯多头策略）
        weight = 1.0 / context.stock_num if context.stock_num > 0 else 0
        for stock in small_cap_stocks:
            order_target_percent(stock, weight)
            
        print(context.stock_account.cash,context.stock_account.total_value)
        c = float(context.stock_account.total_value)

        try:
            withdraw(account_type="STOCK",amount=c- target_total_value_each)
        except ValueError as e:
            print(f"资金调整失败: {e}")
            # 不退出，继续执行
            pass
        
        deposit(account_type="FUTURE",amount=target_total_value_each-b)
        
        # 期货空头逻辑
        if not hasattr(context, "last_far_contract"):
            sell_open(far_contract, keep)
        else:
            if context.last_far_contract == far_contract:
                if keep < context.last_keep:
                    buy_close(far_contract, context.last_keep - keep)
                elif keep > context.last_keep:
                    sell_open(far_contract, keep - context.last_keep)
            else:
                buy_close(context.last_far_contract, context.last_keep)  # 平掉旧合约
                sell_open(far_contract, keep)  # 开新合约
            
    # 存储当前合约和持仓值
    context.last_keep = keep
    context.last_far_contract = far_contract

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