from rqalpha.api import *
import pandas as pd
import os
import rqdatac as rq
from rqalpha import run_func

rq.init('license', 'B1T4WrPGQ0YBin6JPZm_DlLj3JGxAiuGzi9-SuUNqOUce6MrZ7yLejN2O9OWDPBJ3U6cVO-6uaK8Wn29JTxgNRHrqWJgGHTtf483vOI3bFPOMknL3dEAgQZJTLHJ7LyMZcalsTdqlVOyhT0mlNm_9iNEZBgTxhQW0X_DHOzxLBg=efJUVSAH7ub2Gg33v_nzcsj25LD0caxMdEYz93JazlzGzbk5tG6kxZKqGx9LcGX58GmZmd2IxNETyXC1jdWnnH7u97TeyhpkfP1lfJ5h5sp-1vSqT0tXvJH0SOGoIY4spxMnMeEprvKck7cwl1As3xh8y068HH8MApBtIrP7aOA=')

config_0 = {
    "base": {
        "start_date": "2019-01-01",
        "end_date": "2019-01-22",
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
    scheduler.run_daily(rebalance1)
    context.stocks = all_instruments(type='CS', date=None)['order_book_id'].tolist()
    context.stock_num = 100
    context.factor_data = pd.read_pickle(r"C:\Users\19039\Desktop\框架\测试代码\因子数据\factor_data.pkl")
    context.factor_data['order_book_id'] = context.factor_data['order_book_id'].astype(str)
    

def before_trading(context):
    pass

def handle_bar(context, bar_dict):
    pass
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
    #日期处理
    date_str = today.strftime("%Y-%m-%d")
    current_date = f"{today.year}{today.month:02d}{today.day:02d}"
    next_month = (today.month % 12) + 1
    next_year = today.year + (1 if next_month == 1 else 0)
    #次月合约数字部分
    next_code = f"{next_year % 100:02d}{next_month:02d}"
    #剔除所有不需要的股票（非正常）
    df = context.factor_data[context.factor_data['date'] == date_str]
    out_stocks = []
    stocks1 = all_instruments(type='CS',date = current_date).set_index("order_book_id")
    for obid in df['order_book_id']:
        try:
            special_type = stocks1.loc[obid]["special_type"]
            if special_type != "Normal":
                out_stocks.append(obid)
        except KeyError:
            out_stocks.append(obid)  # 没查到的，也剔除
    df_target = df[~df['order_book_id'].isin(out_stocks)]
    #筛选操作的股票
    small_cap_stocks = df_target.sort_values('market_cap').head(context.stock_num)['order_book_id'].tolist()                                                                      
    future_info = select_future_type(["IF"],current_date,next_code)

    contract_prefix, far_contract = future_info[0]
    price = future_info[1][1]
    multiplier = 200 if contract_prefix == "IC" else 300
    
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
        
        for stock in small_cap_stocks:
            order_target_value(stock,keep*(price * multiplier)/context.stock_num)
    # 如果是第一次运行或合约发生变化，根据情况进行卖出或买入
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
        for stock in small_cap_stocks:
            order_target_value(stock,keep*(price * multiplier)/context.stock_num)
        print(context.stock_account.cash,context.stock_account.total_value)
        c = float(context.stock_account.total_value)

        # withdraw(account_type="STOCK",amount=c- target_total_value_each)
        try:
            withdraw(account_type="STOCK",amount=c- target_total_value_each)
        except:
            exit()
        
        
    # 计算应该持有的合约数量
        
        deposit(account_type="FUTURE",amount=target_total_value_each-b)
        
    # 如果是第一次运行或合约发生变化，根据情况进行卖出或买入
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