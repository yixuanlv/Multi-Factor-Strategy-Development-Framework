from rqalpha.api import *
import pandas as pd
import os
import rqdatac

__config__ = {
    "base": {
        "start_date": "2022-01-01",
        "end_date": "2024-03-22",
        "accounts": {
            "future": 10000000
        },
        "benchmark": "000300.XSHG"
    },
    "mod": {
        "sys_analyser": {
            "enabled": True,
            "plot": True
        }
    }
}

def init(context):
    scheduler.run_daily(rebalance)
    # 所有期货
    context.contracts = {"IF": "IF", "IC": "IC", "IH": "IH"}
    context.roll_month = 1 

    # 选取最低贴水最小的期货
    context.futures_num = 1
    # 预加载factor_data.pkl
   
    
def before_trading(context):
    pass


# 你选择的期货数据更新将会触发此段逻辑，例如日线或分钟线更新
def handle_bar(context, bar_dict):
    pass

def rebalance(context,bar_dict):
    
    a = f"{context.now.year}{context.now.month:02d}{context.now.day:02d}"
    if context.now.month == 12:
        b = f"{context.now.year%100+1}01"
    else:
        b = f"{context.now.year%100}{(context.now.month+1):02d}"
    spread_dict = {}
    

    # 遍历所有合约类型（如 IF、IC、IH）
    for contract_prefix in context.contracts.values():
        far_contract = f"{contract_prefix}{b}"
        print(f"尝试获取合约: {far_contract}")

        try:
            basis_df = rqdatac.futures.get_basis(far_contract, a, a)

            if basis_df.empty:
                print(f"✘ 合约 {far_contract} 无数据，跳过")
                continue

            spread = basis_df["basis"].iloc[0]
            price = basis_df["close"].iloc[0]

            if pd.isna(spread) or pd.isna(price) or price <= 0:
                print(f"✘ 合约 {far_contract} 数据非法，跳过")
                continue

            print(f"✔ 合约 {far_contract}：贴水 = {spread:.2f}，收盘价 = {price:.2f}")
            spread_dict[(contract_prefix, far_contract, a)] = [spread, price]

        except Exception as e:
            print(f"✘ 异常跳过合约 {far_contract}: {e}")
            continue

    if not spread_dict:
        print("⚠ 未找到任何有效贴水数据，跳过 rebalance")
        return

    # 选择贴水（或升水）最大的合约
    sorted_spreads = sorted(spread_dict.items(), key=lambda x: x[1][0], reverse=True)
    contract_prefix, far_contract, _ = sorted_spreads[0][0]
    spread = sorted_spreads[0][1][0]
    price = sorted_spreads[0][1][1]

    print(f"\n>>> 选中合约: {far_contract}，升贴水最大: {spread:.2f}，价格: {price:.2f}")

    # 如果是首次 rebalance，开空仓
    if not hasattr(context, "last_far_contract"):
        if far_contract[:2] == "IC":
            keep = (context.future_account.cash/200) // price
        else:
            keep = (context.future_account.cash/300) // price
        print(f"首次建仓：做空 {far_contract}，手数：{keep}")
        sell_open(far_contract,keep)
        context.last_keep = keep

    # 如果已持仓，且合约发生变化，先平仓再建仓
    elif context.last_far_contract != far_contract:
        print(f"换仓操作：平掉 {context.last_far_contract}，建仓 {far_contract}")
        buy_close(context.last_far_contract, context.last_keep)
        if far_contract[:2] == "IC":
            keep = (context.future_account.cash/200) // price
        else:
            keep = (context.future_account.cash/300) // price
        sell_open(far_contract, keep)
        context.last_keep = keep

    # 如果合约没变，就不动
    else:
        print(f"维持当前空仓不动：{context.last_far_contract}")

    # 更新当前持仓记录
    context.last_far_contract = far_contract



if __name__ == '__main__':
    from rqalpha import run_func
    run_func(
        init=init,
        before_trading=before_trading,
        handle_bar=handle_bar,
        config=__config__
    )




