
import glob

import pandas as pd
import numpy as np
import datetime
import time
import os
import sys
from datetime import datetime, timedelta
import datetime

from tqdm import tqdm
import matplotlib.pyplot as plt
import rqdatac as rq
def generate_factor_data(
    license_key='license',
    license_value='FTp2uHaRsHQhI9XoFCwKkJNyYdaZ9f6oMHT4kPr1Pq95n2VHHH_KnxinjLrktuwQ1AuW72X5vSrS01MGqOw8OWRD8W3B_EWHNBWGz5vE2A3cLKxEz25vNYeXIbzDbt5v8crTY1OOkjjypfOnfnItH5r95_8C3ck0QmGHgMyMbTw=U_T-NcVYHMdYw9LovsISVLqco69JY8b6093uhl6PUG4S2gEXQOO_RAuTwjbFEK-GXnzutyBAP-s-3JpXpkjdnfQ96Ypjzl6J1DULxSGzqWQo6LWJwVw8YB125DfT5oSHqk9lUsgriFojSQqG92uKh8HKpZK4fVvNhp3Lv2U410c=',
    start_date='2010-01-01',
    end_date=datetime.datetime.now().strftime('%Y-%m-%d'),
    output_path=r'rqalpha\测试代码\因子数据\factor_data.pkl'
):
    """
    封装因子数据生成流程为函数
    """

    print(f"开始生成因子数据，从{start_date} 到 {end_date}")
    # 初始化
    rq.init(license_key, license_value)

    # 获取全部A股列表
    all_stocks = rq.all_instruments(type='CS', date=None)['order_book_id'].tolist()

    # 设置时间范围
    if end_date is None:
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')

    # 获取后复权日线行情数据
    price_data = rq.get_price(
        all_stocks,
        start_date=start_date,
        end_date=end_date,
        frequency='1d',
        adjust_type='none',
        expect_df=True
    )
    price_data['limit'] = (price_data['close'] >= price_data['limit_up']) | (price_data['close'] <= price_data['limit_down'])
    vwap = rq.get_vwap(all_stocks, start_date=start_date, end_date=end_date, frequency='1d')
    vwap = pd.DataFrame(vwap)
    vwap.columns = ['vwap']
    market_cap_data = rq.get_factor(all_stocks, 'market_cap', start_date=start_date, end_date=end_date, expect_df=True)
    ST_stock = rq.is_st_stock(all_stocks, start_date=start_date, end_date=end_date)
    ST_stock = ST_stock.reset_index().rename(columns={'index': 'date'})
    ST_stock = ST_stock.melt(id_vars='date', var_name='order_book_id', value_name='ST')

    merged_data = pd.merge(vwap, market_cap_data, on=['order_book_id', 'date'], how='outer')
    merged_data = pd.merge(merged_data, price_data, on=['order_book_id', 'date'], how='outer')
    merged_data = pd.merge(merged_data, ST_stock, on=['order_book_id', 'date'], how='outer')


    
    merged_data['is_suspended'] = merged_data.groupby('order_book_id').apply(
        lambda x: ((x['market_cap'] == x['market_cap'].shift()) & (x['total_turnover'] == x['total_turnover'].shift())).astype(int)
    ).reset_index(level=0, drop=True).T

    merged_data.to_pickle(output_path)
    print(f"从{start_date} 到 {end_date} 的因子数据已保存到 {output_path}")
    return merged_data
