
from datetime import datetime
from multiprocessing import cpu_count
from multiprocessing.pool import Pool


from Functions import *

import pandas as pd
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 5000)  # 最多显示数据的行数

# ===数据周期
period_type = 'M'  # W代表周，M代表月
# period_type = 'W'  # W代表周，M代表月
date_start = '2007-01-01'  # 回测开始时间
date_end = '2025-03-01'  # 回测结束时间

# ===读取所有股票代码的列表
# path = r'C:\001Python_program\实习项目\10_高管薪酬\data_02_合并后数据'
path =r'C:\001Python_program\实习项目_二期\00_数据\03_客户端数据\stock-trading-data-pro'
#
stock_code_list = get_stock_code_list_in_one_dir(path)

# ===循环读取并且合并
# 导入上证指数，保证指数数据和股票数据在同一天结束，不然会出现问题。
index_data = import_index_data(r'C:\001Python_program\实习项目_二期\00_数据\03_客户端数据\stock-main-index-data\sh000300.csv', back_trader_start=date_start, back_trader_end=date_end)


def calculate_by_stock(code):
    """
    整理数据核心函数
    :param code: 股票代码
    :return: 一个包含该股票所有历史数据的DataFrame
    """
   #====================================================================================================================
    if 'bj' in code:
        return pd.DataFrame()
    if 'sh68' in code:
        return pd.DataFrame()
    if 'sz30' in code:
        return pd.DataFrame()
   #====================================================================================================================
    print(code)
    # =读入股票数据
    # df = pd.read_csv(path + '/%s.csv' % code, encoding='gbk', skiprows=0, parse_dates=['交易日期'])
    df = pd.read_csv(path + '/%s.csv' % code, encoding='gbk', skiprows=1, parse_dates=['交易日期'])
    df['上市交易天数'] = df.index + 1
    cal_fuquan_price(df, fuquan_type='前复权')
    # 增加想要的字段
   #====================================================================================================================
    # =====================================================导入财务数据，并计算相关衍生指标
    # finance_data_path = r'C:\001Python_program\实习项目_二期\00_数据\03_客户端数据\stock-fin-data-xbx'
    # raw_fin_cols = ['C_cash_paid_of_distribution@xbx'] # 需要的原始数据 ，需要处理的原始字段 ,会被保留
    # # flow_fin_cols = ['R_np_atoopc@xbx']  #  计算 ttm 单季
    # flow_fin_cols = ['C_cash_paid_of_distribution@xbx']  #  计算 ttm 单季
    # cross_fin_cols = []  # 需要计算的截面数据   计算环比 同比
    # # derived_fin_cols = ['R_np_atoopc@xbx_ttm同比','R_np_atoopc@xbx_ttm']
    # derived_fin_cols = ['C_cash_paid_of_distribution@xbx_ttm']
    # # derived_fin_cols = []
    # extra_agg_dict = {}
    #
    # finance_df = import_fin_data(code, finance_data_path)
    #
    # # 提前定义一个空的列表，避免财务数据为空
    # columns_list = []
    # if not finance_df.empty:  # 如果数据不为空
    #
    #     # 计算财务数据：选取需要的字段、计算指定字段的同比、环比、ttm等指标
    #     finance_df, finance_df_ = proceed_fin_data(finance_df, raw_fin_cols, flow_fin_cols, cross_fin_cols,
    #                                                derived_fin_cols)
    #     # 获取去年同期的数据,如果需要计算，把这两行代码取消注释即可
    #     # df_, columns_list = get_his_data(finance_df_, ['R_np_atoopc@xbx', 'R_np@xbx'], span='4q')
    #     # finance_df = pd.merge(left=finance_df, right=df_[['publish_date'] + columns_list], on='publish_date',
    #     #                       how='left')
    #     # 财务数据和股票k线数据合并，使用merge_asof
    #     df = pd.merge_asof(left=df, right=finance_df, left_on='交易日期', right_on='publish_date',
    #                        direction='backward')
    #
    # else:  # 如果数据为空
    #     for col in raw_fin_cols + derived_fin_cols:
    #         df[col] = np.nan
    # # 有修改
    # for col in raw_fin_cols + columns_list:  # 财务数据在周期转换的时候，都是选取最后一天的数据
    #     extra_agg_dict[col] = 'last'
    # =计算财务因子：这个函数需要大家根据需要自行修改
    # print(df.tail(100))
    # print(df.columns)
    # exit()
    # df = calc_fin_factor(df, extra_agg_dict)
    # =====================================================补充其他数据，分红

    # df['换手率'] = df['成交量'] / (df['流通市值'] / df['收盘价'])
    # df['1-换手率'] = 1 - df['换手率']
    # def cal_cgo_factor(df, window=3):
    #     def process_row(index, df, window):
    #         if index < window - 1:
    #             return np.nan
    #         else:
    #             window_df = df.iloc[index - window + 1:index + 1]
    #             window_df['权重_1'] = window_df['1-换手率'].prod() / window_df['1-换手率'].cumprod() * window_df[
    #                 '换手率']
    #             window_df['权重_2'] = window_df['权重_1'] / window_df['权重_1'].sum()
    #             window_df['cgo涨跌幅'] = (window_df['收盘价_复权'] - (
    #                         window_df['权重_2'] * window_df['收盘价_复权']).sum()) / window_df['收盘价_复权']
    #             return (window_df['权重_2'] * window_df['收盘价_复权']).sum()
    #     return df.apply(lambda row: process_row(row.name, df, window), axis=1)
    #
    # # windows = [5, 10, 20, 60, 100, 125]
    # windows = [5, 10, 20]
    # for w in windows:
    #     df[f'RP_{w}'] = cal_cgo_factor(df, window=w)
    #     df[f'CGO_{w}'] = (df['收盘价_复权'] - df[f'RP_{w}']) / df['收盘价_复权']
    # print(df.tail(100))

    # ==============处理cgo
    # df['换手率'] = df['volume'] / (df['circulating_supply'])
    # df['1-换手率'] = 1 - df['换手率']
    def calculate_cgo_numpy(close, volume, circulating_supply, windows):
        close = np.asarray(close)
        volume = np.asarray(volume)
        circulating_supply = np.asarray(circulating_supply)
        turnover = volume / circulating_supply
        one_minus_turnover = 1 - turnover
        n = len(close)
        results = {}
        for window in windows:
            rp_array = np.full(n, np.nan)
            for i in range(window - 1, n):
                # 当前窗口数据
                window_close = close[i - window + 1:i + 1]
                window_turn = turnover[i - window + 1:i + 1]
                window_one_minus_turn = one_minus_turnover[i - window + 1:i + 1]
                cumprod = np.cumprod(window_one_minus_turn)
                total_prod = cumprod[-1]
                # 避免除以0
                cumprod_safe = np.where(cumprod == 0, 1e-10, cumprod)
                weight_1 = total_prod / cumprod_safe * window_turn
                weight_2 = weight_1 / np.sum(weight_1)
                rp_array[i] = np.sum(weight_2 * window_close)
            cgo_array = (close - rp_array) / close
            results[f'RP_{window}'] = rp_array
            results[f'CGO_{window}'] = cgo_array
            # print(f'窗口 {window} 计算完毕')
        return results

    windows = [5, 10, 20, 60, 90,180,120,200,240]
    # windows = [5, 10, 20]

    result_dict = calculate_cgo_numpy(
        close=df['收盘价_复权'].values,
        volume=df['成交量'].values,
        circulating_supply=(df['流通市值']/df['收盘价']).values,
        windows=windows
    )

    # 把结果写回 df
    for key, val in result_dict.items():
        df[f'{key}'] = val

    # ======================================================

    # print(df.tail(100))
    # exit()
    # print(df[['股票代码','交易日期','总市值','总资产','总负债']].tail(1000))
    df.loc[df['新版申万一级行业名称'].astype(str).str.contains('金融服务', na=False), '新版申万一级行业名称'] = '非银金融'
    df.loc[df['新版申万一级行业名称'].astype(str).str.contains('信息设备', na=False), '新版申万一级行业名称'] = '通信'
    df.loc[df['新版申万一级行业名称'].astype(str).str.contains('信息服务', na=False), '新版申万一级行业名称'] = '通信'
    df.loc[df['新版申万一级行业名称'].astype(str).str.contains('交运设备', na=False), '新版申万一级行业名称'] = '交通运输'
    df.loc[df['新版申万一级行业名称'].astype(str).str.contains('建筑建材', na=False), '新版申万一级行业名称'] = '建筑材料'
   #====================================================================================================================
    # df.to_csv(f'处理后数据/{code}.csv',encoding='gbk',index=False)
    # =计算涨跌幅
    df['涨跌幅'] = df['收盘价'] / df['前收盘价'] - 1
    df['开盘买入涨跌幅'] = df['收盘价'] / df['开盘价'] - 1  # 为之后开盘买入做好准备

    # =将股票和上证指数合并，补全停牌的日期，新增数据"是否交易"、"指数涨跌幅"
    df = merge_with_index_data(df, index_data)
    if df.empty:
        return pd.DataFrame()

    # =计算涨跌停价格
    df = cal_zdt_price(df)
    # 12:01 20
    # =计算下个交易的相关情况
    df['下日_是否交易'] = df['是否交易'].shift(-1)
    df['下日_一字涨停'] = df['一字涨停'].shift(-1)
    df['下日_开盘涨停'] = df['开盘涨停'].shift(-1)
    df['下日_是否ST'] = df['股票名称'].str.contains('ST').shift(-1)
    df['下日_是否退市'] = df['股票名称'].str.contains('退').shift(-1)
    df['下日_开盘买入涨跌幅'] = df['开盘买入涨跌幅'].shift(-1)

    # =将日线数据转化为月线或者周线
    df = transfer_to_period_data(df, period_type=period_type)
    # df = transfer_to_period_data_twoweeks(df)
    # column_add_list = ['自由现金流','企业价值','盈利质量']
    # df = transfer_to_period_data_season(df,column_add_list=column_add_list)
    # print(df.tail(100))
    df = df.sort_values('交易日期')
    # =对数据进行整理
    # 删除上市的第一个周期
    df.drop([0], axis=0, inplace=True)  # 删除第一行数据
    # 删除2007年之前的数据
    df = df[df['交易日期'] > pd.to_datetime('20061215')]
    # 计算下周期每天涨幅
    df['下周期每天涨跌幅'] = df['每天涨跌幅'].shift(-1)
    del df['每天涨跌幅']
    # 此处省略，请参考原程序「选股数据整理.py」
    print(code, '计算完成')

    return df  # 返回计算好的数据


if __name__ == '__main__':

    # target = [i for i in stock_code_list if '000100' in i][0]
    # calculate_by_stock(target);exit()
    # calculate_by_stock(stock_code_list[1120]);exit()

    # 标记开始时间
    start_time = datetime.now()

    # 并行处理
    multiply_process = True
    if multiply_process:
        # ===并行提速的办法
        with Pool(max(cpu_count() - 2, 1)) as pool:
            df_list = pool.map(calculate_by_stock, sorted(stock_code_list))
    # 传行处理
    else:
        df_list = []
        for stock_code in stock_code_list:
            data = calculate_by_stock(stock_code)
            df_list.append(data)
    print('读入完成, 开始合并，消耗事件', datetime.now() - start_time)

    # 合并为一个大的DataFrame
    all_stock_data = pd.concat(df_list, ignore_index=True)
    """
    20191107西蒙斯直播的并行加速的方法
    回看地址：https://appr3RLZXlo9494.h5.xeknow.com/st/0mbPG5Flm
    """

    all_stock_data.sort_values(['交易日期', '股票代码'], inplace=True)  # ===将数据存入数据库之前，先排序、reset_index
    all_stock_data.reset_index(inplace=True, drop=True)
    # 将数据存储到hdf文件
    # all_stock_data.to_hdf(
    #     './all_stock_data_' + period_type + '.h5', 'df',
    #     mode='w')
    all_stock_data.to_pickle('股票数据_频.pkl')
    # all_stock_data.to_csv('股票数据_月频.csv',index=False,encoding='gbk')
    # 打印一下benchmark，看一下花了多久
    print(datetime.now() - start_time)
    # ===注意事项
    # 目前我们只根据市值选股，所以数据中只有一些基本数据加上市值。
    # 实际操作中，会根据很多指标进行选股。在增加这些指标的时候，一定要注意在这两个函数中如何增加这些指标：merge_with_index_data(), transfer_to_period_data()
    # 比如增加：成交量、财务数据
