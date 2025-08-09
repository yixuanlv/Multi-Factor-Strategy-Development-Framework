#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速pkl转csv转换脚本
直接运行即可将pkl文件转换为csv
"""

import os

import pandas as pd


def quick_convert(pkl_file_path,output_folder):
    """快速转换函数"""

    print("🚀 开始快速转换...")
    print(f"📁 pkl文件: {pkl_file_path}")
    print(f"📁 输出文件夹: {output_folder}")
    
    # 检查文件是否存在
    if not os.path.exists(pkl_file_path):
        print(f"❌ pkl文件不存在: {pkl_file_path}")
        return
    
    # 显示可用数据表

    # 执行转换
    pkl = pd.read_pickle(pkl_file_path)
    for key in pkl.keys():
        try:
            df = pd.DataFrame(pkl[key])
            df.to_csv(f'{output_folder}/{key}.csv')
        except:
            print(f'===={key}====csv保存失败')
            print(pkl[key])
    # print(f"\n🎉 转换完成！共保存 {len(result)} 个文件")

if __name__ == "__main__":
    pkl_file_path = r"C:\Users\14717\Desktop\rq本地化\rqalpha-localization\策略模板\1_股票多头\测试策略_1.pkl"
    output_folder = r"C:\Users\14717\Desktop\rq本地化\rqalpha-localization\策略模板\1_股票多头"
    quick_convert(pkl_file_path, output_folder) 