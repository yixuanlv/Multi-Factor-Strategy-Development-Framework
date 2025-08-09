from rqalpha import run_file
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))
from 回测参数.小市值参数 import config_0,config_1, config_2, config_3
from 因子数据.因子生成 import generate_factor_data

if __name__ == "__main__":
    #generate_factor_data()  # 生成因子数据

    strategy_file_path = r"rqalpha-localization\测试代码\策略函数\小市值策略.py"

    run_file(strategy_file_path, config_0)
    #run_file(strategy_file_path, config_2)
    #run_file(strategy_file_path, config_3)