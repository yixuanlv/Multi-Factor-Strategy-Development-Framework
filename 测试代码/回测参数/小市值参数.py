# run_file_demo
from rqalpha import run_file

config_0 = {
    "base": {
        "start_date": "2020-01-01",
        "end_date": "2025-07-22",
        "stock_commission_multiplier":0.125,
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
            "output_file": r"rqalpha-localization\测试代码\回测结果\小市值策略_1.pkl"  # 将回测结果存储到本地文件
        }
    }
}
config_1 = {
    "base": {
        "start_date": "2019-01-01",
        "end_date": "2021-09-30",
        "stock_commission_multiplier":0.125,
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
            "output_file": r"rqalpha-localization\测试代码\回测结果\小市值策略_1.pkl"  # 将回测结果存储到本地文件
        }
    }
}
config_2 = {
    "base": {
        "start_date": "2021-09-30",
        "end_date": "2022-12-31",
        "stock_commission_multiplier":0.125,
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
            "output_file": r"rqalpha-localization\测试代码\回测结果\小市值策略_1.pkl"  # 将回测结果存储到本地文件
        }
    }
}
config_3 = {
    "base": {
        "start_date": "2023-01-01",
        "end_date": "2025-07-22",
        "stock_commission_multiplier":0.125,
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
            "output_file": r"rqalpha-localization\测试代码\回测结果\小市值策略_1.pkl"  # 将回测结果存储到本地文件
        }
    }
}



