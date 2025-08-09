import h5py
import pandas as pd
import numpy as np

def read_h5_to_df(file_path, dataset_path=None):
    """
    使用 h5py 读取 h5 文件为 DataFrame，避免使用 pandas 的 HDFStore（需要 tables 库）。

    参数：
    - file_path: str，h5 文件路径
    - dataset_path: str，可选，数据集路径（即类似 '/data' 的 key）

    返回：
    - df: pandas.DataFrame
    """
    with h5py.File(file_path, 'r') as f:
        # 如果用户没指定路径，列出所有路径供参考
        if dataset_path is None:
            print("可用数据集路径如下：")

            def visit_fn(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(name)

            f.visititems(visit_fn)
            return None  # 用户可以重新指定 dataset_path 再调用

        data = f[dataset_path][()]

        # 自动转换成 DataFrame
        if isinstance(data, np.ndarray):
            try:
                df = pd.DataFrame(data)
                return df
            except Exception as e:
                raise ValueError(f"无法转换成 DataFrame: {e}")
        else:
            raise TypeError("不是数组类型，无法转换成 DataFrame")


if __name__ == '__main__':
    # ===========显示 所有表名
    df = read_h5_to_df(r'C:\Users\14717\.rqalpha\bundle\futures.h5')
    # ===========显示当前表内容
    df = read_h5_to_df(r'C:\Users\14717\.rqalpha\bundle\futures.h5','IF2512')
    print(df)