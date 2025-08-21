#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
numpy版本兼容性补丁
解决numpy 2.x版本与旧版本pickle文件的兼容性问题
"""

import sys
import numpy as np

def apply_numpy_compatibility_patches():
    """应用numpy兼容性补丁"""
    print("应用numpy兼容性补丁...")
    
    # 检查numpy版本
    numpy_version = np.__version__
    print(f"当前numpy版本: {numpy_version}")
    
    if numpy_version.startswith('2.'):
        print("检测到numpy 2.x版本，应用兼容性补丁...")
        
        # 补丁1：添加缺失的numpy._core.numeric模块
        try:
            import numpy.core.numeric as numeric
            sys.modules['numpy._core.numeric'] = numeric
            print("✓ 补丁1应用成功：numpy._core.numeric")
        except Exception as e:
            print(f"✗ 补丁1失败：{e}")
        
        # 补丁2：添加缺失的numpy.core.numerictypes模块
        try:
            import numpy.core.numerictypes as numerictypes
            sys.modules['numpy.core.numerictypes'] = numerictypes
            print("✓ 补丁2应用成功：numpy.core.numerictypes")
        except Exception as e:
            print(f"✗ 补丁2失败：{e}")
        
        # 补丁3：添加缺失的numpy.core.multiarray模块
        try:
            import numpy.core.multiarray as multiarray
            sys.modules['numpy.core.multiarray'] = multiarray
            print("✓ 补丁3应用成功：numpy.core.multiarray")
        except Exception as e:
            print(f"✗ 补丁3失败：{e}")
        
        # 补丁4：添加缺失的numpy.core.umath模块
        try:
            import numpy.core.umath as umath
            sys.modules['numpy.core.umath'] = umath
            print("✓ 补丁4应用成功：numpy.core.umath")
        except Exception as e:
            print(f"✗ 补丁4失败：{e}")
        
        # 补丁5：处理pickle兼容性
        try:
            # 重写pickle的find_class方法以处理numpy类型
            import pickle
            original_find_class = pickle.Unpickler.find_class
            
            def patched_find_class(self, module, name):
                try:
                    return original_find_class(self, module, name)
                except (ImportError, AttributeError) as e:
                    # 尝试在numpy模块中查找
                    if module.startswith('numpy'):
                        try:
                            # 尝试导入numpy模块
                            if module == 'numpy.core.numeric':
                                return getattr(np.core.numeric, name)
                            elif module == 'numpy.core.multiarray':
                                return getattr(np.core.multiarray, name)
                            elif module == 'numpy.core.umath':
                                return getattr(np.core.umath, name)
                            elif module == 'numpy':
                                return getattr(np, name)
                            else:
                                # 尝试动态导入
                                import importlib
                                mod = importlib.import_module(module)
                                return getattr(mod, name)
                        except Exception:
                            pass
                    raise e
            
            pickle.Unpickler.find_class = patched_find_class
            print("✓ 补丁5应用成功：pickle兼容性")
        except Exception as e:
            print(f"✗ 补丁5失败：{e}")
        
        print("numpy兼容性补丁应用完成")
    else:
        print("numpy版本兼容，无需应用补丁")

def safe_pickle_load(file_path):
    """安全地加载pickle文件"""
    # 首先应用兼容性补丁
    apply_numpy_compatibility_patches()
    
    try:
        # 尝试直接加载
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"直接加载失败: {e}")
        
        # 尝试使用不同的编码
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f, encoding='latin1')
        except Exception as e2:
            print(f"latin1编码也失败: {e2}")
            
            # 尝试使用joblib
            try:
                import joblib
                return joblib.load(file_path)
            except ImportError:
                print("joblib不可用")
            except Exception as e3:
                print(f"joblib加载也失败: {e3}")
        
        # 如果所有方法都失败，抛出异常
        raise ValueError(f"无法加载文件 {file_path}，所有兼容性方法都失败")

if __name__ == "__main__":
    # 测试补丁
    apply_numpy_compatibility_patches()
