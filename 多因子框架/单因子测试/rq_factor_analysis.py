import pandas as pd
import datetime
import numpy as np


from rqfactor import *
import rqdatac
rqdatac.init('license', 'I6eb8ljE6tv9DWcYa3F0hERxbDdQ3f0RZzgBEnTHuaQlVX56azMDhdh6dYB42IJ7onLu0mAl3A1rRGFVTQuxE4jZcwoByZaySYlNuInciyFGarrHTz24mblwqbrC4RaCvKbkxP-tZ9S7ZjDY8pTNWu4uIslVXYb4XXL9NSwGI58=T7bU8OlDqvS3R5pPNN7s3PsfirJTCFSHPXpm5Ak3n0Dpzaze0NLbHWfZ-JcnlTBz7Oxec6dmkH9X4UB0OT0qxlkHA3pX_muOZI_zgMpCNZFH1wZ-DjeEMXrkqGGBKIo6_rZeaz130Fo1PLRY-rTw71gPmhD1oVg7GVh1kC7SWrk=')

d1 = '20210101'
d2 = '20211101'
f = Factor('pe_ratio_ttm')
ids = rqdatac.index_components('000300.XSHG',d1)
df = execute_factor(f,ids,d1,d2)
 
 
price = rqdatac.get_price(ids,d1,d2,frequency='1m',fields='close',expect_df=False)
target = datetime.time(14, 0)
mask = price.index.get_level_values('datetime').time == target
returns = price[mask].pct_change()
returns.index = pd.DatetimeIndex(returns.index.date)

engine = FactorAnalysisEngine()
engine.append(('winzorization-mad', Winzorization(method='mad')))
engine.append(('neutralization', Neutralization(industry='citics_2019', style_factors='all')))
engine.append(('rank_ic_analysis', ICAnalysis(rank_ic=True, industry_classification='sws')))
engine.append(('quantile_return_analysis', QuantileReturnAnalysis(quantile=10, benchmark=None)))


result = engine.analysis(df, returns, ascending=True, periods=1, keep_preprocess_result=True)

result['rank_ic_analysis'].summary()
result['quantile_return_analysis'].summary()


result['rank_ic_analysis'].show()
