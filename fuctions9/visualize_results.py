import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_result_data(file_path='result.pkl'):
    """加载回测结果数据"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

def create_performance_dashboard(data):
    """创建性能仪表板"""
    if data is None:
        return None
    
    # 提取关键数据
    portfolio = data.get('portfolio', pd.DataFrame())
    benchmark_portfolio = data.get('benchmark_portfolio', pd.DataFrame())
    summary = data.get('summary', {})
    trades = data.get('trades', pd.DataFrame())
    positions_weight = data.get('positions_weight', pd.DataFrame())
    stock_positions = data.get('stock_positions', pd.DataFrame())
    
    # 创建子图 - 将策略vs基准净值曲线放到最上面
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            '策略vs基准净值曲线', '累计收益率对比', '回撤分析', '风险指标对比'
        ),
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}]
        ],
        row_heights=[0.5, 0.15, 0.15, 0.2]
    )
    
    # 1. 策略vs基准净值曲线 - 添加持仓信息到hover
    if not portfolio.empty and not benchmark_portfolio.empty:
        strategy_curve = portfolio['unit_net_value']
        benchmark_curve = benchmark_portfolio['unit_net_value']
        
        # 准备持仓信息
        hover_texts = []
        
        for date in strategy_curve.index:
            # 获取该日期的持仓信息
            daily_positions = stock_positions[stock_positions.index == date]
            if not daily_positions.empty:
                position_info = []
                for _, pos in daily_positions.iterrows():
                    position_info.append(f"{pos['symbol']}: {pos['quantity']}股")
                hover_text = f"日期: {date.strftime('%Y-%m-%d')}<br>持仓: {'<br>'.join(position_info)}"
            else:
                hover_text = f"日期: {date.strftime('%Y-%m-%d')}<br>持仓: 无"
            hover_texts.append(hover_text)
        
        fig.add_trace(
            go.Scatter(
                x=strategy_curve.index, 
                y=strategy_curve.values,
                mode='lines', 
                name='策略净值', 
                line=dict(color='blue', width=3),
                hovertemplate='<b>策略净值</b><br>' + 
                             '净值: %{y:.4f}<br>' +
                             '%{text}<extra></extra>',
                text=hover_texts
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=benchmark_curve.index, 
                y=benchmark_curve.values,
                mode='lines', 
                name='基准净值', 
                line=dict(color='red', width=2),
                hovertemplate='<b>基准净值</b><br>' + 
                             '净值: %{y:.4f}<br>' +
                             '日期: %{x}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # 2. 累计收益率对比
    if not portfolio.empty and not benchmark_portfolio.empty:
        strategy_returns = (strategy_curve - 1) * 100
        benchmark_returns = (benchmark_curve - 1) * 100
        
        fig.add_trace(
            go.Scatter(x=strategy_returns.index, y=strategy_returns.values,
                      mode='lines', name='策略收益率(%)', line=dict(color='blue')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=benchmark_returns.index, y=benchmark_returns.values,
                      mode='lines', name='基准收益率(%)', line=dict(color='red')),
            row=2, col=1
        )
    
    # 3. 回撤分析
    if not portfolio.empty:
        # 计算回撤
        cumulative_returns = strategy_curve
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max * 100
        
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown.values,
                      mode='lines', name='回撤(%)', 
                      fill='tonexty', line=dict(color='orange')),
            row=3, col=1
        )
    
    # 4. 风险指标对比
    if not portfolio.empty and not benchmark_portfolio.empty:
        # 计算滚动夏普比率（假设无风险利率为3%）
        risk_free_rate = 0.03 / 252  # 日化无风险利率
        strategy_returns_daily = strategy_curve.pct_change().dropna()
        benchmark_returns_daily = benchmark_curve.pct_change().dropna()
        
        # 计算滚动夏普比率（20天窗口）
        rolling_sharpe = (strategy_returns_daily.rolling(20).mean() - risk_free_rate) / strategy_returns_daily.rolling(20).std() * np.sqrt(252)
        
        fig.add_trace(
            go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values,
                      mode='lines', name='滚动夏普比率(20日)', 
                      line=dict(color='purple')),
            row=4, col=1
        )
    
    # 4. 持仓权重分布 - 暂时移除，专注于主要图表
    pass
    
    # 5. 交易统计 - 暂时移除，专注于主要图表
    pass
    
    # 更新布局
    fig.update_layout(
        height=1400,
        title_text="量化策略回测结果可视化",
        showlegend=True
    )
    
    # 更新第一个子图的布局，使其更大更突出
    fig.update_xaxes(title_text="日期", row=1, col=1)
    fig.update_yaxes(title_text="净值", row=1, col=1)
    
    # 设置第一个子图的背景色和网格
    fig.update_xaxes(
        gridcolor='lightgray',
        gridwidth=1,
        row=1, col=1
    )
    fig.update_yaxes(
        gridcolor='lightgray',
        gridwidth=1,
        row=1, col=1
    )
    
    # 设置第二个子图（累计收益率）
    fig.update_xaxes(title_text="日期", row=2, col=1)
    fig.update_yaxes(title_text="收益率(%)", row=2, col=1)
    
    # 设置第三个子图（回撤分析）
    fig.update_xaxes(title_text="日期", row=3, col=1)
    fig.update_yaxes(title_text="回撤(%)", row=3, col=1)
    
    # 设置第四个子图（风险指标对比）
    fig.update_xaxes(title_text="日期", row=4, col=1)
    fig.update_yaxes(title_text="滚动夏普比率", row=4, col=1)
    
    return fig

def create_summary_table(data):
    """创建汇总表格"""
    if data is None:
        return None
    
    summary = data.get('summary', {})
    
    # 创建详细的指标表格
    key_metrics = {
        '指标': [
            # 基础收益指标
            '总收益率', '年化收益率', '基准年化收益率', '超额年化收益率',
            '总价值', '现金', '单位净值',
            
            # 风险指标
            '夏普比率', '索提诺比率', '最大回撤', '最大回撤持续天数',
            '波动率', '下行风险', 'VaR',
            
            # 相对表现指标
            '信息比率', '贝塔系数', '阿尔法系数', '跟踪误差',
            '超额收益率', '超额夏普比率', '超额波动率',
            
            # 交易统计
            '胜率', '超额胜率', '盈亏比', '换手率', '平均日换手率',
            
            # 周度指标
            '周度夏普比率', '周度索提诺比率', '周度胜率', '周度波动率',
            '周度信息比率', '周度贝塔系数', '周度阿尔法系数',
            
            # 月度指标
            '月度夏普比率', '月度波动率', '月度超额胜率',
            
            # 回撤相关
            '超额最大回撤', '超额最大回撤持续天数',
            '周度最大回撤', '周度跟踪误差',
            
            # 其他指标
            '周度溃疡指数', '周度溃疡绩效指数', '周度超额溃疡指数', '周度超额溃疡绩效指数'
        ],
        '数值': [
            # 基础收益指标
            f"{summary.get('total_returns', 0):.2%}",
            f"{summary.get('annualized_returns', 0):.2%}",
            f"{summary.get('benchmark_annualized_returns', 0):.2%}",
            f"{summary.get('excess_annual_returns', 0):.2%}",
            f"{summary.get('total_value', 0):,.0f}",
            f"{summary.get('cash', 0):,.0f}",
            f"{summary.get('unit_net_value', 0):.4f}",
            
            # 风险指标
            f"{summary.get('sharpe', 0):.2f}",
            f"{summary.get('sortino', 0):.2f}",
            f"{summary.get('max_drawdown', 0):.2%}",
            f"{summary.get('max_drawdown_duration_days', 0)}天",
            f"{summary.get('volatility', 0):.2%}",
            f"{summary.get('downside_risk', 0):.2%}",
            f"{summary.get('var', 0):.2%}",
            
            # 相对表现指标
            f"{summary.get('information_ratio', 0):.2f}",
            f"{summary.get('beta', 0):.2f}",
            f"{summary.get('alpha', 0):.2f}",
            f"{summary.get('tracking_error', 0):.2%}",
            f"{summary.get('excess_returns', 0):.2%}",
            f"{summary.get('excess_sharpe', 0):.2f}",
            f"{summary.get('excess_volatility', 0):.2%}",
            
            # 交易统计
            f"{summary.get('win_rate', 0):.2%}",
            f"{summary.get('excess_win_rate', 0):.2%}",
            f"{summary.get('profit_loss_rate', 0):.2f}",
            f"{summary.get('turnover', 0):.2f}",
            f"{summary.get('avg_daily_turnover', 0):.2%}",
            
            # 周度指标
            f"{summary.get('weekly_sharpe', 0):.2f}",
            f"{summary.get('weekly_sortino', 0):.2f}",
            f"{summary.get('weekly_win_rate', 0):.2%}",
            f"{summary.get('weekly_volatility', 0):.2%}",
            f"{summary.get('weekly_information_ratio', 0):.2f}",
            f"{summary.get('weekly_beta', 0):.2f}",
            f"{summary.get('weekly_alpha', 0):.2f}",
            
            # 月度指标
            f"{summary.get('monthly_sharpe', 0):.2f}",
            f"{summary.get('monthly_volatility', 0):.2%}",
            f"{summary.get('monthly_excess_win_rate', 0):.2%}",
            
            # 回撤相关
            f"{summary.get('excess_max_drawdown', 0):.2%}",
            f"{summary.get('excess_max_drawdown_duration_days', 0)}天",
            f"{summary.get('weekly_max_drawdown', 0):.2%}",
            f"{summary.get('weekly_tracking_error', 0):.2%}",
            
            # 其他指标
            f"{summary.get('weekly_ulcer_index', 0):.2f}",
            f"{summary.get('weekly_ulcer_performance_index', 0):.2f}",
            f"{summary.get('weekly_excess_ulcer_index', 0):.2f}",
            f"{summary.get('weekly_excess_ulcer_performance_index', 0):.2f}"
        ]
    }
    
    df_metrics = pd.DataFrame(key_metrics)
    
    # 创建HTML表格
    table_html = df_metrics.to_html(
        index=False, 
        classes=['table', 'table-striped', 'table-bordered'],
        float_format='%.2f'
    )
    
    return table_html

def generate_html_report(data, output_file='strategy_report.html'):
    """生成完整的HTML报告"""
    if data is None:
        print("数据加载失败，无法生成报告")
        return
    
    # 创建图表
    fig = create_performance_dashboard(data)
    
    # 创建汇总表格
    summary_table = create_summary_table(data)
    
    # 生成HTML内容
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>量化策略回测报告</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                margin-top: 30px;
            }}
            .summary-section {{
                background-color: #ecf0f1;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            .table th, .table td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: center;
            }}
            .table th {{
                background-color: #3498db;
                color: white;
            }}
            .table tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .info-box {{
                background-color: #d5f4e6;
                border-left: 4px solid #27ae60;
                padding: 10px;
                margin: 10px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>量化策略回测报告</h1>
            
            <div class="info-box">
                <strong>报告生成时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
            
            <h2>策略概览</h2>
            <div class="summary-section">
                {summary_table}
            </div>
            
            <h2>详细分析图表</h2>
            <div id="charts">
                {fig.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': True, 'displaylogo': False})}
            </div>
            
            <div class="info-box">
                <strong>使用说明:</strong> 将鼠标悬停在蓝色策略净值曲线上，可以查看每日的详细持仓信息。
            </div>
            
            <div class="info-box">
                <strong>说明:</strong> 本报告展示了策略的回测结果，包括净值曲线、风险指标、交易统计等关键信息。
            </div>
        </div>
    </body>
    </html>
    """
    
    # 保存HTML文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML报告已生成: {output_file}")

def mian(file_path='result.pkl'):
    """主函数"""
    print("开始生成策略回测可视化报告...")
    
    # 加载数据
    data = load_result_data(file_path=file_path)
    
    if data is not None:
        # 生成HTML报告
        generate_html_report(data)
        print("报告生成完成！")
    else:
        print("无法加载回测数据，请检查result.pkl文件是否存在")

if __name__ == "__main__":
    mian(file_path=r'C:\Users\14717\Desktop\rq本地化\rqalpha-localization\策略模板\1_股票多头\测试策略_1.pkl')