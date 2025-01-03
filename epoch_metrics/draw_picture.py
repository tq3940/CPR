import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np

# 读取Excel文件
data = pd.read_excel("epoch_metrics\MIND_small_all_models_NDCG@2.xlsx")

# 检查并处理数据中的无效值
data = data.replace([np.inf, -np.inf], np.nan)

# 定义列名和颜色
cold_colors = ["#1f77b4", "#17becf", "#2ca02c"]  # 冷色调
warm_colors = ["#ff7f0e", "#d62728", "#C86D39"]  # 暖色调
columns_bpr = ["MF-BPR", "ComiRec-BPR", "LightGCN-BPR"]
columns_cpr = ["MF-CPR", "ComiRec-CPR", "LightGCN-CPR"]

# 创建图表
plt.figure(figsize=(10, 6))

# 增加插值点数
num_points = 500

# 绘制冷色调曲线
for col, color in zip(columns_bpr, cold_colors):
    if col in data.columns:
        col_data = data[col].dropna()
        x = np.arange(len(col_data))
        y = col_data.values
        if len(x) > 1:  # 确保数据点足够进行插值
            x_smooth = np.linspace(x.min(), x.max(), num_points)
            y_smooth = make_interp_spline(x, y)(x_smooth)
            plt.plot(x_smooth, y_smooth, label=col, color=color)
        else:
            plt.plot(x, y, label=col, color=color)

# 绘制暖色调曲线
for col, color in zip(columns_cpr, warm_colors):
    if col in data.columns:
        col_data = data[col].dropna()
        x = np.arange(len(col_data))
        y = col_data.values
        if len(x) > 1:  # 确保数据点足够进行插值
            x_smooth = np.linspace(x.min(), x.max(), num_points)
            y_smooth = make_interp_spline(x, y)(x_smooth)
            plt.plot(x_smooth, y_smooth, label=col, color=color)
        else:
            plt.plot(x, y, label=col, color=color)

# 设置图例、标题和坐标轴标签+
plt.legend()
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows

plt.title("MIND 数据集\n训练过程中 NGCD@2 变化曲线", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("NDCG@2", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)

# 保存或显示图表
plt.tight_layout()
plt.savefig("epoch_metrics\MIND变化曲线.png", dpi=300)  # 保存图片
plt.show()  # 显示图表
