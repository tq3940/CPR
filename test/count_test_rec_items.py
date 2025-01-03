import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm  # 导入 tqdm
import os

'''
    对最终推荐结果前10计数
'''
# 读取CSV文件
PATH = "eval_log"
def fun(model, dataset):
    df_path = os.path.join(PATH, model, model+"__"+dataset, f"rec-{model}-test.csv")
    df = pd.read_csv(df_path, sep='\t')  # 替换为你的文件路径

    # 用于存储所有前10项的列表
    all_items = []

    # 处理rec_items列，提取每行列表的前10项
    for rec_list in tqdm(df['rec_items'], desc="Processing rows", unit="row"):
        # 用eval解析字符串为列表
        rec_list = eval(rec_list)
        # 取前10项
        all_items.extend(rec_list[:10])

    # 对所有前10项进行计数
    item_counts_rec = Counter(all_items)

    # 将计数结果转换为DataFrame
    count_df = pd.DataFrame(item_counts_rec.items(), columns=['Item', 'Count'])
    count_df_path = os.path.join(PATH, model, model+"__"+dataset, f"rec-{model}-test-item_counts.csv")
    # 保存为Excel文件
    count_df.to_csv(count_df_path, index=False)

    print(f"计数结果已保存为 {count_df_path}")


for model in ["BPRMF", "ComiRec", "ComiRecCPR", "CPRMF", "LightGCN", "LightGCNCPR"]:
    for dataset in ["Grocery_and_Gourmet_Food", "MIND_small", "MovieLens_1M"]:
        fun(model, dataset)