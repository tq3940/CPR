import pandas as pd
from collections import Counter
from tqdm import tqdm
import numpy as np
import os 
'''
    根据train中得出的分组，对test按每组的个数来欠采样
'''
PATH = "rechorus\data"

def fun(dataset, random_state = 42):
    # 1. 读取 test.csv 和 item_classes.csv
    test_set_path = os.path.join(PATH, dataset, "test.csv")
    test_df = pd.read_csv(test_set_path, sep='\t')  # 替换为你的 test.csv 路径
    item_classes_df = pd.read_csv(os.path.join(PATH, dataset, "item_classes.csv"))  # 读取 item_classes.csv，包含每个 item_id 的类别
    class_stats_df = pd.read_csv(os.path.join(PATH, dataset, "class_statistics.csv"))  # 读取 class_statistics.csv，包含每个类的 total_items

    # 2. 创建 item_id 到 class 的映射
    item_to_class = dict(zip(item_classes_df['item_id'], item_classes_df['class']))

    # 3. 获取每个类的总样本数（total_items）
    class_total_items = dict(zip(class_stats_df['class'], class_stats_df['total_items']))

    # 4. 计算每个 item_id 的采样权重（与所属类别的 total_items 成正比）
    class_probabilities = {cls: total_items / sum(class_total_items.values()) for cls, total_items in class_total_items.items()}

    # 5. 将 test_df 按照 item_id 映射到类别
    test_df['class'] = test_df['item_id'].map(item_to_class)

    # 6. 按类别进行采样，采样概率正比于该类别的 total_items
    sampled_items = []
    for cls in tqdm(sorted(class_total_items.keys()), desc="Sampling items by class"):
        # 获取当前类别的所有 item_id
        items_in_class = test_df[test_df['class'] == cls]
        
        # 计算该类别的采样权重
        class_weight = class_probabilities[cls]
        
        # 对该类别的 item_id 按照采样概率进行抽样(至少3个)
        if items_in_class.shape[0] > 25:
            sampled_class_items = items_in_class.sample(frac=class_weight, random_state=random_state)
        else:
            sampled_class_items = items_in_class.sample(n = 3)
        # 将抽样结果加入到最终的样本列表
        sampled_items.append(sampled_class_items)

    # 7. 合并所有类别的采样结果
    sampled_df = pd.concat(sampled_items, ignore_index=True)

    # 8. 保存采样后的数据
    sampled_df_path = os.path.join(PATH, dataset, "test_sampled_items.csv")
    sampled_df.to_csv(sampled_df_path, index=False, quoting=3, sep='\t')

    # 9. 统计每个类别在采样后的 item 个数和出现次数之和
    sampling_stats = {}
    for cls in sorted(class_total_items.keys()):
        # 获取当前类别采样后的所有 item_id
        sampled_class_df = sampled_df[sampled_df['class'] == cls]
        
        # 计算该类别中采样的 item 数量
        total_sampled_items = len(sampled_class_df)
        
        # 计算该类别中采样的 item 出现次数之和
        total_sampled_counts = sampled_class_df['item_id'].map(test_df['item_id'].value_counts()).sum()
        
        # 保存统计结果
        sampling_stats[cls] = {'total_sampled_items': total_sampled_items, 'total_sampled_counts': total_sampled_counts}

    # 10. 转换统计结果为 DataFrame 并保存
    sampling_stats_df = pd.DataFrame([
        {'class': cls, 'total_sampled_items': stats['total_sampled_items'], 'total_sampled_counts': stats['total_sampled_counts']}
        for cls, stats in sampling_stats.items()
    ])
    sampling_stats_df_path = os.path.join(PATH, dataset, "test_sampled_items_statistics.csv")
    sampling_stats_df.to_csv(sampling_stats_df_path, index=False)  # 保存为 sampling_statistics.csv

    print(f"采样完成，结果已保存到 {sampled_df_path} 和 {sampling_stats_df_path}")

for dataset in ["Grocery_and_Gourmet_Food", "MIND_small", "MovieLens_1M"]:
    fun(dataset)

