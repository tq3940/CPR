import pandas as pd
from collections import Counter
from tqdm import tqdm
import os
'''
    根据train数据对item分组，每组的item度和大致相等
'''
PATH = "rechorus\data"


# 1. 读取 train.csv，统计 item_id 的出现次数
def fun(train_set):
    train_set_path = os.path.join(PATH, train_set, "train.csv")

    test_df = pd.read_csv(train_set_path, sep='\t')  # 替换为你的 test.csv 路径
    item_counts_test = Counter(test_df['item_id'])

    # 2. 将 item_id 按出现次数从小到大排序
    sorted_items = sorted(item_counts_test.items(), key=lambda x: x[1])

    # 3. 计算所有 item_id 出现次数的总和
    total_count = sum(count for _, count in sorted_items)
    threshold = total_count / 4  # 每组的目标总和

    # 4. 计算前缀和
    prefix_sums = [0]
    for _, count in sorted_items:
        prefix_sums.append(prefix_sums[-1] + count)

    # 5. 遍历分组
    item_classes = {}
    group_start = 0
    current_class = 1
    for i, (item, count) in tqdm(enumerate(sorted_items), desc="Assigning classes"):
        # 计算当前区间的总和
        current_sum = prefix_sums[i + 1] - prefix_sums[group_start]

        # 如果超过阈值，分组
        if current_sum > threshold and current_class < 4:  # 确保不超过 4 组
            group_start = i  # 当前 item 开始新的一组
            current_class += 1

        # 分配类号
        item_classes[item] = current_class

    # 6. 保存每个 item_id 的组号为 CSV 文件
    output_df = pd.DataFrame(item_classes.items(), columns=['item_id', 'class'])
    output_path = os.path.join(PATH, train_set, 'item_classes.csv')
    output_df.to_csv(output_path, index=False)  # 保存为 item_classes.csv

    # 7. 统计每组的总个数和总次数
    group_statistics = {cls: {'total_items': 0, 'total_counts': 0} for cls in range(1, 5)}
    for item, cls in item_classes.items():
        group_statistics[cls]['total_items'] += 1
        group_statistics[cls]['total_counts'] += item_counts_test[item]

    # 转换为 DataFrame 保存
    class_stats_df = pd.DataFrame([
        {'class': cls, 'total_items': stats['total_items'], 'total_counts': stats['total_counts']}
        for cls, stats in group_statistics.items()
    ])
    class_stats_path = os.path.join(PATH, train_set, 'class_statistics.csv')

    class_stats_df.to_csv(class_stats_path, index=False)  # 保存为 class_statistics.csv

    print(f"分类完成，结果已保存到 {output_path} 和 {class_stats_path}")

for train_set in ["Grocery_and_Gourmet_Food", "MIND_small", "MovieLens_1M"]:
    fun(train_set)