import pandas as pd

'''
    看看测试集中item的类别分布
'''

# 1. 读取 test.csv 和 item_classes.csv
test_df = pd.read_csv('data\\Grocery_and_Gourmet_Food\\test.csv', sep='\t')  # 请替换为实际路径
item_classes_df = pd.read_csv('item_classes.csv')  # 读取 item_classes.csv，包含每个 item_id 的类别

# 2. 创建 item_id 到 class 的映射
item_to_class = dict(zip(item_classes_df['item_id'], item_classes_df['class']))

# 3. 将 test_df 的 item_id 映射到 class
test_df['class'] = test_df['item_id'].map(item_to_class)

# 4. 统计每个类的数量及每个类内的 item_id 出现次数
class_statistics = test_df.groupby('class').agg(
    total_items=('item_id', 'count'),        # 计算每个类的 item_id 数量
    total_counts=('item_id', lambda x: x.map(test_df['item_id'].value_counts()).sum())  # 计算该类 item_id 出现次数之和
)

# 5. 打印每个类的统计数据
print(class_statistics)