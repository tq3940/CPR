import pandas as pd
import os
'''
    根据item_classes.csv中每个item分组，对测试集推荐结果计算每组的分布
'''
LOG_PATH = "eval_log"
DATA_PATH = "rechorus\\eval_data"

def fun(model, dataset):
    # 1. 读取 item_classes.csv 和 item_counts.csv
    item_classes_df_path = os.path.join(DATA_PATH, dataset, 'item_classes.csv')
    item_counts_df_path = os.path.join(LOG_PATH, model, model+"__"+dataset, f"rec-{model}-test-item_counts.csv")

    item_classes_df = pd.read_csv(item_classes_df_path)  # 替换为你的 item_classes.csv 路径
    item_counts_df = pd.read_csv(item_counts_df_path)  # 替换为你的 item_counts.xlsx 路径

    # 2. 合并两个数据表
    # 使用 merge 按照 item_id 合并，保留 item_id、class 和 Count
    merged_df = pd.merge(item_counts_df, item_classes_df, how='inner', left_on='Item', right_on='item_id')

    # 3. 按类统计数量
    class_totals = merged_df.groupby('class')['Count'].sum().reset_index()

    # 4. 导出结果
    class_totals.columns = ['Class', 'Total_Count']  # 重命名列

    class_totals_path = os.path.join(LOG_PATH, model, model+"__"+dataset, f"rec-{model}-test-class_totals.csv")
    class_totals.to_csv(class_totals_path, index=False)  # 保存为 class_totals.csv

    print(f"计算完成，结果已保存到 {class_totals_path}")

# for model in ["BPRMF", "ComiRec", "ComiRecCPR", "CPRMF", "LightGCN", "LightGCNCPR"]:
#     for dataset in ["Grocery_and_Gourmet_Food", "MIND_small", "MovieLens_1M"]:
#         file_path = os.path.join(LOG_PATH, model, model+"__"+dataset, f"rec-{model}-test-class_totals.xlsx")
#         if os.path.exists(file_path):
#             os.remove(file_path)
#             print(f"{file_path} 已被删除")
#         else:
#             print(f"{file_path} 文件不存在")

# for model in ["BPRMF", "ComiRec", "ComiRecCPR", "CPRMF", "LightGCN", "LightGCNCPR"]:
#     for dataset in ["Grocery_and_Gourmet_Food", "MIND_small", "MovieLens_1M"]:
#         fun(model, dataset)

def fun2(dataset):
    all_df = pd.DataFrame()
    for model in ["BPRMF", "ComiRec", "ComiRecCPR", "CPRMF", "LightGCN", "LightGCNCPR"]:
        class_totals_path = os.path.join(LOG_PATH, model, model+"__"+dataset, f"rec-{model}-test-class_totals.csv")
        df = pd.read_csv(class_totals_path)
        all_df[f"{model}"] = df["Total_Count"]
    all_df.insert(0, 'Class', df['Class'])
    all_df_path = f"eval_result\\{dataset}_test_class_totals.xlsx"
    all_df.to_excel(all_df_path, index=False)
    print(f"计算完成，结果已保存到 {all_df_path}")


for dataset in ["Grocery_and_Gourmet_Food", "MIND_small", "MovieLens_1M"]:
    fun2( dataset)

