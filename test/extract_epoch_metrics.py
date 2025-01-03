import re
import pandas as pd
import os
'''
    将log中每一轮数据导出为epoch_metrics.xlsx
'''
model_names = ["BPRMF", "CPRMF", "ComiRec", "ComiRecCPR", "LightGCN", "LightGCNCPR"]
dataset_names = ["Grocery_and_Gourmet_Food", "MIND_small", "MovieLens_1M"]
# dataset_names = ["Grocery_and_Gourmet_Food"]

for dataset in dataset_names:
    df_list = []
    for model in model_names:
        # Read log file
        log_file_dir = os.path.join("log", model)

        file_name_pattern = re.compile(f".*{re.escape(dataset)}.*\.txt$")
        file_name = next(filter(file_name_pattern.match, os.listdir(log_file_dir)), None)
        log_file_path = os.path.join(log_file_dir, file_name)
        
        with open(log_file_path, 'r') as file:
            log_text = file.read()

        # Regular expression to extract metrics for each epoch
        pattern = r"Epoch (\d+)\s+loss=([0-9\.]+).*?dev=\(HR@2:([0-9\.]+),NDCG@2:([0-9\.]+)\)"

        # Extract matches
        matches = re.findall(pattern, log_text)

        # Parse the metrics into a list of dictionaries
        data = []
        for epoch, loss, hr2, ndcg2 in matches:
            print(f"[Match]: epoch:{epoch}, NDCG@2:{ndcg2}")

            data.append({"Epoch": int(epoch), f"NDCG@2": float(ndcg2)})

        # Create a DataFrame
        df = pd.DataFrame(data)
        df_list.append(df)

    # 找到最长的Epoch列
    max_epoch_df = max(df_list, key=lambda df: len(df['Epoch']))
    base_epoch = max_epoch_df['Epoch'].sort_values().reset_index(drop=True)

    # 初始化新的数据框
    merged_df = pd.DataFrame({'Epoch': base_epoch})

    # 按顺序合并每个数据框并重命名列
    for df, model_name in zip(df_list, model_names):
        merged_df = merged_df.merge(
            df[['Epoch', 'NDCG@2']].rename(columns={'NDCG@2': model_name}),
            on='Epoch',
            how='left'
        )
    
    merged_df.to_excel(f"epoch_metrics\\{dataset}_all_models_NDCG@@.xlsx", index=False)
        # print("Epoch metrics (loss, HR@2, NDCG@2) have been extracted and saved to 'epoch_metrics.xlsx'.")
