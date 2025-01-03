import re
import pandas as pd
import os

'''
    将log中最终测试结果导出为hr_ndcg_metrics.xlsx
'''
for dataset in ["Grocery_and_Gourmet_Food", "MIND_small", "MovieLens_1M"]:
    df_dataset = pd.DataFrame()
    for model in ["BPRMF", "CPRMF", "ComiRec", "ComiRecCPR", "LightGCN", "LightGCNCPR"]:
        # Read log file
        log_file_dir = os.path.join("eval_log", model)

        file_name_pattern = re.compile(f".*{re.escape(dataset)}.*\.txt$")
        file_name = next(filter(file_name_pattern.match, os.listdir(log_file_dir)), None)
        log_file_path = os.path.join(log_file_dir, file_name)
        
        with open(log_file_path, 'r') as file:
            log_text = file.read()

        # Regular expression to extract only the final Dev and Test metrics
        pattern = r"(Test After Training): \((.*?)\)"

        # Extract matches
        matches = re.findall(pattern, log_text)

        # Parse the metrics into a list of dictionaries
        data = []
        for phase, metrics in matches:
            print(f"[{model} {dataset} Match]: {metrics}")
            metric_pairs = metrics.split(",")
            for pair in metric_pairs:
                key, value = pair.split(":")
                data.append({"Metric": key, "Value": float(value)})

        # Create a DataFrame
        df = pd.DataFrame(data)

        # # Sort by Metric to ensure HR is before NDCG and numbers are in ascending order
        df['Metric'] = pd.Categorical(df['Metric'], ordered=True, categories=sorted(df['Metric'].unique(), key=lambda x: (x.split("@")[0], int(x.split("@")[1]))))
        df_sorted = df.sort_values(by=['Metric']).reset_index(drop=True)

        # print(df_sorted)
        df_dataset[model] = df_sorted['Value']

    df_dataset.insert(0, 'Metric', df_sorted['Metric'])
    df_dataset_path = f"eval_result\\{dataset}_all_models_eval_result.xlsx"
    df_dataset.to_excel(df_dataset_path, index=False)
    print(f"Final Test HR and NDCG metrics have been extracted and saved to {df_dataset_path}.")


