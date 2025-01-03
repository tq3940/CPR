
# 机器学习大作业--使用[reChorus](https://github.com/THUwangcy/ReChorus)复现论文：[Cross Pairwise Ranking for Unbiased Item Recommendation](https://arxiv.org/abs/2204.12176)

## 具体代码：

[./rechorus/src/models/BaseModel.py--CPRModel](https://github.com/tq3940/CPR/blob/main/rechorus/src/models/BaseModel.py#L296)：CPR基类模型，继承自general model，实现了动态采样、CPRLoss等功能

### 具体CPR模型：

*  [./rechorus/src/models/general/LigthGCN.py--LightGCNCPR](https://github.com/tq3940/CPR/blob/main/rechorus/src/models/general/LightGCN.py#L162)

*  [./rechorus/src/models/general/CPRMF.py](https://github.com/tq3940/CPR/blob/main/rechorus/src/models/general/CPRMF.py)

* [./rechorus/src/models/sequential/ComiRec.py--ComiRecCPR](https://github.com/tq3940/CPR/blob/main/rechorus/src/models/sequential/ComiRec.py#L96)

### Reader、Runner模块：

* [./rechorus/src/helpers/CPRReader.py](https://github.com/tq3940/CPR/blob/main/rechorus/src/helpers/CPRReader.py)

* [./rechorus/src/helpers/CPRRunner.py](https://github.com/tq3940/CPR/blob/main/rechorus/src/helpers/CPRRunner.py)


## 其他文件：

[train_log](https://github.com/tq3940/CPR/tree/main/train_log)：训练过程日志记录

[epoch_metrics](https://github.com/tq3940/CPR/tree/main/epoch_metrics)：训练过程在验证集上NDCG@2指标导出记录

[eval_log](https://github.com/tq3940/CPR/tree/main/eval_log)：评测过程日志记录

[model](https://github.com/tq3940/CPR/tree/main/model)：已训练模型

test：评测用代码
    
* [count_train_set_items.py](https://github.com/tq3940/CPR/blob/main/test/count_test_rec_items.py)：对每个数据集的训练集分组

* [sample_test_set.py](https://github.com/tq3940/CPR/blob/main/test/sample_test_set.py)：对原测试集重采样，形成新测试集（./rechorus/eval_data）

* [count_test_rec_items.py](https://github.com/tq3940/CPR/blob/main/test/count_test_rec_items.py)：对模型在测试集上推荐结果前10计数

* [test_rec_items_distribution.py](https://github.com/tq3940/CPR/blob/main/test/test_rec_items_distribution.py)：统计测试结果中每一组分布


* [extract_epoch_metrics.py](https://github.com/tq3940/CPR/blob/main/test/extract_epoch_metrics.py)：导出训练过程每一轮指标

* [extract_final_metrics.py](https://github.com/tq3940/CPR/blob/main/test/extract_final_metrics.py)：导出模型在测试集上评估指标


[eval_result](https://github.com/tq3940/CPR/tree/main/eval_result)：模型在测试集上评估指标及推荐结果的分布


[./rechorus/data](https://github.com/tq3940/CPR/tree/main/rechorus/data)：训练用数据集

[./rechorus/eval_data](https://github.com/tq3940/CPR/tree/main/rechorus/eval_data)：重采样后用来评测的数据集

[./rechorus/run_script](https://github.com/tq3940/CPR/tree/main/rechorus/run_script)：训练用脚本文件（liunx端 使用GPU）

[./rechorus/eval_script](https://github.com/tq3940/CPR/tree/main/rechorus/eval_script)：测试用脚本文件（win端 使用CPU）

[./rechorus/src/main.py](https://github.com/tq3940/CPR/blob/main/rechorus/src/main.py)：训练主函数

[./rechorus/src/main_CPU.py](https://github.com/tq3940/CPR/blob/main/rechorus/src/main_CPU.py)：训练主函数 使用CPU

[./rechorus/src/eval_CPU.py](https://github.com/tq3940/CPR/blob/main/rechorus/src/eval_CPU.py)：评测主函数 使用CPU

