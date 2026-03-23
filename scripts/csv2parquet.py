import os
import numpy as np
import pandas as pd

train_path = "/work/home/maben/project/rec_sys/projects/multi_task_ranking/dataset/sample_skeleton_train.csv"
test_path = "/work/home/maben/project/rec_sys/projects/multi_task_ranking/dataset/sample_skeleton_test.csv"

train_feature_path = "/work/home/maben/project/rec_sys/projects/multi_task_ranking/dataset/common_features_train.csv"
test_feature_path = "/work/home/maben/project/rec_sys/projects/multi_task_ranking/dataset/common_features_test.csv"

path_list = [
    train_path,
    test_path,
    train_feature_path,
    test_feature_path
]

for path in path_list:
    print(f"processing {path.split('/')[-1]}")
    df = pd.read_csv(path,header=None)
    df.columns = df.columns.astype(str)
    df.to_parquet(path[:-4]+".parquet", index=False)
    print(f"processed {path.split('/')[-1]}")