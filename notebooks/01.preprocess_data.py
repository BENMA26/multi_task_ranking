import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import subprocess

    return (subprocess,)


@app.cell
def _():
    import os
    import numpy as np
    import pandas as pd

    return (pd,)


@app.cell
def _(subprocess):
    #! nvidia-smi
    subprocess.call(['nvidia-smi'])
    return


@app.cell
def _():
    train_path= "/work/home/maben/project/rec_sys/projects/multi_task_ranking/dataset/sample_skeleton_train.parquet"
    return (train_path,)


@app.cell
def _(pd, train_path):
    train_df = pd.read_parquet(train_path)
    return


if __name__ == "__main__":
    app.run()
