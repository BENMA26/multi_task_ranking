"""
数据集 - 支持 Sparse, Dense三种特征类型
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pandas as pd
import numpy as np

from src.models.features import SparseFeature, DenseFeature, SequenceFeature

'''
排序数据集
'''
class RankDataset(Dataset):

    def __init__(
        self,
        df: pd.DataFrame,
        sparse_feature_names: List[str],
        dense_feature_names: List[str],
        label_cols: List[str],
    ):
        self.sparse_feature_names = sparse_feature_names
        self.dense_feature_names = dense_feature_names
        self.label_cols = label_cols

        # 稀疏特征 → long
        self.sparse_tensors = {
            col: torch.tensor(df[col].values, dtype=torch.long)
            for col in sparse_feature_names
        }

        # 稠密特征 → float
        self.dense_tensors = {
            col: torch.tensor(df[col].values, dtype=torch.float)
            for col in dense_feature_names
        }

        # 标签 → float
        self.label_tensors = {
            col: torch.tensor(df[col].values, dtype=torch.float)
            for col in label_cols
        }

    def __len__(self) -> int:
        return next(iter(self.label_tensors.values())).size(0)

    def __getitem__(self, idx) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        sparse = {col: self.sparse_tensors[col][idx] for col in self.sparse_feature_names}
        dense  = {col: self.dense_tensors[col][idx]  for col in self.dense_feature_names}
        labels = {col: self.label_tensors[col][idx]  for col in self.label_cols}

        # 将 sparse 和 dense 合并为一个 features dict 返回
        # 模型侧通过 sparse_feature_names / dense_feature_names 分别取用
        features = {**sparse, **dense}
        return features, labels


class RankDataModule(pl.LightningDataModule):

    def __init__(
        self,
        sparse_feature_names: List[str],
        dense_feature_names: List[str],
        label_cols: List[str],
        # ---- 数据来源（二选一） ----
        train_df: Optional[pd.DataFrame] = None,
        val_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        train_path: Optional[str] = None,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        # ---- 训练参数 ----
        batch_size: int = 2048,
        val_batch_size: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: bool = True,
        val_ratio: float = 0.1,
        seed: int = 42,
        train_subset_frac: float = 1.0,
        train_subset_seed: int = 42,
        train_subset_stratify: bool = False,
    ):
        super().__init__()

        self.sparse_feature_names = sparse_feature_names
        self.dense_feature_names  = dense_feature_names
        self.label_cols           = label_cols

        # 数据来源
        self._train_df = train_df
        self._val_df   = val_df
        self._test_df  = test_df
        self.train_path = train_path
        self.val_path   = val_path
        self.test_path  = test_path

        # 训练参数
        self.batch_size     = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers    = num_workers
        self.pin_memory     = pin_memory
        self.val_ratio      = val_ratio
        self.seed           = seed
        self.train_subset_frac = float(train_subset_frac)
        self.train_subset_seed = int(train_subset_seed)
        self.train_subset_stratify = bool(train_subset_stratify)

        if not (0.0 < self.train_subset_frac <= 1.0):
            raise ValueError("train_subset_frac 必须在 (0, 1] 区间")

        # Dataset 占位
        self.train_dataset = None
        self.val_dataset   = None
        self.test_dataset  = None

    def _read_file(self, path: str) -> pd.DataFrame:
        path = Path(path)
        # 只读需要的列，节省内存
        cols = self.sparse_feature_names + self.dense_feature_names + self.label_cols
        if path.suffix == '.parquet':
            return pd.read_parquet(path, columns=cols)
        elif path.suffix in ('.csv', '.tsv'):
            sep = '\t' if path.suffix == '.tsv' else ','
            return pd.read_csv(path, usecols=cols, sep=sep)
        else:
            raise ValueError(f"不支持的文件格式: {path.suffix}")

    def _make_dataset(self, df: pd.DataFrame) -> RankDataset:
        return RankDataset(
            df=df,
            sparse_feature_names=self.sparse_feature_names,
            dense_feature_names=self.dense_feature_names,
            label_cols=self.label_cols,
        )

    def _make_dataloader(self, dataset: RankDataset, shuffle: bool, drop_last: bool) -> DataLoader:
        bs = self.batch_size if shuffle else self.val_batch_size
        return DataLoader(
            dataset,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            drop_last=drop_last,
        )

    def _apply_train_subset(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """可选地对训练集做子采样，用于快速参数选择。"""
        frac = self.train_subset_frac
        if frac >= 1.0:
            return train_df

        target_n = max(1, int(len(train_df) * frac))
        rs = self.train_subset_seed

        if not self.train_subset_stratify:
            return train_df.sample(n=target_n, random_state=rs).reset_index(drop=True)

        stratify_cols = [c for c in ("click", "purchase") if c in train_df.columns]
        if len(stratify_cols) == 0:
            return train_df.sample(n=target_n, random_state=rs).reset_index(drop=True)

        pieces = []
        grouped = train_df.groupby(stratify_cols, dropna=False, sort=False)
        for _, group_df in grouped:
            n = max(1, int(round(len(group_df) * frac)))
            n = min(n, len(group_df))
            pieces.append(group_df.sample(n=n, random_state=rs))

        subset_df = pd.concat(pieces, axis=0)

        # 纠偏到精确目标条数，保证不同实验间可比性
        if len(subset_df) > target_n:
            subset_df = subset_df.sample(n=target_n, random_state=rs)
        elif len(subset_df) < target_n:
            remain_df = train_df.drop(subset_df.index, errors="ignore")
            add_n = min(target_n - len(subset_df), len(remain_df))
            if add_n > 0:
                subset_df = pd.concat(
                    [subset_df, remain_df.sample(n=add_n, random_state=rs)],
                    axis=0,
                )

        return subset_df.sample(frac=1.0, random_state=rs).reset_index(drop=True)

    def setup(self, stage: Optional[str] = None):

        if stage in ('fit', None):
            # 获取训练数据
            if self._train_df is not None:
                train_df = self._train_df
            elif self.train_path is not None:
                train_df = self._read_file(self.train_path)
            else:
                raise ValueError("必须提供 train_df 或 train_path")

            # 获取验证数据
            if self._val_df is not None:
                val_df = self._val_df
            elif self.val_path is not None:
                val_df = self._read_file(self.val_path)
            else:
                val_df   = train_df.sample(frac=self.val_ratio, random_state=self.seed)
                train_df = train_df.drop(val_df.index).reset_index(drop=True)
                val_df   = val_df.reset_index(drop=True)

            original_train_size = len(train_df)
            train_df = self._apply_train_subset(train_df)

            print(f"📊 训练集: {len(train_df):,} 条")
            if self.train_subset_frac < 1.0:
                print(
                    f"📊 训练集子采样: {len(train_df):,}/{original_train_size:,} "
                    f"({100.0 * len(train_df) / max(1, original_train_size):.2f}%)"
                )
            print(f"📊 验证集: {len(val_df):,} 条")

            self.train_dataset = self._make_dataset(train_df)
            self.val_dataset   = self._make_dataset(val_df)

            # 释放 DataFrame 引用，节省内存
            self._train_df = None
            self._val_df   = None

        if stage in ('test', None):
            if self._test_df is not None:
                test_df = self._test_df
            elif self.test_path is not None:
                test_df = self._read_file(self.test_path)
            else:
                return  # 没有测试集，跳过

            print(f"📊 测试集: {len(test_df):,} 条")
            self.test_dataset = self._make_dataset(test_df)
            self._test_df = None

    def train_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.train_dataset, shuffle=True,  drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.val_dataset,   shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("没有测试数据集，请提供 test_df 或 test_path")
        return self._make_dataloader(self.test_dataset,  shuffle=False, drop_last=False)
