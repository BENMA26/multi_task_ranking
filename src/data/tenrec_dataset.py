"""
Tenrec 数据集模块（click + like + collect 三任务）

说明:
1) 标签统一为内部字段: click / like / collect
2) collect 支持从不同源列映射（例如 favorite 或 follow）
3) 默认使用视频场景特征，如列不存在会自动降级到可用列
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


DEFAULT_SPARSE_FEATURES = [
    "user_id",
    "item_id",
    "video_category",
    "gender",
    "age",
]
DEFAULT_DENSE_FEATURES = ["watching_times"]
INTERNAL_LABEL_COLS = ["click", "like", "collect"]
DEFAULT_COLLECT_FALLBACKS = ["follow"]


class TenrecDataset(Dataset):
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

        self.sparse_tensors = {
            c: torch.tensor(df[c].values, dtype=torch.long) for c in sparse_feature_names
        }
        self.dense_tensors = {
            c: torch.tensor(df[c].values, dtype=torch.float32) for c in dense_feature_names
        }
        self.label_tensors = {
            c: torch.tensor(df[c].values, dtype=torch.float32) for c in label_cols
        }

    def __len__(self):
        return next(iter(self.label_tensors.values())).size(0)

    def __getitem__(self, idx) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        sparse = {c: self.sparse_tensors[c][idx] for c in self.sparse_feature_names}
        dense = {c: self.dense_tensors[c][idx] for c in self.dense_feature_names}
        labels = {c: self.label_tensors[c][idx] for c in self.label_cols}
        return {**sparse, **dense}, labels


class TenrecDataModule(pl.LightningDataModule):
    def __init__(
        self,
        input_path: Optional[str] = None,
        train_path: Optional[str] = None,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        sparse_feature_names: Optional[List[str]] = None,
        dense_feature_names: Optional[List[str]] = None,
        click_label: str = "click",
        like_label: str = "like",
        collect_label: str = "favorite",
        collect_label_fallback: str = "follow",
        filter_click0_with_other_behavior: bool = False,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        batch_size: int = 1024,
        num_workers: int = 8,
        pin_memory: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.input_path = input_path
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path

        self._auto_sparse = sparse_feature_names is None
        self._auto_dense = dense_feature_names is None
        self.sparse_feature_names = sparse_feature_names or list(DEFAULT_SPARSE_FEATURES)
        self.dense_feature_names = dense_feature_names or list(DEFAULT_DENSE_FEATURES)
        self.label_cols = list(INTERNAL_LABEL_COLS)

        self.click_label = click_label
        self.like_label = like_label
        self.collect_label = collect_label
        self.collect_label_fallbacks = [
            c.strip()
            for c in (collect_label_fallback or "").split(",")
            if c.strip()
        ]
        self._resolved_collect_label: Optional[str] = None
        self.filter_click0_with_other_behavior = filter_click0_with_other_behavior

        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
            raise ValueError("train/val/test ratio 之和必须为 1.0")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed

        self.train_dataset: Optional[TenrecDataset] = None
        self.val_dataset: Optional[TenrecDataset] = None
        self.test_dataset: Optional[TenrecDataset] = None

        self.sparse_feature_dims: Dict[str, int] = {}
        self._sparse_maps: Dict[str, Dict[object, int]] = {}
        self._dense_stats: Dict[str, Tuple[float, float]] = {}
        self._prepared = False

    def _read_available_columns(self, p: Path) -> List[str]:
        if p.suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
                return pq.ParquetFile(p).schema.names
            except Exception:
                # 回退：仅用于获取列名
                return list(pd.read_parquet(p).columns)
        if p.suffix in (".csv", ".tsv"):
            sep = "\t" if p.suffix == ".tsv" else ","
            return list(pd.read_csv(p, sep=sep, nrows=0).columns)
        raise ValueError(f"不支持的文件格式: {p.suffix}")

    def _resolve_feature_columns(self, available_cols: List[str]):
        available_set = set(available_cols)

        if self._auto_sparse:
            sparse = [c for c in self.sparse_feature_names if c in available_set]
            if len(sparse) == 0:
                fallback_sparse = ["user_id", "item_id", "gender", "age"]
                sparse = [c for c in fallback_sparse if c in available_set]
            if len(sparse) == 0:
                raise ValueError("未找到可用稀疏特征列，请显式传入 --sparse_features")
            self.sparse_feature_names = sparse
            self._auto_sparse = False
        else:
            miss = [c for c in self.sparse_feature_names if c not in available_set]
            if miss:
                raise ValueError(f"稀疏特征列不存在: {miss}")

        if self._auto_dense:
            self.dense_feature_names = [c for c in self.dense_feature_names if c in available_set]
            self._auto_dense = False
        else:
            miss = [c for c in self.dense_feature_names if c not in available_set]
            if miss:
                raise ValueError(f"稠密特征列不存在: {miss}")

    def _resolve_collect_source_label(self, available_cols: List[str]) -> str:
        available_set = set(available_cols)
        if self._resolved_collect_label is not None:
            if self._resolved_collect_label not in available_set:
                raise ValueError(
                    f"collect 源列 {self._resolved_collect_label} 在当前文件中不存在"
                )
            return self._resolved_collect_label

        candidates = [self.collect_label]
        for c in self.collect_label_fallbacks or DEFAULT_COLLECT_FALLBACKS:
            if c not in candidates:
                candidates.append(c)

        for c in candidates:
            if c in available_set:
                self._resolved_collect_label = c
                return c

        raise ValueError(
            f"找不到 collect 标签列，候选={candidates}，当前列={available_cols}"
        )

    def _read_file(self, path: str) -> pd.DataFrame:
        p = Path(path)
        available_cols = self._read_available_columns(p)
        self._resolve_feature_columns(available_cols)
        collect_src = self._resolve_collect_source_label(available_cols)

        source_label_cols = {
            "click": self.click_label,
            "like": self.like_label,
            "collect": collect_src,
        }
        miss = [c for c in source_label_cols.values() if c not in available_cols]
        if miss:
            raise ValueError(f"标签列不存在: {miss}")

        cols = self.sparse_feature_names + self.dense_feature_names + list(source_label_cols.values())
        cols = list(dict.fromkeys(cols))

        if p.suffix == ".parquet":
            df = pd.read_parquet(p, columns=cols)
        elif p.suffix in (".csv", ".tsv"):
            sep = "\t" if p.suffix == ".tsv" else ","
            df = pd.read_csv(p, usecols=cols, sep=sep)
        else:
            raise ValueError(f"不支持的文件格式: {p.suffix}")

        # 统一内部标签命名
        rename_map = {v: k for k, v in source_label_cols.items()}
        out = df.rename(columns=rename_map)

        # 过滤异常样本：click=0 但 like/collect 为正
        if self.filter_click0_with_other_behavior:
            total = len(out)
            click = out["click"].astype("float32")
            like = out["like"].astype("float32")
            collect = out["collect"].astype("float32")
            invalid_mask = (click <= 0.5) & ((like > 0.5) | (collect > 0.5))
            invalid = int(invalid_mask.sum())
            if invalid > 0:
                out = out.loc[~invalid_mask].copy()
            print(
                f"🧹 过滤 click=0 且 like/collect=1 样本: {invalid:,}/{total:,} "
                f"({(invalid / max(total, 1)):.4%}) from {Path(path).name}"
            )

        return out

    def _split_by_order(self, df: pd.DataFrame):
        n = len(df)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        train_df = df.iloc[:n_train].copy()
        val_df = df.iloc[n_train:n_train + n_val].copy()
        test_df = df.iloc[n_train + n_val:].copy()
        return train_df, val_df, test_df

    def _fit_sparse_maps(self, train_df: pd.DataFrame):
        self._sparse_maps = {}
        self.sparse_feature_dims = {}
        for c in self.sparse_feature_names:
            _, uniques = pd.factorize(train_df[c], sort=False)
            # 0 预留给 OOV，从 1 开始编码
            mapping = {val: i + 1 for i, val in enumerate(uniques.tolist())}
            self._sparse_maps[c] = mapping
            self.sparse_feature_dims[c] = len(mapping) + 1

    def _apply_sparse_maps(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in self.sparse_feature_names:
            mapping = self._sparse_maps[c]
            out[c] = out[c].map(mapping).fillna(0).astype("int64")
        return out

    def _fit_dense_stats(self, train_df: pd.DataFrame):
        self._dense_stats = {}
        for c in self.dense_feature_names:
            mean = float(train_df[c].mean())
            std = float(train_df[c].std(ddof=0))
            if std < 1e-6:
                std = 1.0
            self._dense_stats[c] = (mean, std)

    def _apply_dense_norm(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in self.dense_feature_names:
            mean, std = self._dense_stats[c]
            out[c] = ((out[c].astype("float32") - mean) / std).astype("float32")
        return out

    def _cast_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in self.label_cols:
            out[c] = out[c].astype("float32")
        return out

    def prepare_datasets(self):
        if self._prepared:
            return

        if self.input_path is not None:
            full_df = self._read_file(self.input_path)
            train_df, val_df, test_df = self._split_by_order(full_df)
        else:
            if self.train_path is None or self.val_path is None or self.test_path is None:
                raise ValueError(
                    "请提供 input_path（自动顺序切分）或 train/val/test 三个路径"
                )
            train_df = self._read_file(self.train_path)
            val_df = self._read_file(self.val_path)
            test_df = self._read_file(self.test_path)

        self._fit_sparse_maps(train_df)
        self._fit_dense_stats(train_df)

        train_df = self._cast_labels(self._apply_dense_norm(self._apply_sparse_maps(train_df)))
        val_df = self._cast_labels(self._apply_dense_norm(self._apply_sparse_maps(val_df)))
        test_df = self._cast_labels(self._apply_dense_norm(self._apply_sparse_maps(test_df)))

        self.train_dataset = TenrecDataset(
            train_df, self.sparse_feature_names, self.dense_feature_names, self.label_cols
        )
        self.val_dataset = TenrecDataset(
            val_df, self.sparse_feature_names, self.dense_feature_names, self.label_cols
        )
        self.test_dataset = TenrecDataset(
            test_df, self.sparse_feature_names, self.dense_feature_names, self.label_cols
        )

        print(f"📊 Tenrec 训练集: {len(train_df):,} 条")
        print(f"📊 Tenrec 验证集: {len(val_df):,} 条")
        print(f"📊 Tenrec 测试集: {len(test_df):,} 条")
        print(f"🏷️ 标签映射: click={self.click_label}, like={self.like_label}, collect={self._resolved_collect_label}")
        print(f"🧩 稀疏特征: {self.sparse_feature_names}")
        print(f"🧩 稠密特征: {self.dense_feature_names}")
        self._prepared = True

    def setup(self, stage: Optional[str] = None):
        self.prepare_datasets()

    def _make_loader(self, ds: Dataset, shuffle: bool, drop_last: bool):
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("train_dataset 未准备好")
        return self._make_loader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("val_dataset 未准备好")
        return self._make_loader(self.val_dataset, shuffle=False, drop_last=False)

    def test_dataloader(self):
        if self.test_dataset is None:
            raise RuntimeError("test_dataset 未准备好")
        return self._make_loader(self.test_dataset, shuffle=False, drop_last=False)
