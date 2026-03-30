import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_CSV = PROJECT_ROOT / "data" / "train.csv"
DEFAULT_TEST_CSV = PROJECT_ROOT / "data" / "test.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

LABEL_COLS = {"click", "purchase"}
NON_FEATURE_COLS = {"sample_id", "click", "purchase"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ali-CCP preprocessing")
    parser.add_argument("--train_csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--test_csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min_freq", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)


def _feature_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    if not cols:
        raise ValueError("No feature columns found after excluding sample_id/click/purchase")
    return cols


def _to_int_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(np.int64)


def _to_label_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(np.float32)


def _build_field_mapping(train_values: pd.Series, min_freq: int) -> Dict[int, int]:
    counts = train_values.value_counts()
    frequent_ids = counts[counts >= min_freq].index.tolist()
    # 0 保留为 UNK，其他从 1 开始连续映射
    return {int(raw_id): idx + 1 for idx, raw_id in enumerate(frequent_ids)}


def _apply_mapping(values: pd.Series, mapping: Dict[int, int]) -> np.ndarray:
    mapped = values.map(mapping).fillna(0).astype(np.int64)
    return mapped.to_numpy()


def _encode_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    min_freq: int,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    train_feature_list = []
    test_feature_list = []
    vocab_sizes = []

    for col in feature_cols:
        train_col = _to_int_series(train_df[col])
        test_col = _to_int_series(test_df[col])

        mapping = _build_field_mapping(train_col, min_freq=min_freq)

        train_feature_list.append(_apply_mapping(train_col, mapping))
        test_feature_list.append(_apply_mapping(test_col, mapping))
        vocab_sizes.append(len(mapping) + 1)

    train_features = np.stack(train_feature_list, axis=1)
    test_features = np.stack(test_feature_list, axis=1)
    return train_features, test_features, vocab_sizes


def _split_train_val(num_samples: int, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)

    val_size = int(num_samples * val_ratio)
    val_size = max(1, min(val_size, num_samples - 1))

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return train_idx, val_idx


def _pack_split(
    features: np.ndarray,
    click: np.ndarray,
    purchase: np.ndarray,
    indices: np.ndarray,
) -> Dict[str, torch.Tensor]:
    click_slice = click[indices]
    purchase_slice = purchase[indices]
    return {
        "features": torch.from_numpy(features[indices]).long(),
        "click": torch.from_numpy(click_slice).float(),
        "purchase": torch.from_numpy(purchase_slice).float(),
        "cvr_mask": torch.from_numpy(click_slice).float(),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading train CSV: {args.train_csv}")
    train_df = _load_csv(args.train_csv)

    print(f"Loading test CSV:  {args.test_csv}")
    test_df = _load_csv(args.test_csv)

    for col in LABEL_COLS:
        if col not in train_df.columns or col not in test_df.columns:
            raise ValueError(f"Missing label column '{col}' in input CSV")

    feature_cols = _feature_columns(train_df)
    missing_cols = [c for c in feature_cols if c not in test_df.columns]
    if missing_cols:
        raise ValueError(f"Test CSV missing feature columns: {missing_cols[:5]}")

    print(f"Encoding {len(feature_cols)} feature fields with min_freq={args.min_freq}")
    train_features, test_features, vocab_sizes = _encode_features(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        min_freq=args.min_freq,
    )

    train_click = _to_label_series(train_df["click"]).to_numpy()
    train_purchase = _to_label_series(train_df["purchase"]).to_numpy()
    test_click = _to_label_series(test_df["click"]).to_numpy()
    test_purchase = _to_label_series(test_df["purchase"]).to_numpy()

    train_idx, val_idx = _split_train_val(
        num_samples=len(train_df),
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_blob = _pack_split(train_features, train_click, train_purchase, train_idx)
    val_blob = _pack_split(train_features, train_click, train_purchase, val_idx)
    test_blob = {
        "features": torch.from_numpy(test_features).long(),
        "click": torch.from_numpy(test_click).float(),
        "purchase": torch.from_numpy(test_purchase).float(),
        "cvr_mask": torch.from_numpy(test_click).float(),
    }

    meta = {
        "feature_names": feature_cols,
        "vocab_sizes": vocab_sizes,
        "num_features": len(feature_cols),
        "min_freq": args.min_freq,
        "seed": args.seed,
    }

    torch.save(train_blob, args.output_dir / "train.pt")
    torch.save(val_blob, args.output_dir / "val.pt")
    torch.save(test_blob, args.output_dir / "test.pt")
    torch.save(meta, args.output_dir / "meta.pt")

    print("Saved processed tensors:")
    print(f"  train: {args.output_dir / 'train.pt'} ({len(train_idx)} samples)")
    print(f"  val:   {args.output_dir / 'val.pt'} ({len(val_idx)} samples)")
    print(f"  test:  {args.output_dir / 'test.pt'} ({len(test_df)} samples)")
    print(f"  meta:  {args.output_dir / 'meta.pt'}")


if __name__ == "__main__":
    main()
