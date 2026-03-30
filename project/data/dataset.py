from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class AliCCPDataset(Dataset):
    def __init__(self, blob: Dict[str, torch.Tensor]):
        self.features = blob["features"].long()
        self.click = blob["click"].float()
        self.purchase = blob["purchase"].float()
        self.cvr_mask = blob["cvr_mask"].float()

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, idx: int):
        return (
            self.features[idx],
            self.click[idx],
            self.purchase[idx],
            self.cvr_mask[idx],
        )


def load_processed_splits(processed_dir: Path) -> Tuple[AliCCPDataset, AliCCPDataset, AliCCPDataset, Dict]:
    train_blob = torch.load(processed_dir / "train.pt", map_location="cpu")
    val_blob = torch.load(processed_dir / "val.pt", map_location="cpu")
    test_blob = torch.load(processed_dir / "test.pt", map_location="cpu")
    meta = torch.load(processed_dir / "meta.pt", map_location="cpu")

    return (
        AliCCPDataset(train_blob),
        AliCCPDataset(val_blob),
        AliCCPDataset(test_blob),
        meta,
    )


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )
