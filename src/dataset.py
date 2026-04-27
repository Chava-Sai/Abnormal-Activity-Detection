"""
UCF-Crime dataset utilities for violence detection.

Supports both the original flat 2048-dim `_i3d.npy` layout and local recursive
feature directories with generic `.npy` names such as `Abuse001_x264.npy`.
"""

from __future__ import annotations

import os
import random
import re
from collections import defaultdict
from typing import Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from feature_utils import (
    build_feature_index,
    infer_feature_dim,
    load_feature_array,
    resolve_feature_path,
    scan_feature_files,
    strip_feature_suffix,
    temporal_resize,
)

# Violence categories we focus on (subset of UCF-Crime)
VIOLENCE_CATEGORIES = {
    "Abuse": 1,
    "Fighting": 2,
    "Shooting": 3,
    "Explosion": 4,
    "Robbery": 5,
    "Riot": 6,
    # label 0 = Normal
}

VIOLENCE_CLASS_NAMES = set(VIOLENCE_CATEGORIES.keys())

# UCF-Crime has these non-violence anomaly classes — we skip them in training
NON_VIOLENCE_ANOMALIES = {
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "RoadAccidents",
    "Shoplifting",
    "Stealing",
    "Vandalism",
}

Sample = Tuple[str, int, int]


def get_category_from_filename(filename: str) -> str:
    """
    Extract category name from filenames like:
      - Abuse001_x264.npy
      - Abuse001_x264_i3d.npy
      - Normal_Videos_001_i3d.npy
    """
    stem = strip_feature_suffix(filename)
    category = re.sub(r"_?\d+$", "", stem)
    if category.startswith("Normal"):
        return "Normal"
    return category


def _iter_feature_references(list_file: Optional[str], feature_dir: str) -> List[str]:
    """
    Return raw feature references from a `.list` file or recursive directory scan.
    `list_file='auto'` scans `feature_dir`.
    """
    if not list_file or list_file == "auto":
        return scan_feature_files(feature_dir)

    if not os.path.exists(list_file):
        raise FileNotFoundError(
            f"list file not found: {list_file}. Use --train_list auto to scan {feature_dir} recursively."
        )

    with open(list_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _build_samples(
    references: Iterable[str],
    feature_dir: str,
    mode: str,
    violence_only: bool,
    verbose: bool = True,
) -> List[Sample]:
    """Resolve feature references into `(filepath, label, cat_id)` tuples."""
    index = build_feature_index(feature_dir)
    samples: List[Sample] = []
    seen = set()
    missing = 0

    for ref in references:
        filepath = resolve_feature_path(ref, feature_dir, index)
        if filepath is None:
            missing += 1
            continue

        if filepath in seen:
            continue
        seen.add(filepath)

        category = get_category_from_filename(filepath)
        if category == "Normal":
            label, cat_id = 0, 0
        elif category in VIOLENCE_CATEGORIES:
            label, cat_id = 1, VIOLENCE_CATEGORIES[category]
        else:
            if violence_only and mode == "train":
                continue
            if category in NON_VIOLENCE_ANOMALIES:
                label, cat_id = 1, 0
            else:
                continue

        samples.append((filepath, label, cat_id))

    if verbose and missing:
        print(f"[{mode}] Skipped {missing} list entries with no matching local feature file")

    return samples


def _summarize_samples(mode: str, samples: List[Sample], tag: str = "") -> None:
    n_anom = sum(1 for _, label, _ in samples if label == 1)
    n_norm = sum(1 for _, label, _ in samples if label == 0)
    suffix = f" {tag}" if tag else ""
    print(f"[{mode}] Loaded {len(samples)} samples ({n_anom} anomalous, {n_norm} normal){suffix}")


def make_train_val_split(
    list_file: Optional[str],
    feature_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[List[Sample], List[Sample]]:
    """
    Create a stratified train/val split grouped by category name.
    """
    random.seed(seed)

    references = _iter_feature_references(list_file, feature_dir)
    samples = _build_samples(
        references,
        feature_dir=feature_dir,
        mode="train",
        violence_only=True,
        verbose=True,
    )

    by_cat = defaultdict(list)
    for sample in samples:
        by_cat[get_category_from_filename(sample[0])].append(sample)

    train_samples: List[Sample] = []
    val_samples: List[Sample] = []

    for category, items in sorted(by_cat.items()):
        random.shuffle(items)
        if len(items) <= 1:
            train_samples.extend(items)
            continue

        n_val = max(1, int(round(len(items) * val_ratio)))
        n_val = min(n_val, len(items) - 1)
        val_samples.extend(items[:n_val])
        train_samples.extend(items[n_val:])

    return train_samples, val_samples


class UCFCrimeDataset(Dataset):
    """
    Dataset for UCF-Crime violence detection.
    """

    def __init__(
        self,
        feature_dir: str,
        list_file: Optional[str] = "auto",
        mode: str = "train",
        num_segments: Optional[int] = 32,
        violence_only: bool = True,
        _override_samples: Optional[List[Sample]] = None,
    ):
        self.feature_dir = feature_dir
        self.mode = mode
        self.num_segments = num_segments
        self.violence_only = violence_only

        if _override_samples is not None:
            self.samples = list(_override_samples)
            _summarize_samples(self.mode, self.samples, tag="[from split]")
        else:
            self.samples = self._load_list(list_file)

        if not self.samples:
            raise ValueError(
                f"[{self.mode}] No usable feature files found in {feature_dir}. "
                "Check the directory path, list file, and filename pattern."
            )

        self.input_dim = infer_feature_dim(path for path, _, _ in self.samples)
        if self.input_dim is None:
            raise ValueError(f"[{self.mode}] Failed to infer feature dimensionality")

    def _load_list(self, list_file: Optional[str]) -> List[Sample]:
        references = _iter_feature_references(list_file, self.feature_dir)
        samples = _build_samples(
            references,
            feature_dir=self.feature_dir,
            mode=self.mode,
            violence_only=self.violence_only,
            verbose=True,
        )
        _summarize_samples(self.mode, samples)
        return samples

    def _load_features(self, filepath: str):
        return load_feature_array(filepath)

    def _temporal_sample(self, feat):
        return temporal_resize(feat, self.num_segments)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label, cat_id = self.samples[idx]
        feat = self._load_features(filepath)

        if self.num_segments is not None:
            feat = self._temporal_sample(feat)

        return {
            "features": torch.tensor(feat, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "cat_id": torch.tensor(cat_id, dtype=torch.long),
            "filepath": filepath,
        }


class MILDataset(Dataset):
    """
    MIL-style dataset that returns anomalous/normal pairs for ranking loss.
    """

    def __init__(self, base_dataset: UCFCrimeDataset):
        self.anomalous = [(f, l, c) for f, l, c in base_dataset.samples if l == 1]
        self.normal = [(f, l, c) for f, l, c in base_dataset.samples if l == 0]
        self.base = base_dataset

        if not self.anomalous:
            raise ValueError("[MIL] Training set contains no anomalous videos")
        if not self.normal:
            raise ValueError(
                "[MIL] Training set contains no normal videos. "
                "Step 1 cannot run until you extract Normal/Normal_Videos features."
            )

        print(f"[MIL] {len(self.anomalous)} anomalous, {len(self.normal)} normal")

    def __len__(self):
        return max(len(self.anomalous), len(self.normal))

    def __getitem__(self, idx):
        a_idx = idx % len(self.anomalous)
        n_idx = idx % len(self.normal)

        a_path, _, a_cat = self.anomalous[a_idx]
        n_path, _, _ = self.normal[n_idx]

        a_feat = self.base._load_features(a_path)
        n_feat = self.base._load_features(n_path)

        if self.base.num_segments is not None:
            a_feat = self.base._temporal_sample(a_feat)
            n_feat = self.base._temporal_sample(n_feat)

        return {
            "anom_features": torch.tensor(a_feat, dtype=torch.float32),
            "norm_features": torch.tensor(n_feat, dtype=torch.float32),
            "anom_cat": torch.tensor(a_cat, dtype=torch.long),
        }


class InMemoryDataset(Dataset):
    """
    Lightweight dataset backed by a pre-built sample list.
    """

    def __init__(
        self,
        samples: List[Sample],
        feature_dir: Optional[str] = None,
        num_segments: Optional[int] = 32,
    ):
        resolved_samples = []
        for filepath, label, cat_id in samples:
            if os.path.isabs(filepath):
                resolved_path = filepath
            elif feature_dir:
                resolved_path = os.path.join(feature_dir, filepath)
            else:
                resolved_path = filepath
            resolved_samples.append((resolved_path, label, cat_id))

        self.samples = resolved_samples
        self.feature_dir = feature_dir
        self.num_segments = num_segments
        self.input_dim = infer_feature_dim(path for path, _, _ in self.samples)
        if self.input_dim is None:
            raise ValueError("[InMemory] Failed to infer feature dimensionality")
        _summarize_samples("InMemory", self.samples)

    def _load_features(self, filepath: str):
        return load_feature_array(filepath)

    def _temporal_sample(self, feat):
        return temporal_resize(feat, self.num_segments)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label, cat_id = self.samples[idx]
        feat = self._temporal_sample(self._load_features(filepath))
        return {
            "features": torch.tensor(feat, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "cat_id": torch.tensor(cat_id, dtype=torch.long),
            "filepath": filepath,
        }


def build_dataloaders(
    train_feature_dir: str,
    test_feature_dir: Optional[str],
    train_list: Optional[str],
    test_list: Optional[str],
    num_segments: int = 32,
    batch_size: int = 32,
    num_workers: int = 4,
    val_ratio: float = 0.2,
    use_val_split: bool = True,
    seed: int = 42,
    pin_memory: bool = False,
):
    """
    Build train and evaluation dataloaders.
    """
    if use_val_split:
        train_samples, val_samples = make_train_val_split(
            train_list,
            train_feature_dir,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_ds = UCFCrimeDataset(
            train_feature_dir,
            train_list,
            mode="train",
            num_segments=num_segments,
            violence_only=True,
            _override_samples=train_samples,
        )
        eval_ds = InMemoryDataset(val_samples, num_segments=num_segments)
    else:
        if not test_feature_dir:
            raise ValueError("--test_dir is required when --no_val_split is used")
        train_ds = UCFCrimeDataset(
            train_feature_dir,
            train_list,
            mode="train",
            num_segments=num_segments,
            violence_only=True,
        )
        eval_ds = UCFCrimeDataset(
            test_feature_dir,
            test_list,
            mode="test",
            num_segments=num_segments,
            violence_only=False,
        )

    mil_ds = MILDataset(train_ds)

    train_loader = DataLoader(
        mil_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, eval_loader, train_ds, eval_ds


if __name__ == "__main__":
    import sys

    feature_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    list_file = sys.argv[2] if len(sys.argv) > 2 else "auto"
    ds = UCFCrimeDataset(feature_dir, list_file, mode="test", num_segments=32, violence_only=False)

    print(f"\nDataset size: {len(ds)}")
    item = ds[0]
    print(f"features shape : {item['features'].shape}")
    print(f"label          : {item['label']}")
    print(f"cat_id         : {item['cat_id']}")
    print(f"filepath       : {item['filepath']}")
