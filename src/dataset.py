"""
UCF-Crime Dataset loader for violence event detection.
Features format: (T, 10, 2048) — T segments, 10-crop, 2048-dim I3D features.
"""

import os
import re
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

# Violence categories we focus on (subset of UCF-Crime)
VIOLENCE_CATEGORIES = {
    'Abuse':     1,
    'Fighting':  2,
    'Shooting':  3,
    'Explosion': 4,
    'Robbery':   5,
    'Riot':      6,
    # label 0 = Normal
}

# All UCF-Crime anomaly classes (for filtering)
VIOLENCE_CLASS_NAMES = set(VIOLENCE_CATEGORIES.keys())

# UCF-Crime has these non-violence anomaly classes — we skip them in training
NON_VIOLENCE_ANOMALIES = {
    'Arrest', 'Arson', 'Assault', 'Burglary', 'RoadAccidents',
    'Shoplifting', 'Stealing', 'Vandalism'
}


def get_category_from_filename(filename: str) -> str:
    """
    Extract category name from filename like 'Abuse001_x264_i3d.npy'.
    Returns category string or 'Normal' or 'Other' (non-violence anomaly).

    Uses regex to correctly strip trailing digits — avoids rstrip() bug where
    rstrip('0123456789_') applied to extensions would strip nothing or too much.
    """
    basename = os.path.basename(filename)
    # Remove known suffixes
    basename = re.sub(r'_x264_i3d\.npy$', '', basename)
    basename = re.sub(r'_i3d\.npy$', '', basename)
    basename = re.sub(r'\.npy$', '', basename)
    # Remove trailing digits (e.g., 'Abuse001' -> 'Abuse', 'Normal_Videos_001' -> 'Normal_Videos')
    category = re.sub(r'_?\d+$', '', basename)
    # Handle 'Normal_Videos' -> 'Normal'
    if category.startswith('Normal'):
        return 'Normal'
    return category


def make_train_val_split(
    list_file: str,
    feature_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple:
    """
    Create stratified train/val split from a training list file.

    Args:
        list_file:   path to .list file (one feature path per line)
        feature_dir: directory where feature files live
        val_ratio:   fraction of each category to put in val set
        seed:        random seed for reproducibility

    Returns:
        (train_samples, val_samples) — each is list of (filename, label, cat_id)
    """
    random.seed(seed)

    with open(list_file) as f:
        lines = [l.strip() for l in f if l.strip()]

    by_cat = defaultdict(list)
    for line in lines:
        filename = line.replace('\\', '/').split('/')[-1]
        filepath = os.path.join(feature_dir, filename)
        if not os.path.exists(filepath):
            continue
        cat = get_category_from_filename(filename)
        if cat == 'Normal':
            label, cat_id = 0, 0
        elif cat in VIOLENCE_CATEGORIES:
            label, cat_id = 1, VIOLENCE_CATEGORIES[cat]
        elif cat in NON_VIOLENCE_ANOMALIES:
            continue  # skip non-violence anomalies during train/val split
        else:
            continue  # unknown category
        by_cat[cat].append((filename, label, cat_id))

    train_samples, val_samples = [], []
    for cat, items in sorted(by_cat.items()):
        random.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        val_samples.extend(items[:n_val])
        train_samples.extend(items[n_val:])

    return train_samples, val_samples


class UCFCrimeDataset(Dataset):
    """
    Dataset for UCF-Crime violence detection.

    For training: loads anomalous (violence) + normal videos.
    For testing:  loads all videos with ground truth labels.

    Args:
        feature_dir: path to directory containing .npy feature files
        list_file:   path to .list file with one feature path per line
        mode:        'train' or 'test'
        num_segments: fixed number of segments to sample/pad to (None = use all)
        violence_only: if True, skip non-violence anomaly classes during training
    """

    def __init__(
        self,
        feature_dir: str,
        list_file: str,
        mode: str = 'train',
        num_segments: int = 32,
        violence_only: bool = True,
        _override_samples: list = None,
    ):
        self.feature_dir = feature_dir
        self.mode = mode
        self.num_segments = num_segments
        self.violence_only = violence_only

        self.samples = []  # list of (filepath, label, category_id)
        if _override_samples is not None:
            self.samples = _override_samples
            n_a = sum(1 for _, l, _ in self.samples if l == 1)
            n_n = sum(1 for _, l, _ in self.samples if l == 0)
            print(f"[{self.mode}] Loaded {len(self.samples)} samples "
                  f"({n_a} anomalous, {n_n} normal) [from split]")
        else:
            self._load_list(list_file)

    def _load_list(self, list_file: str):
        """Parse list file and build sample list with labels."""
        with open(list_file, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]

        for line in lines:
            # Convert Windows path to just filename
            filename = line.replace('\\', '/').split('/')[-1]
            filepath = os.path.join(self.feature_dir, filename)

            category = get_category_from_filename(filename)

            if category == 'Normal':
                label = 0
                cat_id = 0
            elif category in VIOLENCE_CATEGORIES:
                label = 1
                cat_id = VIOLENCE_CATEGORIES[category]
            else:
                # Non-violence anomaly (Arrest, Arson, etc.)
                if self.violence_only and self.mode == 'train':
                    continue  # skip during training
                label = 1
                cat_id = 0  # unknown violence type

            if os.path.exists(filepath):
                self.samples.append((filepath, label, cat_id))
            # silently skip missing files (handles partial downloads)

        print(f"[{self.mode}] Loaded {len(self.samples)} samples "
              f"({sum(1 for _,l,_ in self.samples if l==1)} anomalous, "
              f"{sum(1 for _,l,_ in self.samples if l==0)} normal)")

    def _load_features(self, filepath: str) -> np.ndarray:
        """
        Load .npy feature file and return shape (T, 2048).
        Input shape is (T, 10, 2048) — average over 10 crops.
        """
        feat = np.load(filepath)  # (T, 10, 2048)
        if feat.ndim == 3:
            feat = feat.mean(axis=1)  # → (T, 2048)
        elif feat.ndim == 2:
            pass  # already (T, 2048)
        else:
            raise ValueError(f"Unexpected feature shape {feat.shape} in {filepath}")
        return feat.astype(np.float32)

    def _temporal_sample(self, feat: np.ndarray) -> np.ndarray:
        """
        Resize temporal dimension to self.num_segments.
        Uses uniform sampling if T > num_segments, padding if T < num_segments.
        """
        T = feat.shape[0]
        N = self.num_segments

        if T == N:
            return feat
        elif T > N:
            # Uniform sampling
            indices = np.linspace(0, T - 1, N, dtype=int)
            return feat[indices]
        else:
            # Pad with zeros at the end
            pad = np.zeros((N - T, feat.shape[1]), dtype=np.float32)
            return np.concatenate([feat, pad], axis=0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label, cat_id = self.samples[idx]
        feat = self._load_features(filepath)

        if self.num_segments is not None:
            feat = self._temporal_sample(feat)

        return {
            'features': torch.tensor(feat, dtype=torch.float32),  # (T, 2048)
            'label':    torch.tensor(label, dtype=torch.long),     # 0 or 1
            'cat_id':   torch.tensor(cat_id, dtype=torch.long),    # 0-6
            'filepath': filepath,
        }


class MILDataset(Dataset):
    """
    MIL-style dataset that returns pairs of (anomalous, normal) videos for ranking loss.
    Used during Stage 1 and 2 training.
    """

    def __init__(self, base_dataset: UCFCrimeDataset):
        self.anomalous = [(f, l, c) for f, l, c in base_dataset.samples if l == 1]
        self.normal    = [(f, l, c) for f, l, c in base_dataset.samples if l == 0]
        self.base = base_dataset
        print(f"[MIL] {len(self.anomalous)} anomalous, {len(self.normal)} normal")

    def __len__(self):
        return max(len(self.anomalous), len(self.normal))

    def __getitem__(self, idx):
        a_idx = idx % len(self.anomalous)
        n_idx = idx % len(self.normal)

        a_path, a_label, a_cat = self.anomalous[a_idx]
        n_path, n_label, n_cat = self.normal[n_idx]

        a_feat = self.base._load_features(a_path)
        n_feat = self.base._load_features(n_path)

        if self.base.num_segments is not None:
            a_feat = self.base._temporal_sample(a_feat)
            n_feat = self.base._temporal_sample(n_feat)

        return {
            'anom_features': torch.tensor(a_feat, dtype=torch.float32),
            'norm_features': torch.tensor(n_feat, dtype=torch.float32),
            'anom_cat':      torch.tensor(a_cat, dtype=torch.long),
        }


class InMemoryDataset(Dataset):
    """
    Lightweight dataset backed by a pre-built sample list (no list file needed).
    Used for the val split from make_train_val_split().
    """

    def __init__(self, samples: list, feature_dir: str, num_segments: int = 32):
        """
        Args:
            samples: list of (filename, label, cat_id) tuples
            feature_dir: directory containing .npy feature files
            num_segments: fixed temporal length
        """
        self.feature_dir = feature_dir
        self.num_segments = num_segments
        # Build full paths
        self.samples = [
            (os.path.join(feature_dir, fn), lbl, cid)
            for fn, lbl, cid in samples
        ]
        n_anom = sum(1 for _, l, _ in self.samples if l == 1)
        n_norm = sum(1 for _, l, _ in self.samples if l == 0)
        print(f"[InMemory] {len(self.samples)} samples ({n_anom} anomalous, {n_norm} normal)")

    def _load_features(self, filepath: str) -> np.ndarray:
        feat = np.load(filepath)
        if feat.ndim == 3:
            feat = feat.mean(axis=1)
        return feat.astype(np.float32)

    def _temporal_sample(self, feat: np.ndarray) -> np.ndarray:
        T, D = feat.shape
        N = self.num_segments
        if T == N:
            return feat
        elif T > N:
            indices = np.linspace(0, T - 1, N, dtype=int)
            return feat[indices]
        else:
            pad = np.zeros((N - T, D), dtype=np.float32)
            return np.concatenate([feat, pad], axis=0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label, cat_id = self.samples[idx]
        feat = self._load_features(filepath)
        feat = self._temporal_sample(feat)
        return {
            'features': torch.tensor(feat, dtype=torch.float32),
            'label':    torch.tensor(label, dtype=torch.long),
            'cat_id':   torch.tensor(cat_id, dtype=torch.long),
            'filepath': filepath,
        }


def build_dataloaders(
    train_feature_dir: str,
    test_feature_dir: str,
    train_list: str,
    test_list: str,
    num_segments: int = 32,
    batch_size: int = 32,
    num_workers: int = 4,
    val_ratio: float = 0.2,
    use_val_split: bool = True,
):
    """
    Build train (MIL pairs), val, and test dataloaders.

    When use_val_split=True, carves out a stratified 20% val set from the
    training list. The val set is used for evaluation during training instead
    of (or in addition to) the test set, which may be incomplete.

    Returns:
        train_loader, eval_loader, train_ds, eval_ds
        eval_loader/eval_ds is the val split if use_val_split else the test set.
    """
    if use_val_split:
        train_samples, val_samples = make_train_val_split(
            train_list, train_feature_dir, val_ratio=val_ratio
        )
        # Rebuild UCFCrimeDataset from train_samples only (anomalous + normal)
        train_ds = UCFCrimeDataset(
            train_feature_dir, train_list,
            mode='train', num_segments=num_segments, violence_only=True,
            _override_samples=[(os.path.join(train_feature_dir, fn), l, c)
                               for fn, l, c in train_samples]
        )
        eval_ds = InMemoryDataset(val_samples, train_feature_dir, num_segments)
    else:
        train_ds = UCFCrimeDataset(
            train_feature_dir, train_list,
            mode='train', num_segments=num_segments, violence_only=True
        )
        eval_ds = UCFCrimeDataset(
            test_feature_dir, test_list,
            mode='test', num_segments=num_segments, violence_only=False
        )

    mil_ds = MILDataset(train_ds)

    train_loader = DataLoader(
        mil_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, eval_loader, train_ds, eval_ds


if __name__ == '__main__':
    # Quick sanity check with test features only
    import sys

    test_dir = sys.argv[1] if len(sys.argv) > 1 else 'ucf_test/UCF_test_feature'
    test_list = sys.argv[2] if len(sys.argv) > 2 else None

    if test_list is None:
        # Build a temporary list from whatever .npy files exist
        files = [f for f in os.listdir(test_dir) if f.endswith('.npy')]
        tmp_list = '/tmp/ucf_test_tmp.list'
        with open(tmp_list, 'w') as f:
            for fn in files:
                f.write(fn + '\n')
        test_list = tmp_list

    ds = UCFCrimeDataset(test_dir, test_list, mode='test', num_segments=32)
    print(f"\nDataset size: {len(ds)}")

    item = ds[0]
    print(f"features shape : {item['features'].shape}")
    print(f"label          : {item['label']}")
    print(f"cat_id         : {item['cat_id']}")
    print(f"filepath       : {item['filepath']}")

    # Check category distribution
    from collections import Counter
    cats = Counter()
    for _, _, c in ds.samples:
        cats[c] += 1
    print(f"\nCategory distribution (id: count): {dict(sorted(cats.items()))}")
