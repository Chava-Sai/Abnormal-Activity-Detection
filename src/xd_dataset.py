"""
XD-Violence dataset loader for weakly-supervised MIL training.
Features: RGB (1024-dim) + Flow (1024-dim) → concatenated (2048-dim)
Labels from filename: label_A = Normal (0), label_B* = Violent (1)
"""
import os, numpy as np, torch
from torch.utils.data import Dataset

# XD-Violence label subtypes → our category mapping
XD_LABEL_MAP = {
    'A':  ('Normal',      0),
    'B1': ('Fighting',    1),
    'B2': ('CarAccident', 2),
    'B3': ('Explosion',   3),
    'B4': ('Riot',        4),
    'B5': ('Shooting',    5),
    'B6': ('Abuse',       6),
}

def parse_xd_label(stem):
    """Extract binary label and subtype from XD-Violence clip stem."""
    for key in sorted(XD_LABEL_MAP.keys(), key=len, reverse=True):
        tag = f'_label_{key}'
        if tag in stem:
            name, cat_id = XD_LABEL_MAP[key]
            binary = 0 if key == 'A' else 1
            return binary, cat_id, name
    return -1, -1, 'Unknown'

def get_stem(fname):
    base = os.path.splitext(fname)[0]
    parts = base.rsplit('__', 1)
    return parts[0] if (len(parts) == 2 and parts[1].isdigit()) else base

def load_avg_crops(dirpath, stem, max_crops=10):
    crops = []
    for i in range(max_crops):
        p = os.path.join(dirpath, f'{stem}__{i}.npy')
        if os.path.exists(p):
            crops.append(np.load(p))
    if not crops:
        return None
    minT = min(c.shape[0] for c in crops)
    return np.stack([c[:minT] for c in crops]).mean(0).astype(np.float32)

def build_xd_split(rgb_dir, flow_dir, num_segments=32, min_clips=1):
    """Scan rgb_dir, match with flow_dir, return list of (feat_array, binary_label, cat_id)."""
    rgb_files = [f for f in os.listdir(rgb_dir) if f.endswith('.npy')]
    stems = sorted(set(get_stem(f) for f in rgb_files))

    samples = []
    skipped = 0
    for stem in stems:
        binary, cat_id, cat_name = parse_xd_label(stem)
        if binary == -1:
            skipped += 1
            continue
        rgb  = load_avg_crops(rgb_dir,  stem)
        flow = load_avg_crops(flow_dir, stem)
        if rgb is None or flow is None:
            skipped += 1
            continue
        minT = min(rgb.shape[0], flow.shape[0])
        feat = np.concatenate([rgb[:minT], flow[:minT]], axis=1)  # (T, 2048)
        # Resample to fixed num_segments
        T = feat.shape[0]
        if T < 2:
            skipped += 1
            continue
        idx  = np.linspace(0, T - 1, num_segments).astype(int)
        feat = feat[idx]                                           # (num_segments, 2048)
        samples.append((feat, binary, cat_id, cat_name, stem))
    print(f'XD split: {len(samples)} clips ({skipped} skipped)')
    n_anom   = sum(1 for s in samples if s[1] == 1)
    n_normal = sum(1 for s in samples if s[1] == 0)
    print(f'  Normal={n_normal}  Violent={n_anom}')
    return samples

class XDViolenceDataset(Dataset):
    """
    Torch Dataset for XD-Violence MIL training.
    Each item: features (T, 2048), label (0/1), cat_id.
    """
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat, binary, cat_id, cat_name, stem = self.samples[idx]
        return {
            'features': torch.tensor(feat, dtype=torch.float32),
            'label':    torch.tensor(binary, dtype=torch.float32),
            'cat_id':   torch.tensor(cat_id,  dtype=torch.long),
            'name':     stem,
        }

def make_xd_train_val(rgb_dir, flow_dir, num_segments=32,
                      val_ratio=0.2, seed=42):
    """Split XD-Violence into train/val by clip, stratified by binary label."""
    samples = build_xd_split(rgb_dir, flow_dir, num_segments)
    anom    = [s for s in samples if s[1] == 1]
    normal  = [s for s in samples if s[1] == 0]
    rng     = np.random.default_rng(seed)
    def split(lst):
        idx = rng.permutation(len(lst))
        cut = int(len(lst) * val_ratio)
        return [lst[i] for i in idx[cut:]], [lst[i] for i in idx[:cut]]
    tr_a, va_a = split(anom)
    tr_n, va_n = split(normal)
    train = tr_a + tr_n
    val   = va_a + va_n
    print(f'XD train={len(train)} (anom={len(tr_a)}, normal={len(tr_n)})')
    print(f'XD val  ={len(val)}   (anom={len(va_a)}, normal={len(va_n)})')
    return XDViolenceDataset(train), XDViolenceDataset(val)

if __name__ == '__main__':
    import sys
    PROJ = '/projectnb/cs585/students/saichava'
    rgb   = f'{PROJ}/datasets/i3d-features/RGB'
    flow  = f'{PROJ}/datasets/i3d-features/Flow'
    train_ds, val_ds = make_xd_train_val(rgb, flow)
    print(f'Train item 0 features shape: {train_ds[0]["features"].shape}')
    print(f'Train item 0 label: {train_ds[0]["label"]}')
