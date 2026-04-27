"""
Utilities for resolving and loading pre-extracted video feature files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

FEATURE_SUFFIXES = (
    "_x264_i3d.npy",
    "_i3d.npy",
    "_x264.npy",
    ".npy",
)


def strip_feature_suffix(path: str) -> str:
    """Return a normalized feature stem for common UCF naming variants."""
    basename = os.path.basename(path)
    for suffix in FEATURE_SUFFIXES:
        if basename.endswith(suffix):
            return basename[: -len(suffix)]
    return os.path.splitext(basename)[0]


def scan_feature_files(feature_dir: str) -> List[str]:
    """Recursively collect `.npy` feature files below `feature_dir`."""
    root = Path(feature_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"feature directory not found: {root}")
    return sorted(str(path.resolve()) for path in root.rglob("*.npy") if path.is_file())


def _candidate_keys(path: str, feature_dir: Optional[str] = None) -> List[str]:
    """Generate lookup keys that tolerate suffix and path-format differences."""
    normalized = path.replace("\\", "/").strip()
    keys = []

    if feature_dir:
        try:
            rel = os.path.relpath(normalized, feature_dir).replace("\\", "/")
            keys.append(rel)
        except ValueError:
            pass

    basename = os.path.basename(normalized)
    stem = strip_feature_suffix(basename)

    keys.extend([
        normalized.lstrip("./"),
        basename,
        stem,
        stem + ".npy",
        stem + "_x264.npy",
        stem + "_i3d.npy",
        stem + "_x264_i3d.npy",
    ])

    deduped = []
    seen = set()
    for key in keys:
        if key and key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


def build_feature_index(feature_dir: str) -> Dict[str, str]:
    """Build a tolerant lookup index for recursively stored feature files."""
    index: Dict[str, str] = {}
    for path in scan_feature_files(feature_dir):
        for key in _candidate_keys(path, feature_dir=feature_dir):
            index.setdefault(key, path)
    return index


def resolve_feature_path(reference: str, feature_dir: str, index: Dict[str, str]) -> Optional[str]:
    """
    Resolve a list-file entry or basename to a real feature file path.
    Returns None when no matching file exists under `feature_dir`.
    """
    if not reference:
        return None

    raw = reference.replace("\\", "/").strip()
    if os.path.isabs(raw) and os.path.exists(raw):
        return str(Path(raw).expanduser().resolve())

    for key in _candidate_keys(raw, feature_dir=feature_dir):
        if key in index:
            return index[key]

        joined = Path(feature_dir).expanduser() / key
        if joined.exists():
            return str(joined.resolve())

    return None


def load_feature_array(feature_path: str) -> np.ndarray:
    """
    Load a feature file as float32 with shape (T, D).
    Supports raw (T, D) and 10-crop (T, 10, D) layouts.
    """
    feat = np.load(feature_path)
    if feat.ndim == 3:
        feat = feat.mean(axis=1)
    elif feat.ndim != 2:
        raise ValueError(f"Unexpected feature shape {feat.shape} in {feature_path}")
    return feat.astype(np.float32, copy=False)


def temporal_resize(feat: np.ndarray, num_segments: Optional[int]) -> np.ndarray:
    """Uniformly sample or zero-pad along time to `num_segments`."""
    if num_segments is None:
        return feat

    t, d = feat.shape
    n = int(num_segments)
    if t == n:
        return feat
    if t > n:
        indices = np.linspace(0, t - 1, n, dtype=int)
        return feat[indices]

    pad = np.zeros((n - t, d), dtype=np.float32)
    return np.concatenate([feat, pad], axis=0)


def infer_feature_dim(feature_paths: Iterable[str]) -> Optional[int]:
    """Infer feature dimensionality from the first readable feature file."""
    for path in feature_paths:
        feat = load_feature_array(path)
        if feat.ndim == 2 and feat.shape[1] > 0:
            return int(feat.shape[1])
    return None

