# utils/mmap_dataset.py — Memory-mapped lazy dataset for .npz cluster files
#
# Problem with naive np.load(npz_path):
#   - Loads ALL arrays into RAM immediately (train+val+test ≈ 316MB/cluster × 9 = ~3GB)
#   - In Ray actors this memory is duplicated per concurrent actor
#
# Solution:
#   - np.load(npz_path, mmap_mode='r') → disk-backed memory map
#     Only the pages that are actually accessed are brought into RAM.
#   - torch.from_numpy(arr.copy()) → zero extra allocation per batch
#   - Result: each Ray actor holds a tiny mmap handle, not a 300MB array.

import gc
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 34-class → 8-class UNB taxonomy runtime remapping
# ============================================================
# Original 34 classes (alphabetical LabelEncoder order) → 8 UNB categories:
#   [0] Benign, [1] BruteForce, [2] DDoS, [3] DoS,
#   [4] Mirai,  [5] Recon,      [6] Spoofing, [7] Web-based
#
# This lookup table is applied ONCE at dataset load time (not per-item).
# Cost: single numpy array indexing → <1ms for 500K samples.
REMAP_34_TO_8 = np.array([
    7,  #  0: Backdoor_Malware        → Web-based
    0,  #  1: BenignTraffic           → Benign
    7,  #  2: BrowserHijacking        → Web-based
    7,  #  3: CommandInjection        → Web-based
    2,  #  4: DDoS-ACK_Fragmentation  → DDoS
    2,  #  5: DDoS-HTTP_Flood         → DDoS
    2,  #  6: DDoS-ICMP_Flood         → DDoS
    2,  #  7: DDoS-ICMP_Fragmentation → DDoS
    2,  #  8: DDoS-PSHACK_Flood       → DDoS
    2,  #  9: DDoS-RSTFINFlood        → DDoS
    2,  # 10: DDoS-SYN_Flood          → DDoS
    2,  # 11: DDoS-SlowLoris          → DDoS
    2,  # 12: DDoS-SynonymousIP_Flood → DDoS
    2,  # 13: DDoS-TCP_Flood          → DDoS
    2,  # 14: DDoS-UDP_Flood          → DDoS
    2,  # 15: DDoS-UDP_Fragmentation  → DDoS
    6,  # 16: DNS_Spoofing            → Spoofing
    1,  # 17: DictionaryBruteForce    → BruteForce
    3,  # 18: DoS-HTTP_Flood          → DoS
    3,  # 19: DoS-SYN_Flood           → DoS
    3,  # 20: DoS-TCP_Flood           → DoS
    3,  # 21: DoS-UDP_Flood           → DoS
    6,  # 22: MITM-ArpSpoofing        → Spoofing
    4,  # 23: Mirai-greeth_flood      → Mirai
    4,  # 24: Mirai-greip_flood       → Mirai
    4,  # 25: Mirai-udpplain          → Mirai
    5,  # 26: Recon-HostDiscovery     → Recon
    5,  # 27: Recon-OSScan            → Recon
    5,  # 28: Recon-PingSweep         → Recon
    5,  # 29: Recon-PortScan          → Recon
    7,  # 30: SqlInjection            → Web-based
    7,  # 31: Uploading_Attack        → Web-based
    5,  # 32: VulnerabilityScan       → Recon
    7,  # 33: XSS                     → Web-based
], dtype=np.int64)

# 6-class taxonomy: Exclude BruteForce (1) and Web-based (7) due to
# insufficient per-client samples in FL (as low as 273 per cluster).
# Remaining 8-class labels [0,2,3,4,5,6] are remapped to contiguous [0..5]:
#   0:Benign→0, 2:DDoS→1, 3:DoS→2, 4:Mirai→3, 5:Recon→4, 6:Spoofing→5
REMAP_8_TO_6 = {0: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
CLASSES_6 = ['Benign', 'DDoS', 'DoS', 'Mirai', 'Recon', 'Spoofing']
EXCLUDE_CLASSES_6 = {1, 7}  # BruteForce, Web-based


def _remap_labels(y, exclude_classes=None):
    """
    Remap 34-class labels to 8-class UNB taxonomy.
    If labels are already in 0-7 range, no remapping is applied.

    Args:
        y: numpy array of labels
        exclude_classes: set of 8-class label IDs to exclude (e.g. {1, 7}).
            If provided, returns (remapped_y, keep_mask) where keep_mask
            is a boolean array indicating which samples to keep.
            If None, returns remapped_y only (backward compatible).
    """
    y_arr = np.asarray(y)
    max_label = y_arr.max()
    if max_label > 7:
        if max_label > 33:
            raise ValueError(f"Unexpected label value {max_label} > 33. "
                             f"Expected CIC-IoT-2023 labels in [0, 33].")
        y_arr = REMAP_34_TO_8[y_arr]

    if exclude_classes is not None:
        # Filter out excluded classes and remap to contiguous labels
        keep_mask = np.ones(len(y_arr), dtype=bool)
        for cls in exclude_classes:
            keep_mask &= (y_arr != cls)

        # Build contiguous remap: kept classes → 0..N-1
        kept_classes = sorted(set(range(8)) - exclude_classes)
        remap = np.zeros(8, dtype=np.int64)
        for new_id, old_id in enumerate(kept_classes):
            remap[old_id] = new_id

        y_remapped = remap[y_arr[keep_mask]]
        return y_remapped, keep_mask

    return y_arr


def _stratified_subsample(y, max_samples, rng, samples_per_class=5000):
    """
    Equal per-class sampling with SMOTE for extreme minority classes.

    Each class gets exactly `samples_per_class` samples:
      - Majority classes (e.g. DDoS ~1-6M): random undersampling.
      - Moderate minority (500+ samples): random oversampling with replacement.
      - Extreme minority (<500 samples, e.g. BruteForce 273 in C3):
        SMOTE synthetic interpolation instead of naive repetition.
        SMOTE generates diverse training signal by interpolating between
        real neighbors, avoiding memorization of repeated samples.

    Reference: Per-client SMOTE in FL recommended by literature
    (escholarship.org, NIH, MDPI 2024-2025).

    Args:
        y: numpy array of (remapped) labels
        max_samples: ignored (kept for API compat)
        rng: numpy random Generator
        samples_per_class: exact samples per class (default 2000)

    Returns:
        tuple: (X_indices or None, X_synthetic, y_synthetic)
        If SMOTE was used, returns combined indices + synthetic data.
        Caller must handle the synthetic data separately.
    """
    classes, counts = np.unique(y, return_counts=True)

    selected_indices = []
    for cls, cnt in zip(classes, counts):
        cls_indices = np.where(y == cls)[0]

        if cnt >= samples_per_class:
            # Majority: random undersampling (no replacement)
            chosen = rng.choice(cls_indices, size=samples_per_class,
                                replace=False)
        else:
            # Minority: oversample with replacement to reach target
            chosen = rng.choice(cls_indices, size=samples_per_class,
                                replace=True)

        selected_indices.append(chosen)

    return np.concatenate(selected_indices)


def _eval_stratified_subsample(y, max_total, rng, min_per_class=200):
    """
    Stratified subsampling for val/test with minimum per-class guarantee.

    Preserves the ORIGINAL imbalanced distribution (honest evaluation)
    while guaranteeing at least `min_per_class` samples per class for
    statistically reliable macro-F1 estimation.

    Without this guarantee, BruteForce (~0.01% of val) gets only ~9
    samples per cluster → macro-F1 estimate is statistically meaningless.

    Strategy:
      1. Each class gets max(min_per_class, proportional_share) samples
      2. Total is capped at max_total
      3. Original distribution is preserved for majority classes

    Args:
        y: numpy array of (remapped) labels (original distribution)
        max_total: total sample budget
        rng: numpy random Generator
        min_per_class: minimum samples per class (default 200)

    Returns:
        numpy array of selected indices
    """
    classes, counts = np.unique(y, return_counts=True)
    num_classes = len(classes)
    N = len(y)

    # Phase 1: guarantee minimum for each class
    selected = []
    remaining_budget = max_total

    for cls, cnt in zip(classes, counts):
        cls_indices = np.where(y == cls)[0]
        # Take up to min_per_class (or all if class is tiny)
        n_take = min(min_per_class, cnt)
        chosen = rng.choice(cls_indices, size=n_take, replace=False)
        selected.append(chosen)
        remaining_budget -= n_take

    # Phase 2: distribute remaining budget proportionally
    if remaining_budget > 0:
        already_selected = {cls: min(min_per_class, cnt)
                            for cls, cnt in zip(classes, counts)}
        for cls, cnt in zip(classes, counts):
            available = cnt - already_selected[cls]
            if available <= 0:
                continue
            # Proportional share of remaining budget
            prop_share = int(remaining_budget * (cnt / N))
            n_extra = min(prop_share, available)
            if n_extra > 0:
                cls_indices = np.where(y == cls)[0]
                # Exclude already-selected indices
                already = selected[list(classes).index(cls)]
                mask = np.ones(len(cls_indices), dtype=bool)
                already_set = set(already.tolist())
                mask = np.array([i not in already_set for i in cls_indices])
                remaining_idx = cls_indices[mask]
                if len(remaining_idx) > 0:
                    n_extra = min(n_extra, len(remaining_idx))
                    extra = rng.choice(remaining_idx, size=n_extra, replace=False)
                    selected.append(extra)

    return np.concatenate(selected)


class MmapNpzDataset(Dataset):
    """
    Lazy-loading Dataset backed by a memory-mapped .npz split.

    v8 improvements over v6:
      1. Dead feature removal: 7 features with near-zero variance are dropped
         (indices identified via variance analysis on cluster_5).
      2. Per-client SMOTE for train split: minority classes with <SMOTE_THRESHOLD
         original samples get synthetic interpolation instead of naive repetition.

    Labels are automatically remapped from 34-class to 8-class UNB taxonomy
    at load time if needed (zero runtime cost per batch).

    Subsampling strategy differs by split (per literature best practice):
      - train: equal per-class sampling with SMOTE for extreme minorities
      - val/test: stratified subsample preserving original distribution

    Args:
        npz_path:    Path to the cluster .npz file
        split:       'train', 'val', or 'test'
        max_samples: If not None, subsample this many rows
        seed:        RNG seed for reproducible subsampling
        seq_len:     Number of consecutive rows per sample (default 1)
    """

    # v9: Dead feature removal DISABLED — LSTM benefits from full 46 features.
    # Centralized test showed LSTM+46f F1=0.694 vs MLP+37f F1=0.675.
    # Features 29,31,32 were incorrectly flagged as dead (var=0.18-0.30).
    DEAD_FEATURE_INDICES = []  # No features removed

    # SMOTE is applied to classes with fewer than this many original samples
    SMOTE_THRESHOLD = 500

    def __init__(self, npz_path: str, split: str,
                 max_samples: int = None, seed: int = 42,
                 seq_len: int = 1, exclude_classes: set = None,
                 remap: bool = True):
        # mmap_mode='r' → file is memory-mapped, not copied into RAM
        npz = np.load(npz_path, mmap_mode='r')
        X_mm = npz[f'X_{split}']   # shape: (N, num_features)
        y_mm = npz[f'y_{split}']   # shape: (N,)
        N = len(X_mm)

        # Remap labels and optionally exclude classes (6-class mode).
        # remap=False: non-CIC dataset (e.g. Edge-IIoTset) whose labels are
        # already contiguous 0..C-1 — use them as-is, never touch _remap_labels.
        if not remap:
            y_remapped = np.asarray(y_mm).astype(np.int64)
            X_mm_filtered = X_mm
        elif exclude_classes:
            y_remapped, keep_mask = _remap_labels(
                np.array(y_mm), exclude_classes=exclude_classes)
            # Apply mask to X as well — filter out excluded samples
            X_mm_filtered = np.array(X_mm)[keep_mask]
            N = len(y_remapped)
        else:
            y_remapped = _remap_labels(np.array(y_mm))
            X_mm_filtered = X_mm

        # Compute live feature mask (remove dead features)
        all_features = list(range(X_mm.shape[1]))
        live_features = [i for i in all_features
                         if i not in self.DEAD_FEATURE_INDICES]
        self._live_features = live_features

        if max_samples is not None and N > max_samples:
            rng = np.random.default_rng(seed=seed)
            if split == 'train':
                idx = _stratified_subsample(y_remapped, max_samples, rng)
                idx = np.sort(idx)

                # Extract selected data and apply SMOTE for extreme minorities
                X_selected = np.array(X_mm_filtered[idx])[:, live_features]
                y_selected = y_remapped[idx]

                X_selected, y_selected = self._apply_smote(
                    X_selected, y_selected, rng
                )

                self.X = X_selected
                self.y = y_selected
            else:
                idx = _eval_stratified_subsample(y_remapped, max_samples, rng,
                                                 min_per_class=100)
                idx = np.sort(idx)
                self.X = np.array(X_mm_filtered[idx])[:, live_features]
                self.y = y_remapped[idx]
        else:
            self.X = np.array(X_mm_filtered)[:, live_features]
            self.y = y_remapped

        self.seq_len = max(1, seq_len)
        # Number of valid sliding-window positions
        self.n = max(1, len(self.X) - self.seq_len + 1)

    def _apply_smote(self, X, y, rng):
        """
        Apply SMOTE-style synthetic oversampling for extreme minority classes.

        For classes with <SMOTE_THRESHOLD original samples (before oversampling),
        we replace the naively duplicated samples with synthetic interpolations
        between randomly selected pairs of real samples from that class.

        This avoids memorization of repeated rows and provides the model with
        diverse training signal, especially important for BruteForce in C3 (273 orig).

        Args:
            X: (N, F) numpy array of features (already subsampled)
            y: (N,) numpy array of labels
            rng: numpy random Generator

        Returns:
            X_new, y_new: arrays with synthetic samples replacing duplicates
        """
        classes, counts = np.unique(y, return_counts=True)
        X_parts = []
        y_parts = []

        for cls, cnt in zip(classes, counts):
            cls_mask = y == cls
            X_cls = X[cls_mask]

            # Find unique rows (original samples before oversampling)
            _, unique_idx = np.unique(X_cls, axis=0, return_index=True)
            n_unique = len(unique_idx)

            if n_unique < self.SMOTE_THRESHOLD and n_unique >= 2:
                # This class was oversampled from few originals → apply SMOTE
                X_unique = X_cls[unique_idx]
                n_target = len(X_cls)  # total needed (= samples_per_class)

                # Keep all unique originals
                X_parts.append(X_unique)
                y_parts.append(np.full(n_unique, cls, dtype=y.dtype))

                # Generate synthetic samples for the remainder
                n_synthetic = n_target - n_unique
                if n_synthetic > 0:
                    X_synth = self._smote_generate(
                        X_unique, n_synthetic, rng, k=min(5, n_unique - 1)
                    )
                    X_parts.append(X_synth)
                    y_parts.append(np.full(n_synthetic, cls, dtype=y.dtype))
            else:
                # Majority class or enough unique samples — keep as-is
                X_parts.append(X_cls)
                y_parts.append(np.full(len(X_cls), cls, dtype=y.dtype))

        return np.vstack(X_parts), np.concatenate(y_parts)

    @staticmethod
    def _smote_generate(X_minority, n_samples, rng, k=5):
        """
        Generate synthetic samples via SMOTE interpolation.

        For each synthetic sample:
          1. Pick a random real sample from X_minority
          2. Find its k nearest neighbors (Euclidean distance)
          3. Pick one neighbor randomly
          4. Interpolate: x_new = x + λ * (x_neighbor - x), λ ∈ [0, 1)

        Args:
            X_minority: (n_real, F) array of real minority samples
            n_samples: number of synthetic samples to generate
            rng: numpy random Generator
            k: number of nearest neighbors to consider

        Returns:
            (n_samples, F) array of synthetic samples
        """
        n_real, n_features = X_minority.shape

        # Pre-compute pairwise distances for neighbor selection
        # For small n_real (<500), this is fast and memory-efficient
        from scipy.spatial.distance import cdist
        distances = cdist(X_minority, X_minority, metric='euclidean')
        # For each sample, get indices of k nearest neighbors (excluding self)
        np.fill_diagonal(distances, np.inf)
        nn_indices = np.argsort(distances, axis=1)[:, :k]

        synthetic = np.empty((n_samples, n_features), dtype=X_minority.dtype)
        for i in range(n_samples):
            # Pick a random real sample
            idx = rng.integers(0, n_real)
            # Pick a random neighbor
            nn_idx = nn_indices[idx, rng.integers(0, k)]
            # Interpolate
            lam = rng.random()
            synthetic[i] = X_minority[idx] + lam * (
                X_minority[nn_idx] - X_minority[idx]
            )

        return synthetic

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        if self.seq_len == 1:
            # Fast path — backward compatible (1, F)
            x = torch.from_numpy(self.X[i].copy()).float().unsqueeze(0)
        else:
            # Sliding window: (seq_len, F)
            x = torch.from_numpy(self.X[i:i + self.seq_len].copy()).float()
        # Label = last row in the window (prediction target)
        y = int(self.y[i + self.seq_len - 1])
        return x, y


def make_loaders(npz_path: str, batch_size: int,
                 max_train: int = None, max_eval: int = None,
                 seed: int = 42, seq_len: int = 1,
                 exclude_classes: set = None, remap: bool = True) -> dict:
    """
    Build train/val/test DataLoader dict from a cluster .npz file.

    Training data is already balanced via equal per-class sampling
    (_stratified_subsample with samples_per_class=1000), so a simple
    shuffle=True is sufficient — no WeightedRandomSampler needed.

    Val/Test loaders preserve the original distribution for honest evaluation.

    Args:
        npz_path:        Path to cluster_N.npz
        batch_size:      DataLoader batch size
        max_train:       Max training samples (None = full)
        max_eval:        Max val/test samples (None = full)
        seed:            RNG seed for subsampling
        seq_len:         Sliding window length for LSTM (default 1)
        exclude_classes: Set of 8-class label IDs to exclude (e.g. {1,7})

    Returns:
        {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    loaders = {}
    for split in ('train', 'val', 'test'):
        ms = max_train if split == 'train' else max_eval
        ds = MmapNpzDataset(npz_path, split, max_samples=ms, seed=seed,
                            seq_len=seq_len, exclude_classes=exclude_classes,
                            remap=remap)

        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == 'train'),  # shuffle train, not val/test
            drop_last=False,
            num_workers=0,
            pin_memory=False,
        )
    gc.collect()
    return loaders


def compute_class_weights_from_npz(npz_path: str,
                                   num_classes: int = 8,
                                   max_samples: int = None,
                                   seed: int = 42):
    """
    Return uniform class weights.

    With equal per-class sampling, all classes have the same count,
    so class-balanced weights are unnecessary (they would all be ~1.0).
    Kept for API compatibility — callers can pass the result to
    FocalLoss(alpha=...) without code changes, but the effect is
    equivalent to alpha=None.
    """
    from models.focal_loss import compute_class_weights
    # Return targeted minority-boost weights:
    # BruteForce (class 1) and Web-based (class 7) get 4x boost.
    # labels arg is unused in the new compute_class_weights — pass dummy.
    return compute_class_weights(np.zeros(1, dtype=int), num_classes=num_classes)


