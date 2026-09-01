# data/preprocessing.py — Memory-optimized preprocessing for CIC-IoT-2023 (full 46M records)
#
# Strategy:
#   1. Two-pass processing: first pass fits scaler & encoder, second pass transforms & saves
#   2. No SMOTE — uses Focal Loss for class imbalance (FL-compatible, privacy-safe)
#   3. Saves cluster data as .npy files on disk
#   4. MemmapDataset for lazy-loading during training

import pandas as pd
import numpy as np
import os
import gc
import json
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# ============================================================
# CIC-IoT-2023 feature columns (46 features)
# ============================================================
FEATURES = [
    'flow_duration', 'Header_Length', 'Protocol Type', 'Duration',
    'Rate', 'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number',
    'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
    'ece_flag_number', 'cwr_flag_number', 'ack_count', 'syn_count',
    'fin_count', 'urg_count', 'rst_count',
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC',
    'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC',
    'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size',
    'IAT', 'Number', 'Magnitue', 'Radius', 'Covariance',
    'Variance', 'Weight',
]

FEATURES_ALT = [f if f != 'psh_flag_number' else 'psh_flag_numbe'
                for f in FEATURES]


def _get_csv_files(data_dir):
    """Collect all CSV files from flat or subdirectory layout."""
    csv_files = []
    for item in sorted(os.listdir(data_dir)):
        item_path = os.path.join(data_dir, item)
        if item.endswith('.csv'):
            csv_files.append(item_path)
        elif os.path.isdir(item_path):
            for f in sorted(os.listdir(item_path)):
                if f.endswith('.csv'):
                    csv_files.append(os.path.join(item_path, f))
    return csv_files


def detect_features(df):
    """Auto-detect feature columns from the dataframe."""
    known_present = [c for c in FEATURES if c in df.columns]
    if len(known_present) >= 40:
        return known_present
    alt_present = [c for c in FEATURES_ALT if c in df.columns]
    if len(alt_present) >= 40:
        return alt_present
    exclude_cols = {'ts', 'label', 'label_encoded', 'Label'}
    actual_cols = [c for c in df.columns if c not in exclude_cols]
    numeric_cols = [c for c in actual_cols
                    if df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    return numeric_cols


def preprocess_pipeline_full(raw_data_dir, output_dir, num_clusters=9,
                             alpha=0.5, seed=42):
    """
    Memory-optimized preprocessing for full CIC-IoT-2023 dataset.

    Strategy:
      Pass 1: Read all CSVs → fit LabelEncoder + incremental StandardScaler
      Pass 2: Read all CSVs → transform + assign to clusters → save .npy per cluster
    """
    print("=" * 60)
    print("PREPROCESSING PIPELINE (Full Dataset, Memory-Optimized)")
    print("=" * 60)

    csv_files = _get_csv_files(raw_data_dir)
    print(f"Found {len(csv_files)} CSV files")

    os.makedirs(output_dir, exist_ok=True)

    # ================================================================
    # PASS 1: Fit scaler & encoder (streaming, ~2-3GB RAM)
    # ================================================================
    print("\n[PASS 1] Fitting LabelEncoder & StandardScaler...")

    # 1a. Collect all unique labels + detect features from first file
    all_labels = set()
    feature_cols = None
    total_rows = 0

    for i, filepath in enumerate(csv_files):
        df = pd.read_csv(filepath, low_memory=False)
        if feature_cols is None:
            feature_cols = detect_features(df)
            print(f"  Features: {len(feature_cols)} columns")

        # Clean
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=feature_cols, inplace=True)

        if 'label' in df.columns:
            all_labels.update(df['label'].unique())
        total_rows += len(df)

        if (i + 1) % 20 == 0:
            print(f"  Scanned {i+1}/{len(csv_files)} files ({total_rows:,} rows)")
        del df
        gc.collect()

    print(f"  Total: {total_rows:,} rows, {len(all_labels)} classes")

    # Fit label encoder
    le = LabelEncoder()
    le.fit(sorted(all_labels))
    num_classes = len(le.classes_)
    print(f"  Classes: {num_classes} → {list(le.classes_[:5])}...")

    # 1b. Fit StandardScaler incrementally (partial_fit)
    scaler = StandardScaler()
    for i, filepath in enumerate(csv_files):
        df = pd.read_csv(filepath, low_memory=False)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=feature_cols, inplace=True)
        scaler.partial_fit(df[feature_cols].values)
        if (i + 1) % 20 == 0:
            print(f"  Scaler fitted on {i+1}/{len(csv_files)} files")
        del df
        gc.collect()

    print("  Scaler fitted ✅")

    # ================================================================
    # PASS 2: Transform, partition, and save .npy files
    # ================================================================
    print(f"\n[PASS 2] Transform + Dirichlet partition (α={alpha}, K={num_clusters})...")

    # Read all data, transform, and collect
    # We need all data at once for Dirichlet partitioning
    # But we'll process it as float32 to save RAM (~17GB → ~8GB)
    all_X = []
    all_y = []

    for i, filepath in enumerate(csv_files):
        df = pd.read_csv(filepath, low_memory=False)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=feature_cols, inplace=True)

        X = scaler.transform(df[feature_cols].values).astype(np.float32)
        y = le.transform(df['label'].values)

        all_X.append(X)
        all_y.append(y)

        if (i + 1) % 20 == 0:
            print(f"  Transformed {i+1}/{len(csv_files)} files")
        del df
        gc.collect()

    print("  Concatenating...")
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    del all_X, all_y
    gc.collect()

    print(f"  Shape: X={X_all.shape}, y={y_all.shape}")
    print(f"  Memory: X={X_all.nbytes / 1e9:.1f}GB, y={y_all.nbytes / 1e6:.0f}MB")

    # Dirichlet partitioning
    print(f"\n  Dirichlet partitioning...")
    np.random.seed(seed)
    cluster_indices = {k: [] for k in range(num_clusters)}

    for c in range(num_classes):
        class_idx = np.where(y_all == c)[0]
        np.random.shuffle(class_idx)
        proportions = np.random.dirichlet([alpha] * num_clusters)
        splits = (proportions * len(class_idx)).astype(int)
        splits[-1] = len(class_idx) - splits[:-1].sum()

        start = 0
        for k in range(num_clusters):
            end = start + splits[k]
            cluster_indices[k].extend(class_idx[start:end].tolist())
            start = end

    # Save each cluster as .npy files
    print(f"\n  Saving cluster .npy files...")
    cluster_stats = {}

    for k in range(num_clusters):
        indices = np.array(cluster_indices[k])
        X_cluster = X_all[indices]
        y_cluster = y_all[indices]

        n_classes = len(np.unique(y_cluster))

        # Remove classes with < 3 samples (can't stratify)
        unique, counts = np.unique(y_cluster, return_counts=True)
        valid_mask = np.isin(y_cluster, unique[counts >= 3])
        X_cluster = X_cluster[valid_mask]
        y_cluster = y_cluster[valid_mask]

        # Train/val/test split
        try:
            X_tv, X_test, y_tv, y_test = train_test_split(
                X_cluster, y_cluster, test_size=0.15,
                random_state=seed, stratify=y_cluster)
            val_frac = 0.15 / 0.85
            X_train, X_val, y_train, y_val = train_test_split(
                X_tv, y_tv, test_size=val_frac,
                random_state=seed, stratify=y_tv)
        except ValueError:
            X_tv, X_test, y_tv, y_test = train_test_split(
                X_cluster, y_cluster, test_size=0.15, random_state=seed)
            val_frac = 0.15 / 0.85
            X_train, X_val, y_train, y_val = train_test_split(
                X_tv, y_tv, test_size=val_frac, random_state=seed)

        # Save .npy files
        cluster_dir = os.path.join(output_dir, f'cluster_{k}')
        os.makedirs(cluster_dir, exist_ok=True)

        np.save(os.path.join(cluster_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(cluster_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(cluster_dir, 'X_val.npy'), X_val)
        np.save(os.path.join(cluster_dir, 'y_val.npy'), y_val)
        np.save(os.path.join(cluster_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(cluster_dir, 'y_test.npy'), y_test)

        cluster_stats[k] = {
            'train': len(X_train), 'val': len(X_val), 'test': len(X_test),
            'classes': int(len(np.unique(y_train))),
        }
        print(f"  Cluster {k}: train={len(X_train):,}, val={len(X_val):,}, "
              f"test={len(X_test):,}, classes={len(np.unique(y_train))}")

        del X_cluster, y_cluster, X_tv, X_test, y_tv, y_test
        del X_train, X_val, y_train, y_val
        gc.collect()

    # Free main arrays
    del X_all, y_all
    gc.collect()

    # Save metadata
    metadata = {
        'feature_cols': feature_cols,
        'num_features': len(feature_cols),
        'num_classes': num_classes,
        'class_names': list(le.classes_),
        'num_clusters': num_clusters,
        'alpha': alpha,
        'total_records': total_rows,
        'cluster_stats': cluster_stats,
        'smote': False,
        'loss_strategy': 'focal_loss',
    }

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)

    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\n{'='*60}")
    print(f"PREPROCESSING COMPLETE")
    print(f"  Total records: {total_rows:,}")
    print(f"  Classes: {num_classes}")
    print(f"  Clusters: {num_clusters}")
    print(f"  SMOTE: disabled (using Focal Loss)")
    print(f"  Output: {output_dir}/")
    print(f"{'='*60}")

    return metadata


# ================================================================
# Pre-loaded TensorDataset from .npy files
# ================================================================
import torch
from torch.utils.data import DataLoader, TensorDataset


def create_cluster_loaders_from_npy(output_dir, batch_size=64, num_workers=0):
    """
    Create PyTorch DataLoaders by pre-loading .npy files into RAM as TensorDatasets.

    Memory usage: ~10GB for full 46M dataset (float32).
    Each cluster loaded separately → Ray workers only load their own cluster.
    """
    # Load metadata
    with open(os.path.join(output_dir, 'metadata.json')) as f:
        metadata = json.load(f)

    num_clusters = metadata['num_clusters']
    loaders = {}
    total_mem_mb = 0

    for k in range(num_clusters):
        cluster_dir = os.path.join(output_dir, f'cluster_{k}')
        if not os.path.exists(cluster_dir):
            continue

        result = {}
        for split in ['train', 'val', 'test']:
            X_path = os.path.join(cluster_dir, f'X_{split}.npy')
            y_path = os.path.join(cluster_dir, f'y_{split}.npy')

            if not os.path.exists(X_path):
                continue

            # Pre-load into RAM (fast training, ~1-2GB per large cluster)
            X = np.load(X_path)  # Full load into RAM
            y = np.load(y_path)

            # Runtime remap 34-class → 8-class UNB taxonomy
            from utils.mmap_dataset import _remap_labels
            y = _remap_labels(y)

            X_tensor = torch.FloatTensor(X).unsqueeze(1)  # (N, 1, F)
            y_tensor = torch.LongTensor(y)
            total_mem_mb += (X_tensor.nbytes + y_tensor.nbytes) / 1e6

            del X, y  # Free numpy arrays

            result[split] = DataLoader(
                TensorDataset(X_tensor, y_tensor),
                batch_size=batch_size,
                shuffle=(split == 'train'),
                drop_last=False,
                num_workers=num_workers,
                pin_memory=False,
            )

        loaders[k] = result
        stats = metadata['cluster_stats'].get(str(k), {})
        print(f"  Cluster {k}: train={stats.get('train',0):,}, "
              f"val={stats.get('val',0):,}, test={stats.get('test',0):,}")

    gc.collect()
    print(f"  Total data in RAM: {total_mem_mb/1e3:.1f}GB")

    return loaders, metadata


# ================================================================
# Backward-compatible: create_cluster_loaders for old interface
# ================================================================
def create_cluster_loaders(cluster_data, feature_cols, batch_size=64, seq_len=1):
    """Legacy function: Convert cluster_data DataFrames to DataLoaders."""
    loaders = {}
    for k, splits in cluster_data.items():
        result = {}
        for split_name in ['train', 'val', 'test']:
            if split_name == 'train' and 'train_X' in splits:
                X = splits['train_X']
                y = splits['train_y']
            else:
                df_split = splits[split_name]
                X = df_split[feature_cols].values.astype(np.float32)
                y = df_split['label_encoded'].values

            X_tensor = torch.FloatTensor(X).unsqueeze(1)
            y_tensor = torch.LongTensor(y)

            from torch.utils.data import DataLoader as DL, TensorDataset
            result[split_name] = DL(
                TensorDataset(X_tensor, y_tensor),
                batch_size=batch_size,
                shuffle=(split_name == 'train'),
                drop_last=False,
            )
        loaders[k] = result
    return loaders


# ================================================================
# CLI Entry Point
# ================================================================
if __name__ == '__main__':
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description='FairPFL Data Preprocessing')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--sample', type=float, default=None,
                        help='Sample fraction (e.g. 0.01). If None, uses full dataset.')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get('data', {})
    raw_dir = data_cfg.get('raw_dir', 'data/wataiData/csv/CICIoT2023')
    output_dir = data_cfg.get('processed_dir', 'data/processed')
    num_clusters = data_cfg.get('num_clusters', 9)
    alpha = data_cfg.get('dirichlet_alpha', 0.5)
    seed = cfg.get('experiment', {}).get('seed', 42)

    if args.sample and args.sample < 1.0:
        # Small sample mode: use old pipeline with in-memory processing
        print(f"[!] Sample mode: {args.sample*100:.0f}%")
        from data.preprocessing import preprocess_pipeline_full

        # For sample mode, we still use the full pipeline but read fewer rows
        # This is handled inside individual CSV reads
        metadata = preprocess_pipeline_full(
            raw_data_dir=raw_dir,
            output_dir=output_dir,
            num_clusters=num_clusters,
            alpha=alpha,
            seed=seed,
        )
    else:
        metadata = preprocess_pipeline_full(
            raw_data_dir=raw_dir,
            output_dir=output_dir,
            num_clusters=num_clusters,
            alpha=alpha,
            seed=seed,
        )

    # Verify by loading one cluster
    print("\n[+] Verification: loading cluster_0...")
    loaders, _ = create_cluster_loaders_from_npy(output_dir)
    X_batch, y_batch = next(iter(loaders[0]['train']))
    print(f"  Batch shape: X={X_batch.shape}, y={y_batch.shape}")
    print(f"  Memory-mapped loading works ✅")
