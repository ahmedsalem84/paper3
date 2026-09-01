#!/usr/bin/env python3
"""
utils/compress_data.py — Compress processed cluster data for cloud transfer.

Converts individual .npy files per cluster → compressed .npz archives.
Achieves ~83% size reduction (8.5GB → ~1.4GB) using zlib compression.

Usage:
    python utils/compress_data.py                         # all clusters
    python utils/compress_data.py --clusters 0 1 2        # specific clusters
    python utils/compress_data.py --output_dir /tmp/npz   # custom output dir
    python utils/compress_data.py --dtype float16         # float16 (smaller, less precise)
    python utils/compress_data.py --verify                 # verify after compression
"""

import argparse
import os
import shutil
import time

import numpy as np


def compress_cluster(cluster_dir: str, output_path: str,
                     dtype: str = 'float32', verify: bool = False) -> dict:
    """Compress one cluster's .npy files into a single .npz archive.

    Args:
        cluster_dir: Path to cluster directory (contains X_train.npy etc.)
        output_path: Output .npz file path
        dtype: Target dtype — 'float32' (safe) or 'float16' (smaller)
        verify: If True, reload and verify shapes after compression

    Returns:
        dict with original_mb, compressed_mb, ratio, time_sec
    """
    splits = ['train', 'val', 'test']
    arrays = {}
    original_bytes = 0

    for split in splits:
        for arr_name in ['X', 'y']:
            fpath = os.path.join(cluster_dir, f'{arr_name}_{split}.npy')
            if not os.path.exists(fpath):
                print(f"  WARNING: {fpath} not found, skipping")
                continue
            arr = np.load(fpath, mmap_mode='r')
            original_bytes += arr.nbytes

            # Convert dtype if requested (y stays int)
            if arr_name == 'X' and dtype == 'float16':
                arr = arr.astype(np.float16)
            elif arr_name == 'X':
                arr = arr.astype(np.float32)
            else:
                arr = arr.astype(np.int32)  # labels: int32 sufficient

            arrays[f'{arr_name}_{split}'] = arr

    if not arrays:
        raise ValueError(f"No .npy files found in {cluster_dir}")

    t0 = time.time()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    elapsed = time.time() - t0

    compressed_bytes = os.path.getsize(output_path + '.npz'
                                        if not output_path.endswith('.npz')
                                        else output_path)

    if verify:
        data = np.load(output_path if output_path.endswith('.npz')
                       else output_path + '.npz')
        for key in arrays:
            assert key in data, f"Key {key} missing in compressed file!"
            assert data[key].shape == arrays[key].shape, \
                f"Shape mismatch for {key}: {data[key].shape} vs {arrays[key].shape}"
        print(f"  ✅ Verification passed ({len(arrays)} arrays)")

    return {
        'original_mb': original_bytes / 1024 / 1024,
        'compressed_mb': compressed_bytes / 1024 / 1024,
        'ratio': compressed_bytes / max(original_bytes, 1),
        'time_sec': elapsed,
        'arrays': list(arrays.keys()),
    }


def main():
    parser = argparse.ArgumentParser(description='Compress cluster data for cloud transfer')
    parser.add_argument('--data_dir', default='data/processed',
                        help='Source directory with cluster_N subdirs')
    parser.add_argument('--output_dir', default='data/compressed',
                        help='Output directory for .npz files')
    parser.add_argument('--clusters', type=int, nargs='*', default=None,
                        help='Cluster IDs to compress (default: all found)')
    parser.add_argument('--dtype', choices=['float32', 'float16'], default='float32',
                        help='Float dtype for X arrays. float16 saves ~50%% more space '
                             'but may reduce accuracy by <0.1%%')
    parser.add_argument('--verify', action='store_true',
                        help='Verify compressed files by reloading and checking shapes')
    parser.add_argument('--also_copy_meta', action='store_true', default=True,
                        help='Also copy metadata.json, scaler.pkl, label_encoder.pkl')
    args = parser.parse_args()

    # Discover clusters
    if args.clusters is not None:
        cluster_ids = args.clusters
    else:
        cluster_ids = sorted([
            int(d.replace('cluster_', ''))
            for d in os.listdir(args.data_dir)
            if d.startswith('cluster_') and
               os.path.isdir(os.path.join(args.data_dir, d))
        ])

    print(f"{'='*60}")
    print(f"FairPFL Data Compressor")
    print(f"Source:  {args.data_dir}")
    print(f"Output:  {args.output_dir}")
    print(f"Clusters: {cluster_ids}")
    print(f"DType:   {args.dtype}")
    print(f"{'='*60}\n")

    os.makedirs(args.output_dir, exist_ok=True)
    total_orig = 0
    total_comp = 0

    for cid in cluster_ids:
        cluster_dir = os.path.join(args.data_dir, f'cluster_{cid}')
        output_path = os.path.join(args.output_dir, f'cluster_{cid}.npz')
        print(f"[cluster_{cid}] Compressing... ", end='', flush=True)

        try:
            stats = compress_cluster(cluster_dir, output_path,
                                     dtype=args.dtype, verify=args.verify)
            total_orig += stats['original_mb']
            total_comp += stats['compressed_mb']
            print(f"{stats['original_mb']:.0f} MB → {stats['compressed_mb']:.0f} MB "
                  f"({stats['ratio']*100:.0f}%) in {stats['time_sec']:.1f}s")
        except Exception as e:
            print(f"ERROR: {e}")

    # Copy metadata files
    if args.also_copy_meta:
        for fname in ['metadata.json', 'scaler.pkl', 'label_encoder.pkl']:
            src = os.path.join(args.data_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, args.output_dir)

    print(f"\n{'='*60}")
    print(f"COMPLETE: {total_orig:.0f} MB → {total_comp:.0f} MB "
          f"({total_comp/max(total_orig,1)*100:.0f}%)")
    print(f"Saved:    {total_orig - total_comp:.0f} MB "
          f"({(1 - total_comp/max(total_orig,1))*100:.0f}% reduction)")
    print(f"Output:   {args.output_dir}/")
    print(f"{'='*60}")
    print(f"\nTransfer command:")
    print(f"  rsync -avz --progress {args.output_dir}/ user@a100-server:~/fairpfl/data/compressed/")
    print(f"  (or: tar czf fairpfl_data.tar.gz {args.output_dir}/)")


if __name__ == '__main__':
    main()

