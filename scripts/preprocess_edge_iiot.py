#!/usr/bin/env python3
"""Preprocess Edge-IIoTset (DNN-EdgeIIoT-dataset.csv) into the FairPFL cluster
format: data_edge_iiot/compressed/cluster_{0..8}.npz + metadata.json.

Mirrors the CIC-IoT-2023 pipeline so FairPFL/baselines run unchanged on a
second dataset: 9 clients via a label-Dirichlet(alpha) non-IID partition,
per-client stratified 70/15/15 split, StandardScaler features, int label codes.

Usage:
    python scripts/preprocess_edge_iiot.py \
        --csv /path/to/DNN-EdgeIIoT-dataset.csv \
        --out data_edge_iiot/compressed \
        --clusters 9 --alpha 0.5 --seed 42
    # add --collapse6 to fold 15 attack types into 6 threat categories
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Identifier / leakage columns dropped in Ferrag et al.'s own reference
# preprocessing (free-text payloads, IPs, timestamps, ports). Only those
# actually present are removed.
DROP_COLS = [
    "frame.time", "ip.src_host", "ip.dst_host",
    "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
    "http.file_data", "http.request.full_uri", "http.request.uri.query",
    "http.referer", "http.request.version", "http.response",
    "tcp.options", "tcp.payload", "tcp.srcport", "tcp.dstport",
    "udp.port", "mqtt.msg", "mqtt.topic", "dns.qry.name.len",
    "mqtt.conack.flags", "mqtt.protoname",
]

LABEL_COL = "Attack_type"          # 15-class multiclass target
BINARY_LABEL_COL = "Attack_label"  # dropped: binary leakage for multiclass

# 5 threat categories + Normal (Ferrag taxonomy) for optional --collapse6.
COLLAPSE6 = {
    "Normal": "Normal",
    "DDoS_UDP": "DoS_DDoS", "DDoS_ICMP": "DoS_DDoS",
    "DDoS_TCP": "DoS_DDoS", "DDoS_HTTP": "DoS_DDoS",
    "Port_Scanning": "Recon", "Vulnerability_scanner": "Recon",
    "Fingerprinting": "Recon",
    "SQL_injection": "Injection", "XSS": "Injection", "Uploading": "Injection",
    "Password": "Injection",
    "Backdoor": "Malware", "Ransomware": "Malware",
    "MITM": "MITM",
}


def clean(df, collapse6):
    drop = [c for c in DROP_COLS + [BINARY_LABEL_COL] if c in df.columns]
    df = df.drop(columns=drop)
    df = df.dropna(subset=[LABEL_COL])
    y_raw = df[LABEL_COL].astype(str).str.strip()
    if collapse6:
        unknown = set(y_raw.unique()) - set(COLLAPSE6)
        if unknown:
            raise ValueError(f"--collapse6: unmapped Attack_type values {unknown}")
        y_raw = y_raw.map(COLLAPSE6)
    X = df.drop(columns=[LABEL_COL])

    # Safety net: drop any high-cardinality object column that slipped past
    # DROP_COLS (stray IDs / free text) so get_dummies can't explode width.
    hi_card = [c for c in X.select_dtypes(include=["object"]).columns
               if X[c].nunique() > 50]
    if hi_card:
        print(f"  dropping high-cardinality object cols: {hi_card}")
        X = X.drop(columns=hi_card)

    # Encode remaining low-cardinality categoricals; coerce everything numeric.
    obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        print(f"  one-hot encoding: {obj_cols}")
        X = pd.get_dummies(X, columns=obj_cols, dummy_na=False)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Drop zero-variance columns (constant -> no signal, breaks scaling).
    nunique = X.nunique()
    X = X.loc[:, nunique[nunique > 1].index]
    return X, y_raw


def dirichlet_partition(y, n_clients, alpha, rng):
    """Label-Dirichlet non-IID split: each class spread over clients by a
    Dirichlet(alpha) draw. Returns list of index arrays, one per client."""
    idx_by_client = [[] for _ in range(n_clients)]
    for c in np.unique(y):
        c_idx = np.where(y == c)[0]
        rng.shuffle(c_idx)
        props = rng.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(props)[:-1] * len(c_idx)).astype(int)
        for k, part in enumerate(np.split(c_idx, cuts)):
            idx_by_client[k].extend(part.tolist())
    return [np.array(sorted(ix)) for ix in idx_by_client]


def split_save(X, y, idx_by_client, out_dir, class_names, alpha, seed):
    os.makedirs(out_dir, exist_ok=True)
    stats = {}
    for k, ix in enumerate(idx_by_client):
        Xc, yc = X[ix], y[ix]
        # stratify only where every class has >=2 samples in this client
        strat = yc if np.min(np.bincount(yc, minlength=len(class_names))[
            np.unique(yc)]) >= 2 else None
        Xtr, Xtmp, ytr, ytmp = train_test_split(
            Xc, yc, test_size=0.30, random_state=seed, stratify=strat)
        strat2 = ytmp if (strat is not None and np.min(np.bincount(
            ytmp, minlength=len(class_names))[np.unique(ytmp)]) >= 2) else None
        Xv, Xte, yv, yte = train_test_split(
            Xtmp, ytmp, test_size=0.50, random_state=seed, stratify=strat2)
        np.savez_compressed(
            os.path.join(out_dir, f"cluster_{k}.npz"),
            X_train=Xtr.astype(np.float32), y_train=ytr.astype(np.int32),
            X_val=Xv.astype(np.float32), y_val=yv.astype(np.int32),
            X_test=Xte.astype(np.float32), y_test=yte.astype(np.int32))
        stats[str(k)] = {"n": int(len(ix)), "train": int(len(ytr)),
                         "val": int(len(yv)), "test": int(len(yte)),
                         "class_counts": np.bincount(
                             yc, minlength=len(class_names)).tolist()}
        print(f"  cluster_{k}: n={len(ix):>8} train={len(ytr):>8} "
              f"val={len(yv):>7} test={len(yte):>7}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="data_edge_iiot/compressed")
    ap.add_argument("--clusters", type=int, default=9)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--collapse6", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"Loading {args.csv} ...")
    df = pd.read_csv(args.csv, low_memory=False)
    print(f"  raw shape: {df.shape}")

    X_df, y_raw = clean(df, args.collapse6)
    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(np.int64)
    class_names = le.classes_.tolist()
    feature_cols = X_df.columns.tolist()
    print(f"  features={len(feature_cols)} classes={len(class_names)} "
          f"rows={len(y)}")
    print(f"  class_names={class_names}")

    X = StandardScaler().fit_transform(X_df.values.astype(np.float64)).astype(
        np.float32)

    print(f"Dirichlet partition: {args.clusters} clients, alpha={args.alpha}")
    idx_by_client = dirichlet_partition(y, args.clusters, args.alpha, rng)
    stats = split_save(X, y, idx_by_client, args.out, class_names,
                       args.alpha, args.seed)

    meta = {
        "feature_cols": feature_cols, "num_features": len(feature_cols),
        "num_classes": len(class_names), "class_names": class_names,
        "num_clusters": args.clusters, "alpha": args.alpha,
        "total_records": int(len(y)), "cluster_stats": stats,
        "source": "Edge-IIoTset DNN-EdgeIIoT-dataset.csv",
        "collapse6": args.collapse6, "seed": args.seed,
    }
    with open(os.path.join(args.out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Done -> {args.out} (input_dim={len(feature_cols)}, "
          f"num_classes={len(class_names)})")


if __name__ == "__main__":
    main()
