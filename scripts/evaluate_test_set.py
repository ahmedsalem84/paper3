#!/usr/bin/env python3
"""
evaluate_test_set.py — Held-out TEST-set evaluation from saved checkpoints.
================================================================================
During federated training, Flower's evaluate() loop scores each client on its
VALIDATION split, so the per-round metrics logged by main.py are validation
numbers. This standalone script reconstructs the final trained model from the
saved checkpoints and evaluates it on the untouched TEST split, producing a
held-out generalization estimate.

REPRODUCIBILITY CONTRACT
------------------------
This script ONLY READS existing artifacts. It does not modify main.py,
fl/strategy.py, fl/client.py, or any results/ file. It reuses the *exact* eval
path (`FairPFLClient._evaluate_impl`), swapping the val loader for the test
loader, so the test number is produced by the same code that produced the
paper's val numbers.

How the final model is reconstructed per run:
  - shared params  ← checkpoint.pkl['model_params']  (server-aggregated, round 200)
  - personal head  ← personal_params/cluster_N.pt     (per-cluster, round 200 fit)
  - config + seed  ← _ray_state.json                  (exact run configuration)

The test loader is built with the SAME per-cluster seed (seed + cid*997) and the
SAME stratified subsampling the val loader used, only with split='test'.

A built-in sanity gate re-evaluates the VAL split too: the reconstructed val
Macro-F1 must match round_metrics.json's last round (within ~1e-3). If it does,
the test number is trustworthy.

Usage:
    python scripts/evaluate_test_set.py
    python scripts/evaluate_test_set.py --data_dir data/compressed
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

# Project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sklearn.metrics import f1_score

from main import create_model
from utils.mmap_dataset import MmapNpzDataset


# (method_label, results_dir, scenario, seed)
RUNS = [
    ('FairPFL', 'results_v20_test',       'FairPFL_noDP', 42),
    ('FairPFL', 'results_v20_seed123',    'FairPFL_noDP', 123),
    ('FairPFL', 'results_v20_seed456',    'FairPFL_noDP', 456),
]
for _m in ('FedAvg', 'FedProx', 'qFFL', 'Ditto', 'pFedMe'):
    for _s in (42, 123, 456):
        RUNS.append((_m, 'results_baselines_6class', _m, _s))


def _reconstruct_model(scen_cfg, input_dim, depth, num_classes,
                       shared_params, personal_state, device):
    """Rebuild a cluster's final model from saved shared + personal params.

    The split between shared and personal differs across methods (FedPer-style
    classifier-personal for FairPFL_noDP vs. LSTM1-personal for Ditto/pFedMe).
    We let the per-cluster personal_params file *declare* which keys are personal;
    the remaining state_dict keys are filled from the checkpoint's shared params
    in state_dict order (which is exactly the order get_parameters() saved them).
    """
    model = create_model(scen_cfg, input_dim=input_dim, depth=depth,
                         num_classes=num_classes)
    sd = model.state_dict()
    personal_keys = set(personal_state.keys()) if personal_state else set()

    for k in personal_keys:
        if k in sd:
            sd[k] = personal_state[k].to(device)

    remaining = [k for k in sd.keys() if k not in personal_keys]
    if len(remaining) != len(shared_params):
        raise ValueError(
            f"shared-param count mismatch: {len(shared_params)} checkpoint "
            f"params vs {len(remaining)} non-personal state_dict keys")
    for k, v in zip(remaining, shared_params):
        sd[k] = torch.tensor(v, device=device)

    model.load_state_dict(sd)
    return model.to(device)


@torch.no_grad()
def _infer_macro_f1(model, loader, device):
    """Replicate FairPFLClient._evaluate_impl's Macro-F1 computation exactly."""
    model.eval()
    preds, labels = [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)[0]  # first element is always fine_logits
        preds.extend(logits.argmax(dim=1).cpu().numpy())
        labels.extend(y_batch.numpy())
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    return float(f1), len(labels)


def _build_eval_loader(npz_path, split, cluster_seed, cfg):
    """Replicate make_loaders' val/test loader exactly (shuffle=False)."""
    exclude = cfg.get('exclude_classes')
    if exclude is not None and not isinstance(exclude, set):
        exclude = set(exclude)
    ds = MmapNpzDataset(
        npz_path, split,
        max_samples=cfg.get('eval_max_samples'),
        seed=cluster_seed,
        seq_len=cfg.get('seq_len', 1),
        exclude_classes=exclude,
    )
    return DataLoader(ds, batch_size=cfg.get('batch_size', 128),
                      shuffle=False, drop_last=False, num_workers=0)


def evaluate_run(label, results_dir, scenario, seed, data_dir, device):
    run_dir = os.path.join(results_dir, scenario, f'seed_{seed}')
    state_path = os.path.join(run_dir, '_ray_state.json')
    ckpt_path = os.path.join(run_dir, 'checkpoint.pkl')
    rm_path = os.path.join(run_dir, 'round_metrics.json')

    with open(state_path) as f:
        state = json.load(f)
    fl_cfg = dict(state['FL_CONFIG'])
    scen_cfg = dict(state['SCENARIO_CONFIG'])
    cluster_depths = {int(k): v for k, v in state.get('CLUSTER_DEPTHS', {}).items()}
    npz_dir = data_dir or state.get('DATA_DIR') or 'data/compressed'
    personal_dir = os.path.join(run_dir, 'personal_params')

    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)
    shared_params = [np.asarray(a) for a in ckpt['model_params']]

    num_clusters = fl_cfg.get('num_clusters', 9)
    num_classes = fl_cfg.get('num_classes', 6)
    base_seed = fl_cfg.get('seed', seed)

    per_cluster = {}  # cid -> {'val': f1, 'test': f1, 'n_test': int}
    for cid in range(num_clusters):
        npz_path = os.path.join(npz_dir, f'cluster_{cid}.npz')
        if not os.path.exists(npz_path):
            continue
        cluster_seed = base_seed + cid * 997
        config = {**fl_cfg, **scen_cfg}
        depth = cluster_depths.get(cid, 1)

        # Load this cluster's personal head (declares which keys are personal)
        personal_state = None
        if scen_cfg.get('personalization', False):
            ppath = os.path.join(personal_dir, f'cluster_{cid}.pt')
            if os.path.exists(ppath):
                personal_state = torch.load(ppath, map_location=device,
                                            weights_only=True)

        results = {}
        for split in ('val', 'test'):
            loader = _build_eval_loader(npz_path, split, cluster_seed, config)
            sample_X, _ = next(iter(loader))
            input_dim = sample_X.shape[-1]
            model = _reconstruct_model(
                scen_cfg, input_dim, depth, num_classes,
                shared_params, personal_state, device)
            f1, n = _infer_macro_f1(model, loader, device)
            results[split] = f1
            if split == 'test':
                results['n_test'] = n
        per_cluster[cid] = results

    val_f1s = [v['val'] for v in per_cluster.values()]
    test_f1s = [v['test'] for v in per_cluster.values()]
    avg_val = float(np.mean(val_f1s))
    avg_test = float(np.mean(test_f1s))
    worst_test = float(np.min(test_f1s))

    # Sanity gate: reconstructed val vs reported round_metrics last round
    reported_val = None
    if os.path.exists(rm_path):
        with open(rm_path) as f:
            rm = json.load(f)
        if rm:
            reported_val = rm[-1].get('avg_f1')

    return {
        'method': label, 'seed': seed,
        'avg_val_f1_reconstructed': avg_val,
        'avg_val_f1_reported': reported_val,
        'sanity_gap': (abs(avg_val - reported_val) if reported_val is not None else None),
        'avg_test_f1': avg_test,
        'worst_cluster_test_f1': worst_test,
        'per_cluster': {str(k): v for k, v in per_cluster.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=None,
                    help='Override npz dir (default: DATA_DIR from _ray_state.json)')
    ap.add_argument('--out', default='paper/test_set_evaluation.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    rows = []
    for label, results_dir, scenario, seed in RUNS:
        run_dir = os.path.join(results_dir, scenario, f'seed_{seed}')
        if not os.path.exists(os.path.join(run_dir, 'checkpoint.pkl')):
            print(f"[skip] {label} seed={seed}: no checkpoint at {run_dir}")
            continue
        print(f"[eval] {label:8s} seed={seed} ...", flush=True)
        try:
            r = evaluate_run(label, results_dir, scenario, seed,
                             args.data_dir, device)
            rows.append(r)
            gap = r['sanity_gap']
            gap_str = f"{gap:.4f}" if gap is not None else "n/a"
            flag = "OK" if (gap is not None and gap < 5e-3) else "CHECK"
            print(f"        val(recon)={r['avg_val_f1_reconstructed']:.4f} "
                  f"val(reported)={r['avg_val_f1_reported']} "
                  f"gap={gap_str} [{flag}]  ->  TEST={r['avg_test_f1']:.4f} "
                  f"(worst={r['worst_cluster_test_f1']:.4f})")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"        FAILED: {e}")

    # Aggregate per method: mean +/- std across seeds (test F1)
    print("\n" + "=" * 64)
    print("TEST-SET MACRO-F1 SUMMARY (mean +/- std across seeds)")
    print("=" * 64)
    by_method = {}
    for r in rows:
        by_method.setdefault(r['method'], []).append(r)
    summary = {}
    for method, rs in by_method.items():
        tests = [x['avg_test_f1'] for x in rs]
        vals = [x['avg_val_f1_reconstructed'] for x in rs]
        worsts = [x['worst_cluster_test_f1'] for x in rs]
        summary[method] = {
            'test_f1_mean': float(np.mean(tests)),
            'test_f1_std': float(np.std(tests)),
            'val_f1_mean': float(np.mean(vals)),
            'worst_test_f1_mean': float(np.mean(worsts)),
            'n_seeds': len(rs),
        }
        print(f"  {method:8s}  TEST={np.mean(tests):.4f} +/- {np.std(tests):.4f}"
              f"   (VAL={np.mean(vals):.4f})  worst-cluster TEST={np.mean(worsts):.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'runs': rows, 'summary': summary}, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == '__main__':
    main()
