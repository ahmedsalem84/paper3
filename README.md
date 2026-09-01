# FairPFL

Reference implementation for **"FairPFL: A Unified Fairness-Privacy-Personalization Framework for Federated IoT Intrusion Detection."**

FairPFL is a federated learning framework for IoT intrusion detection that treats fairness as an operational requirement: no device population should be left systematically under-protected by the shared detector. It combines a self-correcting fairness controller in the aggregation rule, a split personalization architecture, and a selective noise mechanism that perturbs only the shared backbone.

This repository contains everything needed to reproduce the experiments. Datasets, trained checkpoints and result directories are **not** included; the sections below explain how to regenerate them.

---

## 1. What is here

```
main.py                  single entry point; --scenario selects the method
config.yaml              the paper's configuration (documentation + defaults)
fl/
  baselines.py           scenario dictionaries: FairPFL, baselines, ablations
  client.py              local training, evaluation, selective DP noise
  strategy.py            FairPFLStrategy: fairness-weighted aggregation, q-controller
models/
  fairpfl_model.py       split LSTM + attention model (shared / personal params)
  fairpfl_model_dp.py    DP-compatible variant (Opacus path)
  lstm_baseline.py       BaselineLSTM used by FedAvg / FedProx / qFFL
  focal_loss.py          focal loss and class-weighting
  dual_personalizer.py   dual-head personalization components
data/
  preprocessing.py       raw CIC-IoT-2023 CSVs -> per-cluster .npy (two-pass)
utils/
  compress_data.py       per-cluster .npy -> compressed .npz used in training
  mmap_dataset.py        memory-mapped cluster loader, label remapping
  depth_selector.py      per-cluster personalization depth from JS divergence
  checkpoint.py          checkpoint save / resume
  dp_opacus.py           Opacus privacy engine wiring
metrics/                 evaluation and fairness metrics
analysis/                statistical tests, plots, LaTeX table generation
scripts/
  preprocess_edge_iiot.py  Edge-IIoTset preprocessing (second corpus)
  evaluate_test_set.py     held-out TEST-split evaluation from checkpoints
  paper_figures.py         figures
run_baselines_6class.sh  baselines, 3 seeds
run_v20_ablation.sh      ablations A1-A5, 3 seeds
run_v20_dp.sh            differential-privacy experiments
run_edge_iiot*.sh        cross-dataset replication
```

There is no build step and no formal test suite; everything runs through `main.py`.

---

## 2. Environment

Tested with Python 3.10-3.12, CUDA 12.1, on a single GPU.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

A smoke test that needs no dataset and verifies the pipeline end to end:

```bash
python3 main.py --scenario FairPFL_noDP --seed 42 --rounds 3 --synthetic
```

---

## 3. Data

**CIC-IoT-2023.** Download the CSV release from the University of New Brunswick and point
`configs/default.yaml` at it (`data.raw_dir`). Preprocessing runs in two stages.

```bash
# 1. raw CSVs -> per-cluster .npy files under data/processed
#    (two-pass: fits scaler and label encoder, then transforms and writes)
python3 data/preprocessing.py --config configs/default.yaml
#    add --sample 0.01 for a fast 1% dry run

# 2. per-cluster .npy -> compressed .npz archives used by the training code
python3 utils/compress_data.py --data_dir data/processed --output_dir data/compressed --verify
```

This produces `data/compressed/cluster_{0..8}.npz` (nine device-type clusters, each holding
train/val/test splits) plus `metadata.json`. Labels are remapped at load time from the raw
34 classes to the 8 UNB categories; passing `--exclude-minority` further reduces this to the
6 contiguous classes used throughout the paper (BruteForce and Web-based are dropped because
several clusters hold too few samples per cluster for a stable per-cluster F1).

**Edge-IIoTset.** Download the `DNN-EdgeIIoT-dataset.csv` release, then:

```bash
python3 scripts/preprocess_edge_iiot.py --csv <path_to_csv> --out data_edge_iiot/compressed \
        --clusters 9 --alpha 0.5 --seed 42
```

This produces nine clusters under a label-Dirichlet partition over the 15 Edge-IIoTset classes.
Edge-IIoTset labels are already contiguous, so those runs pass `--no-remap` (the provided
scripts do this for you).

---

## 4. Reproducing the paper

Every experiment writes `round_metrics.json` under `{results_dir}/{scenario}/seed_{seed}/`.
That file is the experiment output; all tables in the paper derive from it.

**Main comparison (FairPFL, three seeds).**

```bash
python3 main.py --scenario FairPFL_noDP --seed 42 --rounds 200 \
    --data_dir data/compressed --batch_size 128 \
    --eval_every_n_rounds 5 --eval_max_samples 100000 \
    --exclude-minority --num_cpus 1 --results_dir results_v20_test

python3 main.py --scenario FairPFL_noDP --seed 123 --rounds 200 \
    --data_dir data/compressed --batch_size 128 \
    --eval_every_n_rounds 5 --eval_max_samples 100000 \
    --exclude-minority --num_cpus 1 --results_dir results_v20_seed123

python3 main.py --scenario FairPFL_noDP --seed 456 --rounds 200 \
    --data_dir data/compressed --batch_size 128 \
    --eval_every_n_rounds 5 --eval_max_samples 100000 \
    --exclude-minority --num_cpus 1 --results_dir results_v20_seed456
```

The three directory names are not arbitrary: the analysis scripts look for exactly
these, so using them here means the post-hoc steps below work without editing.

**Baselines, ablations, differential privacy.**

```bash
bash run_baselines_6class.sh    # FedAvg, FedProx, qFFL, Ditto, pFedMe x 3 seeds
bash run_v20_ablation.sh        # A1-A5 x 3 seeds
bash run_v20_dp.sh              # FairPFL_DP and FedAvg_DP
```

**Cross-dataset replication.**

```bash
bash run_edge_iiot.sh
bash run_edge_iiot_baselines.sh
```

**Post-hoc analysis.**

```bash
python3 scripts/evaluate_test_set.py --out test_set_evaluation.json
python3 compute_fairness_metrics.py      # fairness panel (Tables 6-8)
python3 compute_paper_metrics.py         # Jain, bottom-quintile, per-class, t-tests
python3 scripts/paper_figures.py         # figures 3-6, written to paper/figures/
```

These three read results by path rather than by flag, and they expect the directory
names used above:

| script reads | directory |
|---|---|
| FairPFL, three seeds | `results_v20_test/`, `results_v20_seed123/`, `results_v20_seed456/` |
| baselines | `results_baselines_6class/` |
| ablations | `results_v20_ablation/` |
| DP runs | `results_v20_dp/` |
| sensitivity sweep | `results_sensitivity/` |

If you used different names, edit the run map at the top of each script. Each one stops
with an explanatory message if it finds no results, rather than emitting an empty table
or a blank figure.

The sensitivity sweep (`results_sensitivity/`) has no runner script, because only `q_max`
is exposed on the command line (`--q`); the proximal coefficient and the warmup length are
read from the scenario dictionary. To reproduce those nine cells, copy the `FairPFL_noDP`
entry in `fl/baselines.py`, change `proximal_mu` or `personal_warmup_rounds`, and run seed 42
into `results_sensitivity/<name>/`, using the names `qmax_1.0`, `qmax_2.0`, `qmax_5.0`,
`mu_0.000`, `mu_0.010`, `mu_0.100`, `warmup_0`, `warmup_10`, `warmup_40`. Without them
`compute_paper_metrics.py` reports that block as MISSING and still produces everything else.

Expect roughly 110 GPU-hours in total: a FairPFL run takes about 2.8 h over 200 rounds,
non-personalized baselines 1.3-1.8 h, and DP runs 2.4-2.7 h.

---

## 5. Things worth knowing before you run

**Runs resume, they do not restart.** `FairPFLStrategy` writes `checkpoint.pkl` and
`round_metrics.json` after every evaluation round, and startup auto-resumes from the saved
round. Re-running a command with an existing `--results_dir` therefore continues that run.
To start clean, use a fresh directory.

**Scenario dictionaries are the single switch.** `--scenario` selects a configuration dict in
`fl/baselines.py`, and that dict is what turns one codebase into FairPFL, a baseline, or an
ablation. To change what an experiment does, edit its dict there rather than branching on the
scenario name elsewhere.

**Ray workers do not inherit module globals.** Training runs under `flwr.simulation` with one
Ray actor per cluster, in separate processes. `main.py` therefore serializes the shared state
to `results_dir/_ray_state.json` and each worker restores it in `client_fn`. Any new global
that clients need must be added to both the serialization block and the restore function, or
workers will silently see `None`.

**The shared/personal split is defined by methods, not by module names.** In `FairPFLModel`,
`get_shared_params()` and `get_personal_params()` define the split; the modules named
`personal.*` are in fact shared, and only the classifier head stays on device. Personal
parameters persist across rounds on disk because Flower recreates a fresh client each round.

**Three configuration sources.** `config.yaml` documents the paper's intended configuration,
the `FL_CONFIG` dict at the top of `main.py` holds the runtime defaults, and CLI flags plus
the scenario dict override both. If a value in `config.yaml` appears not to take effect, check
whether it is actually consumed by `main.py`. Paper runs always pass `--exclude-minority`
and `--batch_size 128`.

**Determinism.** Results are statistically reproducible rather than bit-exact: cuDNN LSTM
kernels are non-deterministic, and floating-point accumulation order differs between CPU and
GPU. Per-cluster F1 reproduces to within about 1e-3 across devices, which is well below the
effects the paper reports.

---

## 6. Privacy accounting

The differential-privacy runs use the manual mode by default: gradients are clipped and
Gaussian noise is added to the **shared** parameters only, after the backward pass, so the
personal head never receives noise. The cumulative privacy budget reported in the paper is
computed offline with Renyi DP composition and is presented as a noise budget rather than a
deployment-grade privacy guarantee.

---

## 7. Citation

Please cite the paper if you use this code. Citation details will be added on acceptance.
