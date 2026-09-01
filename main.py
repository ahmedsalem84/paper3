# main.py — Single entry point for ALL FairPFL experiments
# Usage:
#   python main.py --scenario FairPFL_noDP --seed 42 --rounds 100
#   python main.py --scenario S0 --seed 42 --rounds 100
#   python main.py --scenario FairPFL --q 2.0 --seed 42

import argparse
import json
import os
import sys
import time
import pickle

import torch
import numpy as np
import flwr as fl
from flwr.simulation import start_simulation
from flwr.common import Context
from flwr.common import ndarrays_to_parameters

from utils.device_utils import get_device, set_seed
from utils.data_utils import create_dataloaders
from utils.checkpoint import ExperimentLogger
from utils.depth_selector import select_personalization_depth
from models.fairpfl_model import FairPFLModel, AdaptiveFairPFLModel
from models.lstm_baseline import BaselineLSTM
from models.dual_personalizer import DualFairPFLModel, DualFairPFLModelWithLLM
from fl.client import FairPFLClient
from fl.strategy import FairPFLStrategy
from fl.baselines import BASELINE_CONFIGS, ABLATION_CONFIGS

# DP models — optional, loaded lazily
try:
    from models.fairpfl_model_dp import FairPFLModelDP, DualFairPFLModelDP
    DP_MODELS_AVAILABLE = True
except ImportError:
    DP_MODELS_AVAILABLE = False


# ============================================================
# DEFAULT FL CONFIG
# ============================================================
FL_CONFIG = {
    'num_clusters': 9,
    'num_rounds': 100,
    'local_epochs': 5,
    'batch_size': 512,
    'learning_rate_personal': 0.001,
    'learning_rate_shared': 0.001,
    'participation_rate': 0.67,  # BUG-1 FIX: was 0.5 → only 4/9 clusters trained/eval'd
    'proximal_mu': 0.01,
    'fairness_q_init': 1.0,
    'fairness_q_delta': 0.1,
    'fairness_q_max': 3.0,        # v12c test: 2.0 hurt performance → reverted to 3.0
    # dp_epsilon_local removed — personal params never transmitted, no DP noise needed
    'dp_epsilon_global': 5.0,
    'dp_delta': 1e-5,
    'dp_clip_norm_personal': 1.0,
    'dp_clip_norm_shared': 1.0,
    'fairness_delta1': 0.05,
    'fairness_delta2': 0.02,  # lower F1-gap threshold for q relaxation (dead-zone [0.02,0.05])
    'seq_len': 1,  # CIC-IoT-2023 is flow-level: each row is independent, no temporal order
    'seed': 42,
    'num_classes': 8,
    'personalization': True,
}


# Global state for Flower simulation
CLUSTER_LOADERS = None
DATA_DIR = None
SCENARIO_CONFIG = None
DEVICE = None
CLUSTER_DEPTHS = {}  # {cluster_id: depth} for adaptive depth
PERSONAL_PARAMS_DIR = None  # directory for per-cluster personal-param checkpoints
_STATE_FILE = None  # Path to serialized state for Ray actor communication


def create_model(scenario_config, input_dim=46, depth=1, num_classes=8):
    """Create model based on scenario configuration.
    
    Args:
        scenario_config: dict with model_type, dp, adaptive_depth, etc.
        input_dim: feature dimension (46 for CIC-IoT-2023, or 110 with LLM)
        depth: personalization depth (1/2/3) for AdaptiveFairPFLModel
        num_classes: number of output classes (from FL_CONFIG, default 8)
    """
    model_type = scenario_config.get('model_type', 'fairpfl')
    use_dp = scenario_config.get('dp', False)

    if model_type == 'baseline':
        return BaselineLSTM(input_dim=input_dim, num_classes=num_classes)
    elif model_type == 'fairpfl':
        dp_mode = scenario_config.get('dp_mode', 'opacus')
        if use_dp and dp_mode == 'opacus' and DP_MODELS_AVAILABLE:
            return FairPFLModelDP(input_dim=input_dim, num_classes=num_classes)
        # dp_mode='manual' → use normal FairPFLModel + manual gradient noise
        # IMPORTANT: FairPFLModel and AdaptiveFairPFLModel have DIFFERENT state_dict
        # structures (different module names). All clusters in one experiment MUST use
        # the same model class — mixing them causes size-mismatch errors on state_dict load.
        use_adaptive = scenario_config.get('adaptive_depth', False)
        if use_adaptive:
            # All clusters use AdaptiveFairPFLModel (depth varies per cluster: 1/2/3)
            return AdaptiveFairPFLModel(depth=depth, input_dim=input_dim,
                                        num_classes=num_classes)
        else:
            # All clusters use FairPFLModel (fixed single-depth personalization)
            return FairPFLModel(input_dim=input_dim, num_classes=num_classes)
    elif model_type == 'dual':
        if use_dp and DP_MODELS_AVAILABLE:
            return DualFairPFLModelDP(input_dim=input_dim)
        return DualFairPFLModel(input_dim=input_dim)
    elif model_type == 'dual_llm':
        return DualFairPFLModelWithLLM(raw_dim=input_dim)
    else:
        return FairPFLModel(input_dim=input_dim)


def _restore_globals_from_state_file():
    """Restore global state in Ray worker processes.

    Flower 1.13+ / Ray 2.51+ runs ClientAppActor in separate processes
    that do NOT inherit Python globals from the main process. This function
    reads serialized state from a shared file so that client_fn can access
    DATA_DIR, FL_CONFIG, SCENARIO_CONFIG, etc.
    """
    global DATA_DIR, FL_CONFIG, SCENARIO_CONFIG, CLUSTER_DEPTHS, PERSONAL_PARAMS_DIR
    if DATA_DIR is not None:
        return  # Already restored

    # Find state file: try global var first, then environment variable
    state_file = _STATE_FILE or os.environ.get('FAIRPFL_STATE_FILE')
    if state_file is None:
        return  # No state file available
    try:
        import json as _json
        with open(state_file, 'r') as f:
            state = _json.load(f)
        DATA_DIR = state.get('DATA_DIR')
        FL_CONFIG.update(state.get('FL_CONFIG', {}))
        # Restore exclude_classes from list (JSON) back to set
        ec = FL_CONFIG.get('exclude_classes')
        if ec is not None and isinstance(ec, list):
            FL_CONFIG['exclude_classes'] = set(ec)
        SCENARIO_CONFIG = state.get('SCENARIO_CONFIG')
        CLUSTER_DEPTHS = {int(k): v for k, v in state.get('CLUSTER_DEPTHS', {}).items()}
        PERSONAL_PARAMS_DIR = state.get('PERSONAL_PARAMS_DIR')
    except Exception as e:
        print(f"[_restore_globals] Warning: {e}")


def client_fn(context: Context):
    """Factory function for Flower simulation.

    Creates a fresh client each round. In FL, model parameters flow via
    get/set_parameters. Not caching clients avoids OOM from storing all
    clusters' data tensors in actor memory (~12.6GB for 9 clusters).
    """
    import gc
    # Restore globals in Ray worker (no-op if already set)
    _restore_globals_from_state_file()

    # Use Flower's partition_id for unique cluster assignment (0..num_clusters-1).
    # WARNING: Do NOT use node_id % num_clusters — Flower 1.27 assigns large
    # random node_ids (18-digit ints) that cause hash collisions under modulo.
    # partition_id is guaranteed unique by Flower's internal mapping.
    cluster_id = int(context.node_config['partition-id'])

    # Load data for this cluster — lazy memory-mapped loading
    class_weights_list = None
    if CLUSTER_LOADERS is not None:
        # Pre-cached loaders (e.g. for synthetic mode)
        loaders = CLUSTER_LOADERS[cluster_id]
    else:
        # === Lazy-loaded path using MmapNpzDataset ===
        # np.load(mmap_mode='r') → disk-backed, only touched pages enter RAM.
        # Each Ray actor holds a ~1MB mmap handle instead of a ~300MB array copy.
        from utils.mmap_dataset import make_loaders, compute_class_weights_from_npz

        if DATA_DIR is None:
            raise RuntimeError(
                f"[client_fn] DATA_DIR is None but CLUSTER_LOADERS is also None "
                f"for cluster {cluster_id}. This should not happen — "
                f"did Ray actor lose global state?")

        batch_size = FL_CONFIG.get('batch_size', 512)
        max_samples = FL_CONFIG.get('max_samples_per_cluster', None)
        eval_max_samples = FL_CONFIG.get('eval_max_samples', None)
        seed = FL_CONFIG.get('seed', 42)
        cluster_seed = seed + cluster_id * 997

        npz_path = os.path.join(DATA_DIR, f'cluster_{cluster_id}.npz')
        use_npz = os.path.exists(npz_path)

        if use_npz:
            # .npz path: fully memory-mapped (no full-array RAM copy)
            seq_len = FL_CONFIG.get('seq_len', 5)
            exclude_classes = FL_CONFIG.get('exclude_classes', None)
            loaders = make_loaders(
                npz_path=npz_path,
                batch_size=batch_size,
                max_train=max_samples,
                max_eval=eval_max_samples,
                seed=cluster_seed,
                seq_len=seq_len,
                exclude_classes=exclude_classes,
                remap=FL_CONFIG.get('remap_labels', True),
            )
            # Class weights via mmap — only reads y_train, not X_train
            num_classes_cw = FL_CONFIG.get('num_classes', 8)
            class_weights_list = compute_class_weights_from_npz(
                npz_path=npz_path,
                num_classes=num_classes_cw,
                max_samples=max_samples,
                seed=cluster_seed,
            )
        else:
            # Legacy .npy fallback with mmap
            import numpy as _np
            from torch.utils.data import DataLoader as DL, TensorDataset as TDS
            from models.focal_loss import compute_class_weights
            from utils.mmap_dataset import _remap_labels
            exclude_classes = FL_CONFIG.get('exclude_classes', None)
            cluster_dir = os.path.join(DATA_DIR, f'cluster_{cluster_id}')
            rng = _np.random.default_rng(seed=cluster_seed)
            loaders = {}
            for split in ['train', 'val', 'test']:
                X_mm = _np.load(os.path.join(cluster_dir, f'X_{split}.npy'), mmap_mode='r')
                y_mm = _np.load(os.path.join(cluster_dir, f'y_{split}.npy'), mmap_mode='r')
                N = len(X_mm)
                ms = max_samples if split == 'train' else eval_max_samples
                if ms and N > ms:
                    idx = _np.sort(rng.choice(N, ms, replace=False))
                    X, y = X_mm[idx].copy(), y_mm[idx].copy()
                else:
                    X, y = _np.array(X_mm), _np.array(y_mm)
                # Remap 34-class labels to 8-class, then optionally to 6-class
                if exclude_classes:
                    y, keep_mask = _remap_labels(y, exclude_classes=exclude_classes)
                    X = X[keep_mask]
                else:
                    y = _remap_labels(y)
                if split == 'train':
                    class_weights_list = compute_class_weights(
                        y, FL_CONFIG.get('num_classes', 8)).tolist()
                loaders[split] = DL(
                    TDS(torch.FloatTensor(X).unsqueeze(1), torch.LongTensor(y)),
                    batch_size=batch_size, shuffle=(split == 'train'), drop_last=False)
                del X, y
        gc.collect()

    sample_X, _ = next(iter(loaders['train']))
    input_dim = sample_X.shape[-1]
    depth = CLUSTER_DEPTHS.get(cluster_id, 1)
    num_classes = FL_CONFIG.get('num_classes', 8)
    model = create_model(SCENARIO_CONFIG, input_dim=input_dim, depth=depth,
                         num_classes=num_classes)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.to(device)
    else:
        device = torch.device('cpu')

    config = {**FL_CONFIG, **SCENARIO_CONFIG}
    if class_weights_list is not None:
        config['class_weights'] = class_weights_list

    # class_counts for LogitAdjustedCE: use balanced training counts
    # (equal per-class sampling gives 2000/class)
    num_classes_cfg = FL_CONFIG.get('num_classes', 8)
    samples_per_class = FL_CONFIG.get('samples_per_class', 2000)
    config['class_counts'] = [float(samples_per_class)] * num_classes_cfg

    # Compute path for persisting this cluster's personal params between rounds
    personal_params_path = None
    if (PERSONAL_PARAMS_DIR and SCENARIO_CONFIG and
            SCENARIO_CONFIG.get('personalization', False)):
        personal_params_path = os.path.join(
            PERSONAL_PARAMS_DIR, f"cluster_{cluster_id}.pt")

    return FairPFLClient(
        cluster_id=cluster_id,
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        config=config,
        device=str(device),
        personal_params_path=personal_params_path,
    ).to_client()


def run_experiment(args):
    """Run a single experiment with given configuration."""
    global CLUSTER_LOADERS, SCENARIO_CONFIG, DEVICE, FL_CONFIG, DATA_DIR, \
        PERSONAL_PARAMS_DIR

    # Setup
    DEVICE = get_device()
    set_seed(args.seed)
    FL_CONFIG['seed'] = args.seed
    FL_CONFIG['num_rounds'] = args.rounds

    # Determine scenario config
    scenario = args.scenario
    if scenario in BASELINE_CONFIGS:
        SCENARIO_CONFIG = BASELINE_CONFIGS[scenario]
    elif scenario in ABLATION_CONFIGS:
        SCENARIO_CONFIG = ABLATION_CONFIGS[scenario]
    else:
        SCENARIO_CONFIG = BASELINE_CONFIGS.get('FairPFL_noDP', {})

    # Override q and epsilon if specified
    if args.q is not None:
        FL_CONFIG['fairness_q_init'] = args.q
        FL_CONFIG['fairness_q_max'] = args.q   # Fixed q: cap = init → true fixed-q ablation
        SCENARIO_CONFIG['fairness_q_init'] = args.q
        SCENARIO_CONFIG['fairness_q_max'] = args.q  # Prevents adaptive controller from overshooting
    if args.epsilon is not None:
        FL_CONFIG['dp_epsilon_global'] = args.epsilon
    if args.dirichlet_alpha is not None:
        FL_CONFIG['dirichlet_alpha'] = args.dirichlet_alpha

    # Update num_clusters
    num_clusters = args.num_clusters or FL_CONFIG['num_clusters']
    FL_CONFIG['num_clusters'] = num_clusters

    # Directory for persisting personal params between rounds
    # Keyed by scenario+seed so different experiments don't share state
    PERSONAL_PARAMS_DIR = os.path.join(
        args.results_dir, scenario, f"seed_{args.seed}", 'personal_params')

    # Activate adaptive depth selection per cluster
    global CLUSTER_DEPTHS
    CLUSTER_DEPTHS = {}
    if SCENARIO_CONFIG.get('adaptive_depth', True) and \
            SCENARIO_CONFIG.get('personalization', False):
        try:
            from utils.depth_selector import select_personalization_depth
            import numpy as _np
            data_dir = args.data_dir
            cluster_data = {}
            all_labels = []
            for cid in range(num_clusters):
                npz_path = os.path.join(data_dir, f'cluster_{cid}.npz')
                npy_path = os.path.join(data_dir, f'cluster_{cid}', 'y_train.npy')
                if os.path.exists(npz_path):
                    d = _np.load(npz_path)
                    y = d['y_train']
                elif os.path.exists(npy_path):
                    y = _np.load(npy_path, mmap_mode='r')
                else:
                    continue
                cluster_data[cid] = {'train_y': _np.array(y)}
                all_labels.append(_np.array(y))
            if cluster_data and all_labels:
                global_labels = _np.concatenate(all_labels)
                CLUSTER_DEPTHS = select_personalization_depth(
                    cluster_data, global_labels)
                print(f"Adaptive depth: {CLUSTER_DEPTHS}")
            del cluster_data, all_labels
        except Exception as e:
            print(f"Adaptive depth selection failed ({e}), using depth=1")

    # Subsampling limit: 0 means use full dataset
    max_samples = getattr(args, 'max_samples', 20000)
    FL_CONFIG['max_samples_per_cluster'] = max_samples if max_samples > 0 else None

    # Eval max samples: 0 means use full val set
    eval_max = getattr(args, 'eval_max_samples', 0)
    FL_CONFIG['eval_max_samples'] = eval_max if eval_max > 0 else None

    # Eval frequency: evaluate every N rounds (1 = every round)
    FL_CONFIG['eval_every_n_rounds'] = getattr(args, 'eval_every_n_rounds', 1)

    # Batch size override (for A100 scaling)
    if getattr(args, 'batch_size', None):
        FL_CONFIG['batch_size'] = args.batch_size

    # 6-class mode: exclude BruteForce and Web-based (insufficient FL samples)
    if getattr(args, 'exclude_minority', False):
        from utils.mmap_dataset import EXCLUDE_CLASSES_6, CLASSES_6
        FL_CONFIG['exclude_classes'] = EXCLUDE_CLASSES_6  # {1, 7}
        FL_CONFIG['num_classes'] = 6
        print(f"6-CLASS MODE: Excluding BruteForce & Web-based")
        print(f"  Classes: {CLASSES_6}")
    else:
        FL_CONFIG['exclude_classes'] = None

    # Second-dataset support (e.g. Edge-IIoTset): explicit class count + no remap.
    # Defaults preserve CIC behavior (remap on, num_classes via logic above).
    if getattr(args, 'no_remap', False):
        FL_CONFIG['remap_labels'] = False
        print("LABEL REMAP DISABLED (non-CIC dataset: labels used as-is)")
    if getattr(args, 'num_classes', 0) > 0:
        FL_CONFIG['num_classes'] = args.num_classes
        print(f"NUM_CLASSES override: {args.num_classes}")

    print(f"\n{'=' * 60}")
    print(f"RUNNING: {scenario} | seed={args.seed} | rounds={args.rounds}")
    num_cls = FL_CONFIG.get('num_classes', 8)
    print(f"Config: q={FL_CONFIG.get('fairness_q_init')}, "
          f"ε={FL_CONFIG.get('dp_epsilon_global')}, classes={num_cls}")
    ms = FL_CONFIG['max_samples_per_cluster']
    print(f"Data: max_samples/cluster={ms if ms else 'FULL'}, "
          f"num_cpus/client={getattr(args, 'num_cpus', 4)}")
    print(f"{'=' * 60}\n")

    # Load data
    if args.synthetic:
        print("Using SYNTHETIC data for testing...")
        CLUSTER_LOADERS = _create_synthetic_loaders(
            num_clusters, FL_CONFIG['batch_size'])
        DATA_DIR = None  # Signal: use pre-loaded CLUSTER_LOADERS
    else:
        # For .npy-based data: DON'T pre-load all clusters (OOM risk)
        # Instead, each Ray worker loads only its own cluster in client_fn
        metadata_path = os.path.join(args.data_dir, 'metadata.json')
        pkl_path = os.path.join(args.data_dir, 'cluster_loaders.pkl')

        if os.path.exists(metadata_path):
            import json as _json
            with open(metadata_path) as f:
                data_meta = _json.load(f)
            print(f"Data ready: {data_meta['total_records']:,} records, "
                  f"{data_meta['num_classes']} classes, "
                  f"{data_meta['num_clusters']} clusters")
            print("  Workers will load their own cluster data lazily (memory-safe)")
            DATA_DIR = args.data_dir
            CLUSTER_LOADERS = None  # Workers load lazily
        elif os.path.exists(pkl_path):
            print("Loading data from legacy cluster_loaders.pkl...")
            with open(pkl_path, 'rb') as f:
                CLUSTER_LOADERS = pickle.load(f)
            DATA_DIR = None
        else:
            print(f"ERROR: No processed data found in {args.data_dir}")
            print("Run preprocessing first: python -m data.preprocessing")
            sys.exit(1)

    # Setup logger
    results_dir = os.path.join(args.results_dir, scenario, f"seed_{args.seed}")
    logger = ExperimentLogger(results_dir)

    # Create strategy (with round-based checkpoint enabled)
    config = {**FL_CONFIG, **SCENARIO_CONFIG}

    # Pass current round number to clients so they can compute LR decay
    def on_fit_config_fn(server_round: int):
        return {"server_round": server_round}

    strategy = FairPFLStrategy(
        config=config,
        checkpoint_dir=results_dir,
        min_fit_clients=max(1, int(num_clusters * FL_CONFIG['participation_rate'])),
        # BUG-1 FIX: fraction_evaluate=1.0 → ALL clusters evaluated each round.
        # Previously missing: clusters 2,3,6 never appeared in 100 rounds of
        # evaluation because Flower's virtual engine skipped actors that were
        # not selected for fit in that round. fraction_evaluate=1.0 forces
        # ALL available clients to evaluate regardless of fit participation.
        # min_evaluate_clients relaxed to tolerate DP client failures.
        fraction_evaluate=1.0,
        min_evaluate_clients=max(1, num_clusters // 2),
        min_available_clients=num_clusters,
        on_fit_config_fn=on_fit_config_fn,
    )

    # Run Flower simulation
    start_time = time.time()

    # ============================================================
    # RESUME FROM CHECKPOINT (if available)
    # ============================================================
    total_rounds = FL_CONFIG['num_rounds']
    initial_params = None

    ckpt = FairPFLStrategy.load_checkpoint(results_dir)
    if ckpt and 'model_params' in ckpt:
        completed_round = ckpt['round']
        remaining = total_rounds - completed_round
        if remaining <= 0:
            print(f"  [resume] Already completed {completed_round}/{total_rounds} rounds — skipping")
            # Create minimal history object and skip simulation
            class _DummyHistory:
                losses_distributed = []
                metrics_distributed_evaluate = {}
            history = _DummyHistory()
            strategy.restore_state(ckpt)
            elapsed = time.time() - start_time
        else:
            print(f"  [resume] Resuming from round {completed_round} → {remaining} rounds remaining")
            initial_params = ndarrays_to_parameters(ckpt['model_params'])
            strategy.restore_state(ckpt)
            # Override strategy's initial_parameters so Flower uses our checkpoint
            strategy.initial_parameters = initial_params
            total_rounds = remaining
    else:
        print(f"  [resume] No checkpoint found — starting fresh ({total_rounds} rounds)")

    # Determine GPU allocation for simulation
    use_gpu_for_sim = DEVICE.type == 'cuda' if DEVICE else False
    num_cpus_per_client = getattr(args, 'num_cpus', 1)

    if use_gpu_for_sim:
        import os as _os
        total_gpu = 1.0
        target_gpu_concurrent = max(2, getattr(args, 'max_gpu_actors', 4))
        gpu_per_client = round(total_gpu / target_gpu_concurrent, 4)
    else:
        gpu_per_client = 0.0

    print(f"Ray actor resources: num_cpus={num_cpus_per_client}, "
          f"num_gpus={gpu_per_client} "
          f"(max GPU-concurrent: {int(1.0/gpu_per_client) if gpu_per_client > 0 else 'N/A'})")

    # ============================================================
    # Serialize state for Ray workers (Flower 1.13+ / Ray 2.51+)
    # ClientAppActor runs in separate processes that don't inherit
    # Python globals. Write state to a JSON file that workers read.
    # ============================================================
    global _STATE_FILE
    _STATE_FILE = os.path.join(results_dir, '_ray_state.json')
    os.makedirs(results_dir, exist_ok=True)
    # Make a JSON-serializable copy of FL_CONFIG (sets → lists)
    fl_config_json = dict(FL_CONFIG)
    if fl_config_json.get('exclude_classes') is not None:
        fl_config_json['exclude_classes'] = list(fl_config_json['exclude_classes'])
    with open(_STATE_FILE, 'w') as f:
        json.dump({
            'DATA_DIR': DATA_DIR,
            'FL_CONFIG': fl_config_json,
            'SCENARIO_CONFIG': SCENARIO_CONFIG,
            'CLUSTER_DEPTHS': CLUSTER_DEPTHS,
            'PERSONAL_PARAMS_DIR': PERSONAL_PARAMS_DIR,
        }, f)
    # Also set as env var so workers find the file
    os.environ['FAIRPFL_STATE_FILE'] = _STATE_FILE

    if total_rounds > 0:
        history = start_simulation(
            client_fn=client_fn,
            num_clients=num_clusters,
            config=fl.server.ServerConfig(num_rounds=total_rounds),
            strategy=strategy,
            client_resources={"num_cpus": num_cpus_per_client,
                              "num_gpus": gpu_per_client},
            ray_init_args={
                "num_cpus": max(4, num_clusters + 1),
                "object_store_memory": 500 * 1024 ** 2,  # 500MB — minimal
                "ignore_reinit_error": True,
                "include_dashboard": False,
                "runtime_env": {
                    "env_vars": {"FAIRPFL_STATE_FILE": _STATE_FILE},
                },
                # Minimize subprocess overhead (cgroup PID limit workaround)
                "_system_config": {
                    "enable_metrics_collection": False,
                },
                **( {"_temp_dir": os.environ["RAY_TMPDIR"]}
                    if "RAY_TMPDIR" in os.environ else {} ),
            },
        )

    elapsed = time.time() - start_time

    # ============================================================
    # COMPREHENSIVE RESULTS COLLECTION
    # ============================================================
    results = {
        'scenario': scenario,
        'seed': args.seed,
        'rounds': args.rounds,
        'elapsed_seconds': elapsed,
        'config': config,
    }

    # 1. Loss history from Flower
    if hasattr(history, 'losses_distributed'):
        results['losses'] = history.losses_distributed

    # 2. Per-round metrics from strategy (accuracy, F1, DP, q, etc.)
    if hasattr(strategy, 'round_metrics'):
        results['round_metrics'] = strategy.round_metrics

    # 3. Aggregated evaluate metrics from Flower
    if hasattr(history, 'metrics_distributed_evaluate'):
        results['flower_eval_metrics'] = {
            k: v for k, v in history.metrics_distributed_evaluate.items()
        } if isinstance(history.metrics_distributed_evaluate, dict) else {}
    elif hasattr(history, 'metrics_distributed'):
        results['flower_eval_metrics'] = {
            k: v for k, v in history.metrics_distributed.items()
        } if isinstance(history.metrics_distributed, dict) else {}

    # 4. Final test-set evaluation
    # NOTE: Skip for full-dataset mode (DATA_DIR) because:
    # - We can't recover trained weights from Flower simulation easily
    # - Loading 46M test samples into RAM is slow and risky
    # - Real metrics are already in strategy.round_metrics from evaluate()
    if DATA_DIR is not None:
        print("\n[+] Skipping final test-set eval (metrics from strategy.round_metrics)")
        if results.get('round_metrics'):
            last = results['round_metrics'][-1]
            results['final_evaluation'] = {
                'accuracy': last['avg_accuracy'],
                'macro_f1': last['avg_f1'],
                'demographic_parity': last.get('demographic_parity', 0),
                'equalized_odds': last.get('avg_fpr', 0) - last.get('avg_fnr', 0),
            }
            print(f"  Final Accuracy: {last['avg_accuracy']:.4f}")
            print(f"  Final F1-macro: {last['avg_f1']:.4f}")
        else:
            results['final_evaluation'] = {'note': 'No round_metrics available'}
    else:
        print("\n[+] Running final test-set evaluation...")
        try:
            from metrics.evaluation import evaluate_all
            import json as _json

            test_loaders = {}
            if CLUSTER_LOADERS is not None:
                test_loaders = {k: v['test'] for k, v in CLUSTER_LOADERS.items() if 'test' in v}

            sample_X, _ = next(iter(list(test_loaders.values())[0]))
            input_dim = sample_X.shape[-1]
            final_model = create_model(SCENARIO_CONFIG, input_dim=input_dim)
            final_model.eval()

            cluster_predictions = {}
            cluster_labels = {}

            for k, test_loader in test_loaders.items():
                all_preds, all_labels_k = [], []
                with torch.no_grad():
                    for X_batch, y_batch in test_loader:
                        logits = final_model(X_batch)[0]
                        preds = logits.argmax(dim=1)
                        all_preds.extend(preds.numpy())
                        all_labels_k.extend(y_batch.numpy())
                cluster_predictions[k] = np.array(all_preds)
                cluster_labels[k] = np.array(all_labels_k)

            final_eval = evaluate_all(
                cluster_predictions, cluster_labels,
                model=final_model, config=config
            )
            results['final_evaluation'] = final_eval

            print(f"  Final Accuracy: {final_eval['accuracy']:.4f}")
            print(f"  Final F1-macro: {final_eval['macro_f1']:.4f}")
            print(f"  Final DP: {final_eval['demographic_parity']:.4f}")
            print(f"  Final EO: {final_eval['equalized_odds']:.4f}")

        except Exception as e:
            print(f"  Warning: Final evaluation failed: {e}")
            results['final_evaluation'] = {'error': str(e)}

    # 5. Summary statistics (for quick comparison)
    if results.get('round_metrics'):
        last_round = results['round_metrics'][-1]
        all_rounds = results['round_metrics']

        # Convergence analysis: round where F1 stabilizes within 1% for 5 consecutive rounds
        convergence_round = len(all_rounds)  # Default: not converged
        if len(all_rounds) >= 10:
            for i in range(4, len(all_rounds)):
                window = [r['weighted_f1'] for r in all_rounds[i-4:i+1]]
                if max(window) > 0 and (max(window) - min(window)) / max(window) < 0.01:
                    convergence_round = all_rounds[i-4]['round']
                    break

        results['summary'] = {
            # Performance
            'final_avg_accuracy': last_round.get('avg_accuracy', 0),
            'final_weighted_accuracy': last_round.get('weighted_accuracy', 0),
            'final_avg_f1': last_round.get('avg_f1', 0),
            'final_weighted_f1': last_round.get('weighted_f1', 0),
            'final_min_f1': last_round.get('min_f1', 0),
            'final_max_f1': last_round.get('max_f1', 0),
            'final_loss': last_round.get('avg_loss', 0),

            # G7: Precision & Recall
            'final_avg_precision': last_round.get('avg_precision', 0),
            'final_avg_recall': last_round.get('avg_recall', 0),

            # IDS-Specific
            'final_avg_tpr': last_round.get('avg_tpr', 0),
            'final_avg_fpr': last_round.get('avg_fpr', 0),
            'final_avg_fnr': last_round.get('avg_fnr', 0),

            # Fairness
            'final_dp': last_round.get('demographic_parity', 0),
            'final_eo': last_round.get('equalized_odds', 0),
            'final_gini': last_round.get('gini_f1', 0),
            'final_min_cluster_f1': last_round.get('min_f1', 0),
            'final_std_cluster_f1': last_round.get('std_f1', 0),
            'final_accuracy_variance': last_round.get('accuracy_variance', 0),
            'final_accuracy_std': last_round.get('accuracy_std', 0),
            'final_kl_divergence': last_round.get('kl_divergence', 0),
            'final_q': last_round.get('q', 0),
            'final_lambda1': last_round.get('lambda1', 0),
            'final_lambda2': last_round.get('lambda2', 0),

            # G2: Bi-level Fairness (Metric 6, Eq. 27)
            'final_gini_class_worst': last_round.get('gini_class_worst', 0),
            'final_gini_class_avg': last_round.get('gini_class_avg', 0),

            # Privacy
            'final_epsilon_spent': last_round.get('epsilon_spent', 0),

            # Timing
            'avg_round_time_sec': float(np.mean([r['round_time_sec'] for r in all_rounds])),
            'avg_fit_time_sec': float(np.mean([r['fit_time_sec'] for r in all_rounds])),
            'avg_eval_time_sec': float(np.mean([r['eval_time_sec'] for r in all_rounds])),
            'total_wall_clock_sec': last_round.get('wall_clock_sec', 0),

            # Convergence
            'convergence_round': convergence_round,

            # G8: Best round (highest weighted F1)
            'best_round': max(all_rounds, key=lambda r: r.get('weighted_f1', 0)).get('round', 0),
            'best_weighted_f1': max(r.get('weighted_f1', 0) for r in all_rounds),

            # G5: Per-class F1 from last round (named for paper tables)
            'final_per_cluster_per_class_f1': last_round.get('per_cluster_per_class_f1', {}),
            'final_per_cluster_f1': last_round.get('per_cluster_f1', {}),
        }

        # G3 + G4: Model size & communication cost
        # Compute from a fresh model instance (no GPU needed for param counting)
        try:
            _model = create_model(SCENARIO_CONFIG, input_dim=46, num_classes=8)
            shared_params = _model.get_shared_params() if hasattr(_model, 'get_shared_params') \
                            else dict(_model.named_parameters())
            model_size_mb = sum(p.numel() * 4 / 1e6 for p in _model.parameters())
            params_per_round_mb = sum(p.numel() * 4 / 1e6 for p in shared_params.values())
            total_comm_mb = params_per_round_mb * config.get('num_rounds', 200) * 2  # up+down
            results['summary']['model_size_mb'] = float(model_size_mb)
            results['summary']['params_per_round_mb'] = float(params_per_round_mb)
            results['summary']['total_communication_mb'] = float(total_comm_mb)
            results['summary']['total_model_params'] = sum(p.numel() for p in _model.parameters())
            del _model
        except Exception as e:
            results['summary']['model_size_error'] = str(e)

    # Save everything
    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✅ Completed {scenario} in {elapsed:.1f}s")
    print(f"Results saved to: {results_dir}")

    return results


def _create_synthetic_loaders(num_clusters, batch_size, num_features=46,
                               num_classes=8, samples_per_cluster=500):
    """Create synthetic data for pipeline testing."""
    from torch.utils.data import DataLoader, TensorDataset

    loaders = {}
    for k in range(num_clusters):
        X = torch.randn(samples_per_cluster, 1, num_features)
        y = torch.randint(0, num_classes, (samples_per_cluster,))

        n_train = int(0.7 * samples_per_cluster)
        n_val = int(0.15 * samples_per_cluster)

        loaders[k] = {
            'train': DataLoader(
                TensorDataset(X[:n_train], y[:n_train]),
                batch_size=batch_size, shuffle=True),
            'val': DataLoader(
                TensorDataset(X[n_train:n_train + n_val],
                              y[n_train:n_train + n_val]),
                batch_size=batch_size),
            'test': DataLoader(
                TensorDataset(X[n_train + n_val:], y[n_train + n_val:]),
                batch_size=batch_size),
        }

    print(f"Created synthetic loaders: {num_clusters} clusters × "
          f"{samples_per_cluster} samples")
    return loaders


def main():
    parser = argparse.ArgumentParser(description='FairPFL Experiment Runner')

    # Scenario selection
    all_scenarios = list(BASELINE_CONFIGS.keys()) + list(ABLATION_CONFIGS.keys())
    parser.add_argument('--scenario', type=str, default='FairPFL_noDP',
                        help=f'Experiment scenario: {all_scenarios}')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--results_dir', type=str, default='results')

    # Hyperparameter overrides
    parser.add_argument('--q', type=float, default=None,
                        help='Override fairness q parameter')
    parser.add_argument('--epsilon', type=float, default=None,
                        help='Override global DP epsilon')
    parser.add_argument('--dirichlet_alpha', type=float, default=None,
                        help='Override Dirichlet alpha')
    parser.add_argument('--num_clusters', type=int, default=None,
                        help='Override number of clusters')
    parser.add_argument('--num_classes', type=int, default=0,
                        help='Explicit class count for non-CIC datasets '
                             '(0=use default/exclude-minority logic). '
                             'Edge-IIoTset: 15.')
    parser.add_argument('--no-remap', dest='no_remap', action='store_true',
                        help='Disable CIC 34->8 label remapping. Required for '
                             'second datasets (e.g. Edge-IIoTset) whose labels '
                             'are already contiguous 0..C-1.')

    # Performance / resource control
    parser.add_argument('--max_samples', type=int, default=500000,
                        help='Max training samples per cluster per round '
                             '(0=use full dataset). Default: 500000')
    parser.add_argument('--eval_max_samples', type=int, default=100000,
                        help='Max validation samples per cluster for evaluation '
                             '(0=use full val set). Default: 100000 (stratified)')
    parser.add_argument('--eval_every_n_rounds', type=int, default=1,
                        help='Run evaluation every N rounds. Default: 1 (every round). '
                             'A100 recommended: 5 (saves ~80%% eval time)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size (A100 recommended: 2048). '
                             'Default: uses FL_CONFIG value (512)')
    parser.add_argument('--num_cpus', type=int, default=4,
                        help='CPUs per Ray client actor. Lower → more concurrency. '
                             'Default: 4 (allows 8 concurrent with 32 CPUs)')

    # Testing flags
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data for pipeline testing')
    parser.add_argument('--exclude-minority', action='store_true',
                        help='6-class mode: exclude BruteForce and Web-based '
                             'classes (insufficient per-client samples for FL)')

    args = parser.parse_args()
    run_experiment(args)


if __name__ == '__main__':
    main()
