# fl/strategy.py — Fairness-aware aggregation strategy with full metrics tracking

import time
import os
import json
import pickle
import flwr as fl
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from flwr.common import (
    FitRes, Parameters, Scalar,
    parameters_to_ndarrays, ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy


def weighted_average(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """Aggregate evaluation metrics from all clients (weighted by num_examples)."""
    if not metrics:
        return {}

    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}

    aggregated = {}

    # Weighted average for scalar metrics
    for key in ['accuracy', 'f1_macro', 'loss', 'tpr', 'fpr', 'fnr']:
        vals = [(n, m.get(key, 0.0)) for n, m in metrics if key in m]
        if vals:
            aggregated[key] = sum(n * v for n, v in vals) / total

    return aggregated


class FairPFLStrategy(fl.server.strategy.FedAvg):
    """
    Custom aggregation strategy with fairness-aware weights.
    α_k = (|D_k| · φ_k) / Σ_j(|D_j| · φ_j)
    where φ_k = (max_j F1_j / F1_k)^q

    Includes:
    - Self-correcting q parameter
    - Lagrange multipliers for DP/EO constraints
    - Per-cluster F1 tracking
    - Comprehensive per-round metrics collection (literature-backed)
    """

    def __init__(self, config, checkpoint_dir=None, **kwargs):
        # Add metrics aggregation function
        kwargs['evaluate_metrics_aggregation_fn'] = weighted_average
        super().__init__(**kwargs)
        self.config = config
        self.q = config.get('fairness_q_init', 1.0)
        num_clusters = config.get('num_clusters', 9)
        self.cluster_f1_history = {k: [0.5] for k in range(num_clusters)}
        self.lambda1 = 0.0  # Lagrange multiplier for DP constraint
        self.lambda2 = 0.0  # Lagrange multiplier for EO constraint
        self.rho = 0.1      # Penalty parameter
        self.round = 0

        # === Comprehensive metrics storage ===
        self.round_metrics: list = []  # List of dicts, one per round
        self._round_start_time: Optional[float] = None
        self._fit_start_time: Optional[float] = None
        self._fit_end_time: Optional[float] = None
        self._experiment_start_time: float = time.time()

        # === Round-based checkpoint (supports resume) ===
        self._checkpoint_dir = checkpoint_dir
        self._last_aggregated_params = None  # track latest global model
        self._round_offset = 0  # for resume: maps server_round to real round
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

    def configure_evaluate(self, server_round, parameters, client_manager):
        """Evaluate on ALL available clients every eval round.

        BUG-1 ROOT FIX: super().configure_evaluate() (FedAvg) only picks clients
        that participated in the current fit round. With participation_rate=0.67,
        3/9 clusters are permanently excluded from evaluation.
        This override explicitly samples ALL available clients for evaluation.
        """
        every_n = self.config.get('eval_every_n_rounds', 1)
        num_rounds = self.config.get('num_rounds', 200)
        is_eval_round = (server_round % every_n == 0) or (server_round == num_rounds)
        if not is_eval_round:
            return []

        # Sample ALL available clients (not just fit participants)
        import flwr as fl
        all_clients = client_manager.all()
        if not all_clients:
            return []

        eval_config = {}
        if self.on_evaluate_config_fn is not None:
            eval_config = self.on_evaluate_config_fn(server_round)

        return [
            (client, fl.common.EvaluateIns(parameters, eval_config))
            for client in all_clients.values()
        ]


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Fairness-aware weighted aggregation with timing."""
        self.round = server_round
        self._round_start_time = time.time()

        if not results:
            return None, {}

        # Extract cluster info
        cluster_f1 = {}
        for _, fit_res in results:
            cid = fit_res.metrics.get("cluster_id", 0)
            cluster_f1[cid] = self.cluster_f1_history.get(cid, [0.5])[-1]

        # Compute fairness-aware weights
        weights = self._compute_fair_weights(results, cluster_f1)

        # Weighted aggregation
        params_list = [
            parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]

        # Aggregate each layer
        aggregated = []
        for i in range(len(params_list[0])):
            layer_avg = sum(
                w * params[i] for w, params in zip(weights, params_list)
            )
            aggregated.append(layer_avg)

        # Check fairness constraints and update q is now done in
        # aggregate_evaluate() where real current-round F1 values are available.

        self._fit_end_time = time.time()

        agg_params = ndarrays_to_parameters(aggregated)
        self._last_aggregated_params = agg_params  # keep reference for checkpoint

        return agg_params, {
            "round": server_round,
            "q": float(self.q),
        }

    def _compute_fair_weights(self, results, cluster_f1):
        """
        Compute fairness-aware weights:
        α_k = (|D_k| · φ_k) / Σ_j(|D_j| · φ_j)
        where φ_k = min((max_j F1_j / F1_k)^q, φ_max)

        When config['fairness_phi_max'] is None (default), no clipping is
        applied and behavior is identical to the original formulation.
        """
        if not cluster_f1 or self.q == 0:
            # Equal data-proportional weights (standard FedAvg)
            total_samples = sum(fit_res.num_examples for _, fit_res in results)
            return [fit_res.num_examples / total_samples
                    for _, fit_res in results]

        max_f1 = max(cluster_f1.values()) if cluster_f1 else 1.0
        # φ_max: upper bound on fairness boost factor (None = no clipping)
        phi_max = self.config.get('fairness_phi_max', None)

        weights = []
        for _, fit_res in results:
            cid = fit_res.metrics.get("cluster_id", 0)
            n_samples = fit_res.num_examples
            f1_k = cluster_f1.get(cid, max_f1)
            # Fairness correction: boost underperforming clusters
            phi_k = (max_f1 / max(f1_k, 1e-6)) ** self.q
            # B1: Optional clipping to prevent extreme weight concentration
            if phi_max is not None:
                phi_k = min(phi_k, phi_max)
            weights.append(n_samples * phi_k)

        total = sum(weights)
        return [w / total for w in weights] if total > 0 else \
               [1.0 / len(results)] * len(results)

    def _update_fairness_params(self, cluster_f1):
        """Check constraints and update Lagrange multipliers + q.

        Implements the dead-zone controller from Eq.(7) of the paper:
          - If F1-gap > delta1 (fairness too poor):  increase q (more pressure)
          - If F1-gap < delta2 (fairness already good): decrease q (relax pressure)
          - Otherwise (in dead-zone): keep q unchanged (prevent oscillation)

        Skipped when config['adaptive_q'] = False (e.g., qFFL baseline uses fixed q).
        """
        # For baselines with fixed q (qFFL, q-sweep), skip adaptive update
        if not self.config.get('adaptive_q', True):
            return

        if len(cluster_f1) < 2:
            return

        f1_values = list(cluster_f1.values())
        # min-max F1 gap (performance-fairness proxy)
        # Note: NOT standard demographic parity (P(Y^=1|group) difference)
        f1_gap = max(f1_values) - min(f1_values)

        delta1 = self.config.get('fairness_delta1', 0.05)   # upper threshold
        delta2 = self.config.get('fairness_delta2', 0.02)   # lower threshold
        q_delta = self.config.get('fairness_q_delta', 0.1)
        q_max   = self.config.get('fairness_q_max', 3.0)    # v12c: 2.0 tested → hurt perf, reverted to 3.0

        if f1_gap > delta1:
            # Fairness gap too large — increase pressure
            self.lambda1 += self.rho * (f1_gap - delta1)
            old_q = self.q
            self.q = min(self.q + q_delta, q_max)            # BUG-2 FIX: enforce cap
            if old_q < q_max and self.q >= q_max:
                print(f"  [Fairness] q reached cap q_max={q_max:.1f} "
                      f"(f1_gap={f1_gap:.4f})")
        elif f1_gap < delta2 and self.q > 0:
            # Fairness gap already small — relax pressure (anti-overfit)
            self.q = max(self.q - q_delta, 0.0)
            print(f"  [Fairness] q relaxed to {self.q:.2f} "
                  f"(f1_gap={f1_gap:.4f} < delta2={delta2})",
                  flush=True)


    @staticmethod
    def _compute_kl_divergence(values):
        """KL divergence of distribution vs uniform (qFFL metric)."""
        if not values or len(values) < 2:
            return 0.0
        arr = np.array(values, dtype=float)
        total = arr.sum()
        if total <= 0:
            return 0.0
        p = arr / total  # Normalize to probability distribution
        q = np.ones_like(p) / len(p)  # Uniform distribution
        # KL(P || Q) — clip to avoid log(0)
        p_clipped = np.clip(p, 1e-10, None)
        q_clipped = np.clip(q, 1e-10, None)
        return float(np.sum(p_clipped * np.log(p_clipped / q_clipped)))

    @staticmethod
    def _compute_gini(values):
        """Compute Gini coefficient of a list of values."""
        sorted_vals = np.sort(np.array(values, dtype=float))
        n = len(sorted_vals)
        if n == 0 or np.sum(sorted_vals) == 0:
            return 0.0
        index = np.arange(1, n + 1)
        return float(
            (2 * np.sum(index * sorted_vals) - (n + 1) * np.sum(sorted_vals)) /
            (n * np.sum(sorted_vals) + 1e-10)
        )

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures,
    ):
        """Collect comprehensive per-round metrics for paper."""
        eval_end_time = time.time()

        # --- Per-cluster extraction ---
        cluster_accuracy = {}
        cluster_f1 = {}
        cluster_loss = {}
        cluster_samples = {}
        cluster_tpr = {}
        cluster_fpr = {}
        cluster_fnr = {}
        cluster_per_class_f1 = {}
        cluster_precision = {}
        cluster_recall = {}

        for _, eval_res in results:
            cid = int(eval_res.metrics.get("cluster_id", 0))
            f1 = eval_res.metrics.get("f1_macro", 0.0)
            acc = eval_res.metrics.get("accuracy", 0.0)

            cluster_f1[cid] = f1
            cluster_accuracy[cid] = acc
            cluster_loss[cid] = eval_res.loss
            cluster_samples[cid] = eval_res.num_examples
            cluster_tpr[cid] = eval_res.metrics.get("tpr", 0.0)
            cluster_fpr[cid] = eval_res.metrics.get("fpr", 0.0)
            cluster_fnr[cid] = eval_res.metrics.get("fnr", 0.0)

            # Per-class F1 for Metric 6 (intra-client class Gini, Eq. 27)
            # per_class_f1 is serialized as JSON string by the client
            # (Flower metrics only accept Scalar types, not lists)
            pcf1_raw = eval_res.metrics.get("per_class_f1", None)
            if pcf1_raw is not None:
                if isinstance(pcf1_raw, str):
                    try:
                        pcf1 = json.loads(pcf1_raw)
                    except (json.JSONDecodeError, TypeError):
                        pcf1 = None
                elif isinstance(pcf1_raw, (list, tuple)):
                    pcf1 = pcf1_raw
                else:
                    pcf1 = None
                if pcf1 is not None:
                    cluster_per_class_f1[cid] = pcf1
            cluster_precision[cid] = eval_res.metrics.get("precision_weighted", 0.0)
            cluster_recall[cid] = eval_res.metrics.get("recall_weighted", 0.0)

            # Update history for next round's weight computation
            if cid in self.cluster_f1_history:
                self.cluster_f1_history[cid].append(f1)
            else:
                self.cluster_f1_history[cid] = [f1]

        # --- Aggregate round-level metrics ---
        f1_vals = list(cluster_f1.values()) if cluster_f1 else [0.0]
        acc_vals = list(cluster_accuracy.values()) if cluster_accuracy else [0.0]
        loss_vals = list(cluster_loss.values()) if cluster_loss else [0.0]
        tpr_vals = list(cluster_tpr.values()) if cluster_tpr else [0.0]
        fpr_vals = list(cluster_fpr.values()) if cluster_fpr else [0.0]
        fnr_vals = list(cluster_fnr.values()) if cluster_fnr else [0.0]

        # Weighted averages (by sample count)
        total_samples = sum(cluster_samples.values()) if cluster_samples else 1
        weighted_f1 = sum(cluster_f1.get(k, 0) * cluster_samples.get(k, 0)
                          for k in cluster_f1) / total_samples
        weighted_acc = sum(cluster_accuracy.get(k, 0) * cluster_samples.get(k, 0)
                          for k in cluster_accuracy) / total_samples

        # Fairness metrics
        # BUG-5 FIX: renamed 'dp' → 'f1_gap' (min-max F1 gap ≠ standard Demographic Parity)
        # Standard DP = max|P(Y^=1|group=a) - P(Y^=1|group=b)| (prediction probability)
        # min-max F1 gap is a valid fairness proxy but must be labeled correctly in paper
        f1_gap = max(f1_vals) - min(f1_vals) if len(f1_vals) >= 2 else 0.0

        f1_std = float(np.std(f1_vals)) if len(f1_vals) >= 2 else 0.0
        acc_variance = float(np.var(acc_vals)) if len(acc_vals) >= 2 else 0.0
        acc_std = float(np.std(acc_vals)) if len(acc_vals) >= 2 else 0.0
        gini = self._compute_gini(f1_vals)
        kl_div = self._compute_kl_divergence(acc_vals)

        # Metric 6: Intra-Client Class Fairness (Eq. 27)
        # Worst-cluster per-class Gini — measures class-level detection equity
        gini_class_per_cluster = {}
        for cid, pcf1 in cluster_per_class_f1.items():
            if len(pcf1) >= 2:
                gini_class_per_cluster[cid] = self._compute_gini(pcf1)
        gini_class_worst = max(gini_class_per_cluster.values()) \
            if gini_class_per_cluster else 0.0
        gini_class_avg = float(np.mean(list(gini_class_per_cluster.values()))) \
            if gini_class_per_cluster else 0.0

        # Equalized Odds: max pairwise (|FPR_i - FPR_j| + |FNR_i - FNR_j|)
        eo = 0.0
        for i in range(len(fpr_vals)):
            for j in range(i + 1, len(fpr_vals)):
                eo_ij = abs(fpr_vals[i] - fpr_vals[j]) + abs(fnr_vals[i] - fnr_vals[j])
                eo = max(eo, eo_ij)

        # Privacy spent (epsilon) — collected from client metrics if DP is enabled
        eps_vals = []
        for _, eval_res in results:
            eps = eval_res.metrics.get('epsilon_spent', None)
            if eps is not None and eps > 0:
                eps_vals.append(float(eps))
        epsilon_spent = float(np.mean(eps_vals)) if eps_vals else 0.0

        # Timing
        round_time = eval_end_time - self._round_start_time \
            if self._round_start_time else 0.0
        fit_time = self._fit_end_time - self._round_start_time \
            if (self._fit_end_time and self._round_start_time) else 0.0
        eval_time = round_time - fit_time
        wall_clock = eval_end_time - self._experiment_start_time

        # Store round metrics (use real round number accounting for resume offset)
        real_round = server_round + self._round_offset
        round_data = {
            'round': real_round,

            # === Performance (aggregated) ===
            'avg_accuracy': float(np.mean(acc_vals)),
            'weighted_accuracy': float(weighted_acc),
            'avg_f1': float(np.mean(f1_vals)),
            'weighted_f1': float(weighted_f1),
            'min_f1': float(min(f1_vals)),
            'max_f1': float(max(f1_vals)),
            'std_f1': f1_std,
            'avg_loss': float(np.mean(loss_vals)),

            # === IDS-Specific (aggregated) ===
            'avg_tpr': float(np.mean(tpr_vals)),  # Detection Rate
            'avg_fpr': float(np.mean(fpr_vals)),  # False Positive Rate
            'avg_fnr': float(np.mean(fnr_vals)),  # False Negative Rate

            # === Fairness ===
            # 'f1_gap' = min-max F1 gap (NOT standard demographic parity)
            'f1_gap': float(f1_gap),
            'demographic_parity': float(f1_gap),  # backward-compat alias
            'equalized_odds': float(eo),
            'gini_f1': float(gini),
            'accuracy_variance': acc_variance,  # qFFL metric
            'accuracy_std': acc_std,
            'kl_divergence': float(kl_div),  # qFFL metric
            'gini_class_worst': float(gini_class_worst),  # Metric 6 (Eq. 27)
            'gini_class_avg': float(gini_class_avg),      # avg intra-client class Gini
            'q': float(self.q),
            'lambda1': float(self.lambda1),
            'lambda2': float(self.lambda2),

            # === Privacy ===
            'epsilon_spent': epsilon_spent,

            # === Timing ===
            'round_time_sec': float(round_time),
            'fit_time_sec': float(fit_time),
            'eval_time_sec': float(eval_time),
            'wall_clock_sec': float(wall_clock),

            # === Per-cluster detail ===
            'per_cluster_f1': {str(k): float(v) for k, v in cluster_f1.items()},
            'per_cluster_accuracy': {str(k): float(v) for k, v in cluster_accuracy.items()},
            'per_cluster_loss': {str(k): float(v) for k, v in cluster_loss.items()},
            'per_cluster_tpr': {str(k): float(v) for k, v in cluster_tpr.items()},
            'per_cluster_fpr': {str(k): float(v) for k, v in cluster_fpr.items()},
            'per_cluster_fnr': {str(k): float(v) for k, v in cluster_fnr.items()},
            'per_cluster_per_class_f1': {str(k): v for k, v in cluster_per_class_f1.items()},
            'per_cluster_precision': {str(k): float(v) for k, v in cluster_precision.items()},
            'per_cluster_recall': {str(k): float(v) for k, v in cluster_recall.items()},

            # === Aggregate precision/recall ===
            'avg_precision': float(np.mean(list(cluster_precision.values()))) if cluster_precision else 0.0,
            'avg_recall': float(np.mean(list(cluster_recall.values()))) if cluster_recall else 0.0,
        }
        self.round_metrics.append(round_data)

        # Print summary every 10 rounds or first 5
        if real_round % 10 == 0 or server_round <= 5:
            print(f"[R{real_round:3d}] acc={weighted_acc:.4f} "
                  f"F1={weighted_f1:.4f} (min={min(f1_vals):.4f}) "
                  f"TPR={np.mean(tpr_vals):.4f} FPR={np.mean(fpr_vals):.4f} "
                  f"F1gap={f1_gap:.4f} q={self.q:.2f} "
                  f"t={round_time:.1f}s")


        # === Round-based checkpoint: save model + state after every eval ===
        if self._checkpoint_dir:
            real_round = server_round + self._round_offset
            self._save_checkpoint(real_round)

        # === Update q with this round's real F1 values ===
        # NOTE: Must happen AFTER cluster_f1_history is updated above,
        # and here (not in aggregate_fit) because aggregate_fit only has
        # stale F1 values from the *previous* round.
        # BUG-19 FIX: Only adapt q for fairness-aware scenarios (q_init > 0).
        # FedAvg/FedProx use q_init=0 → q must stay at 0 (uniform weights).
        if cluster_f1 and self.config.get('fairness_q_init', 0) > 0:
            self._update_fairness_params(cluster_f1)

        return super().aggregate_evaluate(server_round, results, failures)

    # =================================================================
    # CHECKPOINT / RESUME
    # =================================================================
    def _save_checkpoint(self, real_round: int):
        """Save full resumable checkpoint: model weights + strategy state."""
        try:
            # 1. Round metrics (human-readable)
            metrics_path = os.path.join(self._checkpoint_dir, 'round_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(self.round_metrics, f, default=str)

            # 2. Model weights + strategy state (binary, for resume)
            ckpt = {
                'round': real_round,
                'q': self.q,
                'lambda1': self.lambda1,
                'lambda2': self.lambda2,
                'cluster_f1_history': self.cluster_f1_history,
                'round_metrics': self.round_metrics,
            }
            if self._last_aggregated_params is not None:
                ckpt['model_params'] = parameters_to_ndarrays(
                    self._last_aggregated_params
                )

            ckpt_path = os.path.join(self._checkpoint_dir, 'checkpoint.pkl')
            tmp_path = ckpt_path + '.tmp'
            with open(tmp_path, 'wb') as f:
                pickle.dump(ckpt, f)
            os.replace(tmp_path, ckpt_path)  # atomic write

        except Exception as e:
            print(f"  [checkpoint] Warning: save failed: {e}")

    @staticmethod
    def load_checkpoint(checkpoint_dir: str):
        """Load checkpoint for resume. Returns (params_ndarrays, state_dict, completed_round) or None."""
        ckpt_path = os.path.join(checkpoint_dir, 'checkpoint.pkl')
        if not os.path.exists(ckpt_path):
            return None
        try:
            with open(ckpt_path, 'rb') as f:
                ckpt = pickle.load(f)
            print(f"  [resume] Loaded checkpoint from round {ckpt['round']}")
            return ckpt
        except Exception as e:
            print(f"  [resume] Warning: failed to load checkpoint: {e}")
            return None

    def restore_state(self, ckpt: dict):
        """Restore strategy state from checkpoint (q, lambdas, history, metrics)."""
        self.q = ckpt.get('q', self.q)
        self.lambda1 = ckpt.get('lambda1', self.lambda1)
        self.lambda2 = ckpt.get('lambda2', self.lambda2)
        self.cluster_f1_history = ckpt.get('cluster_f1_history', self.cluster_f1_history)
        self.round_metrics = ckpt.get('round_metrics', [])
        self._round_offset = ckpt.get('round', 0)
        self.round = self._round_offset
        print(f"  [resume] Restored state: q={self.q:.2f}, round_offset={self._round_offset}, "
              f"{len(self.round_metrics)} metrics entries")
