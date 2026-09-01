# fl/client.py — Flower client for FairPFL

import os
import re
import json

import flwr as fl
import torch
import torch.nn.functional as F
import numpy as np
from models.focal_loss import LogitAdjustedCE, FocalLoss, compute_class_weights
from models.dual_personalizer import labels_to_categories


def _sync_dplstm_state_dict(state_dict):
    """Sync DPLSTM standard param names → internal param names.

    Opacus DPLSTM stores parameters under TWO naming conventions:
      - Standard (returned by named_parameters): 'lstm.weight_ih_l0'
      - Internal  (only in state_dict):          'lstm.l0.ih.weight'

    When we load only the 14 standard-named params from the server,
    the 4 internal DPLSTM keys are left with stale values. This
    function copies the updated standard values into the internal keys
    so both representations stay consistent.

    Works recursively on any prefix depth (e.g. 'weight_ih_l0' inside
    a sub-module whose state_dict has no additional prefix).
    """
    pattern = re.compile(r'^(.*?)weight_ih_l(\d+)$')
    for key in list(state_dict.keys()):
        m = pattern.match(key)
        if m:
            prefix = m.group(1)   # e.g. 'lstm.' or ''
            layer  = m.group(2)   # e.g. '0'
            pairs = [
                (f'{prefix}weight_ih_l{layer}', f'{prefix}l{layer}.ih.weight'),
                (f'{prefix}bias_ih_l{layer}',   f'{prefix}l{layer}.ih.bias'),
                (f'{prefix}weight_hh_l{layer}', f'{prefix}l{layer}.hh.weight'),
                (f'{prefix}bias_hh_l{layer}',   f'{prefix}l{layer}.hh.bias'),
            ]
            for std_key, int_key in pairs:
                if std_key in state_dict and int_key in state_dict:
                    state_dict[int_key] = state_dict[std_key]
    return state_dict


class FairPFLClient(fl.client.NumPyClient):
    """
    Flower client for a device-type cluster.
    Supports: FairPFL, baselines (FedAvg, FedProx), and DP scenarios.
    """

    def __init__(self, cluster_id, model, train_loader, val_loader, config,
                 device='cpu', personal_params_path=None):
        self.cluster_id = cluster_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Determine if model has personal/shared split
        has_personal = hasattr(model, 'personal') or hasattr(model, 'device_personal')
        self.has_personalization = has_personal and config.get('personalization', True)

        # Optimizers
        if self.has_personalization:
            personal_params = list(model.get_personal_params().values())
            shared_params = list(model.get_shared_params().values())

            # v18: SGD+momentum for personal head — better generalization
            # on small local data, and momentum state persists across rounds
            self.optimizer_personal = torch.optim.SGD(
                personal_params, lr=config.get('learning_rate_personal', 0.001),
                momentum=0.9, weight_decay=1e-4,
            ) if personal_params else None
            self.optimizer_shared = torch.optim.Adam(
                shared_params, lr=config.get('learning_rate_shared', 0.001)
            ) if shared_params else None
        else:
            self.optimizer_shared = torch.optim.Adam(
                model.parameters(), lr=config.get('learning_rate_shared', 0.001)
            )
            self.optimizer_personal = None

        # LR decay config — round-aware cosine annealing
        # NOTE: Flower simulation recreates clients each round via client_fn,
        # so stateful schedulers (CosineAnnealingLR etc.) reset and never decay.
        # Instead, we compute the target LR from the current round number
        # which the server passes in the fit() config.
        #
        # v18: SGD needs higher LR than Adam (no per-param adaptation).
        # Base 0.01 with cosine decay + momentum=0.9 gives stable convergence.
        self.base_lr_shared = config.get('learning_rate_shared', 0.001)
        self.base_lr_personal = config.get('learning_rate_personal', 0.01)
        self.total_rounds = config.get('num_rounds', 100)
        self.min_lr_shared = 5e-5     # deeper decay for body fine-tuning
        self.min_lr_personal = 5e-4   # SGD head — moderate floor

        # Loss function — FocalLoss with class weights
        # FocalLoss(gamma=1, alpha=4x BF/WB) proven optimal in v6 (F1=0.614)
        # LogitAdjustedCE with balanced class_counts = CE → less stable
        num_classes = config.get('num_classes', 8)
        if 'class_weights' in config:
            class_weights = torch.FloatTensor(config['class_weights']).to(device)
            self.criterion = FocalLoss(alpha=class_weights, num_classes=num_classes)
        else:
            self.criterion = FocalLoss(num_classes=num_classes)

        # Proximal term state
        self.prev_personal_params = None
        self.proximal_mu = config.get('proximal_mu', 0.01)

        # FedProx: save initial global params for proximal term
        # (applies to ALL model params, not just personal)
        self.prev_global_params = None

        # Dual model detection
        self.is_dual = hasattr(model, 'attack_personal')
        self.dual_alpha = 0.3  # weight for category loss

        # DP support
        self.use_dp = config.get('dp', False)
        self.privacy_engine = None
        self._dp_shared_inner = None   # inner module when shared submodel is DP-wrapped
        if self.use_dp:
            self._setup_dp(config)

        # Personal parameter + optimizer state persistence (survives between rounds)
        # v18: Also persist SGD optimizer state (momentum buffer) so the personal
        # head maintains cross-round momentum — equivalent to the implicit momentum
        # that baselines get from server aggregation.
        self.personal_params_path = personal_params_path
        self._personal_optim_path = (
            personal_params_path.replace('.pt', '_optim.pt')
            if personal_params_path else None
        )
        if self.has_personalization and personal_params_path and \
                os.path.exists(personal_params_path):
            try:
                saved = torch.load(personal_params_path,
                                   map_location=self.device,
                                   weights_only=True)
                base = self._base_model
                state = base.state_dict()
                for k, v in saved.items():
                    if k in state:
                        state[k] = v.to(self.device)
                base.load_state_dict(state)
            except Exception as e:
                print(f"[Cluster {self.cluster_id}] Could not load personal "
                      f"params ({e}); using random init")

            # v18: Restore optimizer state (momentum buffers)
            if self.optimizer_personal and self._personal_optim_path and \
                    os.path.exists(self._personal_optim_path):
                try:
                    optim_state = torch.load(self._personal_optim_path,
                                            map_location=self.device,
                                            weights_only=False)
                    self.optimizer_personal.load_state_dict(optim_state)
                except Exception:
                    pass  # first round — no saved state yet

    @property
    def _base_model(self):
        """Return the underlying model, unwrapping Opacus GradSampleModule if present.

        When Opacus wraps the model with make_private(), self.model becomes a
        GradSampleModule.  GradSampleModule in Opacus 1.5+ does NOT delegate
        custom method calls (get_shared_params, etc.) via __getattr__, so we
        must access the original module through ._module explicitly.
        Tensor storage is shared, so parameter updates propagate both ways.
        """
        if hasattr(self.model, '_module'):
            return self.model._module
        return self.model

    def _setup_dp(self, config):
        """Apply DP to the shared submodel.

        Two modes:
        - 'opacus': Full Opacus PrivacyEngine with DPLSTM (requires FairPFLModelDP).
        - 'manual': Standard LSTM model + manual gradient clipping + Gaussian noise.
          This avoids Opacus/DPLSTM convergence issues while providing (ε,δ)-DP.

        Personal parameters never leave the device, so they need no DP.
        """
        self.dp_mode = config.get('dp_mode', 'opacus')

        if self.dp_mode == 'manual':
            # Pre-compute noise sigma: σ = clip_norm * sqrt(2*ln(1.25/δ)) / ε
            import math
            eps = config.get('dp_epsilon_global', 5.0)
            delta = config.get('dp_delta', 1e-5)
            clip = config.get('dp_clip_norm_shared', 1.0)
            self._dp_sigma = clip * math.sqrt(2 * math.log(1.25 / delta)) / eps
            self._dp_clip = clip
            print(f"[Cluster {self.cluster_id}] Manual DP: ε={eps}, δ={delta}, "
                  f"clip={clip}, σ={self._dp_sigma:.4f}")
            return  # No Opacus wrapping needed

        # Opacus mode (requires DPLSTM model)
        try:
            from utils.dp_opacus import make_private_model_and_optimizer
            if self.optimizer_shared is None:
                return

            shared_submodel = getattr(self.model, 'shared', None)

            if shared_submodel is not None:
                # Keep a reference to the original inner module BEFORE wrapping.
                # Opacus wraps in-place (same Python object), so this reference
                # always points to the live parameters even after wrapping.
                self._dp_shared_inner = shared_submodel

                # Rebuild shared optimizer scoped to the shared submodel only
                lr = config.get('learning_rate_shared', 0.001)
                self.optimizer_shared = torch.optim.Adam(
                    shared_submodel.parameters(), lr=lr
                )

                # Wrap only the shared submodel + its optimizer with Opacus
                wrapped_shared, self.optimizer_shared, self.train_loader, \
                    self.privacy_engine = make_private_model_and_optimizer(
                        shared_submodel, self.optimizer_shared,
                        self.train_loader, config)

                # Swap the submodule in the parent model
                self.model.shared = wrapped_shared
                print(f"[Cluster {self.cluster_id}] Opacus PrivacyEngine attached "
                      f"to shared submodel only")
            else:
                # Non-personalized fallback: wrap the entire model
                self.model, self.optimizer_shared, self.train_loader, \
                    self.privacy_engine = make_private_model_and_optimizer(
                        self.model, self.optimizer_shared,
                        self.train_loader, config)
                print(f"[Cluster {self.cluster_id}] Opacus PrivacyEngine attached")
        except Exception as e:
            print(f"[Cluster {self.cluster_id}] Opacus setup failed: [{type(e).__name__}('{e}')]")
            print("  Falling back to manual gradient clipping + noise")
            self.privacy_engine = None

    def get_parameters(self, config):
        """Return ONLY shared parameters to server.

        Uses _base_model to unwrap Opacus GradSampleModule if present,
        so that custom methods like get_shared_params() are accessible.
        """
        base = self._base_model
        if self.has_personalization:
            shared_state = base.get_shared_params()
            return [val.cpu().detach().numpy() for val in shared_state.values()]
        else:
            return [val.cpu().detach().numpy()
                    for val in base.state_dict().values()]

    def set_parameters(self, parameters):
        """Load shared parameters from server.

        Uses KEY-BASED mapping (not positional) so that DPLSTM's extra
        internal state-dict entries (l0.ih.weight, etc.) do not shift
        the parameter assignment and cause size-mismatch errors.
        After loading, _sync_dplstm_state_dict copies the updated
        standard-named weights into the DPLSTM-internal names.
        Uses _base_model to unwrap Opacus GradSampleModule if present.
        """
        base = self._base_model  # unwraps GradSampleModule transparently
        if self.has_personalization:
            # Build {key: tensor} from the same ordering used by get_parameters()
            shared_keys = list(base.get_shared_params().keys())
            param_map = {
                key: torch.tensor(param, device=self.device)
                for key, param in zip(shared_keys, parameters)
            }

            if hasattr(base, 'shared'):
                # FairPFLModel / FairPFLModelDP path.
                # FedPer-style split: shared params span BOTH personal and
                # shared sub-modules (personal LSTM1 + shared LSTM2 + Attention).
                # Keys are full-model-scoped (e.g. 'personal.lstm.weight_ih_l0',
                # 'shared.lstm.weight_ih_l0') so they match base.state_dict().
                #
                # When Opacus wraps the shared submodel, we need to handle DP
                # keys separately. For non-DP, direct state_dict loading works.
                full_state = base.state_dict()
                for key, value in param_map.items():
                    if key in full_state:
                        full_state[key] = value
                # Keep DPLSTM internal keys in sync with the standard names
                _sync_dplstm_state_dict(full_state)
                base.load_state_dict(full_state, strict=False)

            elif hasattr(base, 'shared_lstm'):
                # DualFairPFLModel / DualFairPFLModelDP path
                for name in ['shared_lstm', 'shared_attention', 'shared_norm']:
                    if hasattr(base, name):
                        sub = getattr(base, name)
                        sub_state = sub.state_dict()
                        prefix = f'{name}.'
                        for key in list(sub_state.keys()):
                            full_key = f'{prefix}{key}'
                            if full_key in param_map:
                                sub_state[key] = param_map[full_key]
                        # Sync DPLSTM internals (no-op for non-DPLSTM modules)
                        _sync_dplstm_state_dict(sub_state)
                        sub.load_state_dict(sub_state, strict=False)
        else:
            state = base.state_dict()
            for (key, _), param in zip(state.items(), parameters):
                state[key] = torch.tensor(param, device=self.device)
            base.load_state_dict(state)

    def fit(self, parameters, config):
        """Local training with optional proximal regularization."""
        self.set_parameters(parameters)

        # === Round-aware cosine LR decay ===
        # Recover current round from the Flower server config
        import math
        current_round = config.get('server_round', 1) if config else 1
        progress = current_round / max(self.total_rounds, 1)
        cos_factor = (1 + math.cos(math.pi * progress)) / 2  # 1.0 → 0.0
        lr_shared = self.min_lr_shared + (self.base_lr_shared - self.min_lr_shared) * cos_factor
        lr_personal = self.min_lr_personal + (self.base_lr_personal - self.min_lr_personal) * cos_factor
        for pg in self.optimizer_shared.param_groups:
            pg['lr'] = lr_shared
        if self.optimizer_personal:
            for pg in self.optimizer_personal.param_groups:
                pg['lr'] = lr_personal

        # Save personal params for proximal term
        if self.has_personalization and self.optimizer_personal:
            if self.prev_personal_params is None:
                self.prev_personal_params = {
                    k: v.clone().detach()
                    for k, v in self._base_model.get_personal_params().items()
                }

        # FedProx: save global params snapshot for proximal term
        # This applies to ALL model params (not just personal)
        if self.proximal_mu > 0 and self.prev_global_params is None:
            self.prev_global_params = {
                k: v.clone().detach()
                for k, v in self._base_model.state_dict().items()
            }

        self.model.train()
        total_loss = 0
        num_batches = 0

        for epoch in range(self.config.get('local_epochs', 5)):
            for X_batch, y_batch in self.train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Forward — dual models return 3-tuple, others return 2-tuple
                output = self.model(X_batch)
                if self.is_dual and len(output) == 3:
                    logits, _, cat_logits = output
                    cat_labels = labels_to_categories(y_batch)
                    loss_fine = self.criterion(logits, y_batch)
                    loss_cat = F.cross_entropy(cat_logits, cat_labels)
                    loss = (1 - self.dual_alpha) * loss_fine + \
                           self.dual_alpha * loss_cat
                else:
                    logits, _ = output[:2]
                    loss = self.criterion(logits, y_batch)

                # Proximal term for personalized layers (personal drift penalty)
                # Cold-start fix: disable for the first N rounds so that personal
                # layers can freely adapt to the incoming shared weights.
                # Without warmup, proximal anchors personal layers to their RANDOM
                # initialization, actively harming convergence for 20+ rounds.
                warmup_rounds = self.config.get('personal_warmup_rounds', 20)
                past_warmup = current_round > warmup_rounds
                if (self.has_personalization and
                        self.prev_personal_params and self.proximal_mu > 0
                        and past_warmup):
                    proximal_loss = 0
                    for key, param in self._base_model.get_personal_params().items():
                        if key in self.prev_personal_params:
                            proximal_loss += torch.norm(
                                param - self.prev_personal_params[key]) ** 2
                    loss = loss + (self.proximal_mu / 2) * proximal_loss

                # FedProx global proximal term: μ/2 * ‖w - w_global‖²
                # Applies to ALL model weights vs server-received global model
                if (self.proximal_mu > 0 and self.prev_global_params
                        and not self.has_personalization):
                    fedprox_loss = 0
                    for key, param in self._base_model.named_parameters():
                        if key in self.prev_global_params:
                            fedprox_loss += torch.norm(
                                param - self.prev_global_params[key].to(
                                    self.device)) ** 2
                    loss = loss + (self.proximal_mu / 2) * fedprox_loss

                # pFedMe Moreau envelope regularization
                moreau_beta = self.config.get('moreau_beta', 0)
                if (moreau_beta > 0 and self.has_personalization and
                        self.prev_personal_params):
                    moreau_loss = 0
                    for key, param in self._base_model.get_personal_params().items():
                        if key in self.prev_personal_params:
                            moreau_loss += torch.norm(
                                param - self.prev_personal_params[key]) ** 2
                    loss = loss + (moreau_beta / 2) * moreau_loss

                # Backward
                if self.optimizer_personal:
                    self.optimizer_personal.zero_grad()
                self.optimizer_shared.zero_grad()
                loss.backward()

                # Gradient clipping — DP uses tight norm, non-DP is generous
                if self.use_dp:
                    clip_shared = self.config.get('dp_clip_norm_shared', 1.0)
                    clip_personal = self.config.get('dp_clip_norm_personal', 1.0)
                else:
                    clip_shared = self.config.get('grad_clip_norm', 10.0)
                    clip_personal = self.config.get('grad_clip_norm', 10.0)

                if self.has_personalization:
                    shared_params = list(self._base_model.get_shared_params().values())
                    if shared_params:
                        torch.nn.utils.clip_grad_norm_(shared_params, clip_shared)
                    if self.optimizer_personal:
                        personal_params = list(
                            self._base_model.get_personal_params().values())
                        if personal_params:
                            torch.nn.utils.clip_grad_norm_(
                                personal_params, clip_personal)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), clip_shared)

                # Manual DP: inject Gaussian noise into shared gradients
                # σ was pre-computed in _setup_dp()
                if self.use_dp and getattr(self, 'dp_mode', 'opacus') == 'manual':
                    sigma = getattr(self, '_dp_sigma', 0)
                    if sigma > 0:
                        if self.has_personalization:
                            # Only noise shared params (personal head stays clean)
                            noise_params = list(
                                self._base_model.get_shared_params().values())
                        else:
                            # Non-personalized model: noise ALL params
                            noise_params = list(self.model.parameters())
                        for p in noise_params:
                            if p.grad is not None:
                                noise = torch.randn_like(p.grad) * sigma
                                p.grad.add_(noise)

                # Step
                self.optimizer_shared.step()
                if self.optimizer_personal:
                    self.optimizer_personal.step()

                total_loss += loss.item()
                num_batches += 1

        # Update personal param snapshot
        if self.has_personalization:
            self.prev_personal_params = {
                k: v.clone().detach()
                for k, v in self._base_model.get_personal_params().items()
            }

            # Persist personal params + optimizer state to disk
            if self.personal_params_path:
                try:
                    os.makedirs(os.path.dirname(self.personal_params_path),
                                exist_ok=True)
                    torch.save(
                        {k: v.cpu() for k, v in self.prev_personal_params.items()},
                        self.personal_params_path,
                    )
                    # v18: Save optimizer state (SGD momentum buffers)
                    if self.optimizer_personal and self._personal_optim_path:
                        torch.save(
                            self.optimizer_personal.state_dict(),
                            self._personal_optim_path,
                        )
                except Exception as e:
                    print(f"[Cluster {self.cluster_id}] Could not save personal "
                          f"params: {e}")

        avg_loss = total_loss / max(num_batches, 1)
        return (
            self.get_parameters(config={}),
            len(self.train_loader.dataset),
            {"cluster_id": self.cluster_id, "train_loss": float(avg_loss)}
        )

    def evaluate(self, parameters, config):
        """Evaluate on validation set, return per-cluster metrics including IDS-specific."""
        return self._evaluate_impl(parameters, config)

    def _evaluate_impl(self, parameters, config):
        """Internal evaluate implementation."""
        self.set_parameters(parameters)
        self.model.eval()

        all_preds, all_labels = [], []
        total_loss = 0

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                logits = self.model(X_batch)[0]  # first element is always fine_logits
                loss = self.criterion(logits, y_batch)
                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        import numpy as np
        from sklearn.metrics import (f1_score, accuracy_score, confusion_matrix,
                                     precision_score, recall_score)
        from models.dual_personalizer import BENIGN_CLASS_ID

        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted',
                                    zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted',
                              zero_division=0)

        # Per-class F1 for Metric 6 (intra-client class Gini, Eq. 27)
        per_class_f1 = f1_score(all_labels, all_preds, average=None,
                                zero_division=0).tolist()

        # IDS-specific: binary benign vs attack detection metrics
        y_true_bin = (np.array(all_labels) != BENIGN_CLASS_ID).astype(int)
        y_pred_bin = (np.array(all_preds) != BENIGN_CLASS_ID).astype(int)
        cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        tpr = float(tp / (tp + fn + 1e-10))  # Detection Rate
        fpr = float(fp / (fp + tn + 1e-10))  # False Positive Rate
        fnr = float(fn / (fn + tp + 1e-10))  # False Negative Rate

        return (
            float(total_loss / max(len(self.val_loader), 1)),
            len(self.val_loader.dataset),
            {
                "cluster_id": float(self.cluster_id),
                "f1_macro": float(f1),
                "accuracy": float(acc),
                "precision_weighted": float(precision),
                "recall_weighted": float(recall),
                "tpr": tpr,
                "fpr": fpr,
                "fnr": fnr,
                # Flower metrics dict only accepts Scalar types (int, float,
                # str, bool, bytes). Serialize list as JSON string.
                "per_class_f1": json.dumps(per_class_f1),
            }
        )
