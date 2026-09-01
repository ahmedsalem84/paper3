# fl/baselines.py — Baseline configuration for all comparison methods

BASELINE_CONFIGS = {
    'FedAvg': {
        'model_type': 'baseline',
        'fairness_q_init': 0,
        'proximal_mu': 0,
        'dp': False,
        'personalization': False,
    },
    'FedAvg_DP': {
        'model_type': 'baseline',
        'fairness_q_init': 0,
        'proximal_mu': 0,
        'dp': True,
        'dp_mode': 'manual',        # Manual DP: gradient clip + Gaussian noise
        'personalization': False,
        'dp_epsilon_global': 5.0,
        'dp_clip_norm_shared': 1.0,
    },
    'FedProx': {
        'model_type': 'baseline',
        'fairness_q_init': 0,
        'proximal_mu': 0.01,
        'dp': False,
        'personalization': False,
    },
    'qFFL': {
        'model_type': 'baseline',
        'fairness_q_init': 1.0,
        'proximal_mu': 0,
        'dp': False,
        'personalization': False,
        'objective': 'alpha_fairness',
        'adaptive_q': False,    # Fixed q=1.0 matching Li et al. (2020); no adaptive controller
    },
    'Ditto': {
        'model_type': 'fairpfl',
        'fairness_q_init': 0,
        'proximal_mu': 0.01,
        'dp': False,
        'personalization': True,
        'ditto_lambda': 0.1,
    },
    'pFedMe': {
        'model_type': 'fairpfl',
        'fairness_q_init': 0,
        'proximal_mu': 0,
        'dp': False,
        'personalization': True,
        'moreau_beta': 1.0,
    },
    'FairPFL_noDP': {
        'model_type': 'fairpfl',
        'fairness_q_init': 1.0,
        'proximal_mu': 0.001,       # Reduced from 0.01 — softer personal drift penalty
        'personal_warmup_rounds': 20,  # First 20 rounds: no personal proximal (cold-start fix)
        'dp': False,
        'personalization': True,
    },
    'FairPFL_DP': {
        'model_type': 'fairpfl',
        'fairness_q_init': 1.0,
        'proximal_mu': 0.001,       # Aligned with methodology (μ=0.001) and FairPFL_noDP
        'dp': True,
        'dp_mode': 'manual',        # Manual DP: gradient clip + Gaussian noise (no Opacus/DPLSTM)
        'personalization': True,
        'personal_warmup_rounds': 20,  # Match FairPFL_noDP
        'dp_epsilon_global': 5.0,  # shared params only
        # Manual DP uses same LR as noDP — noise is injected AFTER gradient step
        'learning_rate_shared': 0.001,
        'learning_rate_personal': 0.01,  # Personal head has NO DP → same as noDP
        'dp_clip_norm_shared': 1.0,      # Per-sample gradient clip (standard for DP-SGD)
    },
}

# Ablation scenario configs — v18 aligned
# Each removes ONE component from FairPFL_noDP to measure its contribution.
# FairPFL_noDP (full): fairpfl, q=1.0, μ=0.001, warmup=20, personalization=True
ABLATION_CONFIGS = {
    # A1: No fairness (q=0) — removes adaptive q-FFL reweighting
    'A1_no_fairness': {
        'model_type': 'fairpfl',
        'fairness_q_init': 0,
        'proximal_mu': 0.001,
        'personal_warmup_rounds': 20,
        'dp': False,
        'personalization': True,
    },
    # A2: No personalization — shared model only (like FedAvg but with q)
    'A2_no_personal': {
        'model_type': 'baseline',
        'fairness_q_init': 1.0,
        'proximal_mu': 0,
        'dp': False,
        'personalization': False,
    },
    # A3: No adaptive depth — fixed depth=1 for all clusters
    'A3_fixed_depth': {
        'model_type': 'fairpfl',
        'fairness_q_init': 1.0,
        'proximal_mu': 0.001,
        'personal_warmup_rounds': 20,
        'dp': False,
        'personalization': True,
        'adaptive_depth': False,  # Force depth=1 everywhere
    },
    # A4: No warmup — personal head trained from round 1
    'A4_no_warmup': {
        'model_type': 'fairpfl',
        'fairness_q_init': 1.0,
        'proximal_mu': 0.001,
        'personal_warmup_rounds': 0,  # No warmup
        'dp': False,
        'personalization': True,
    },
    # A5: No proximal — removes personal drift penalty
    'A5_no_proximal': {
        'model_type': 'fairpfl',
        'fairness_q_init': 1.0,
        'proximal_mu': 0,  # No proximal regularization
        'personal_warmup_rounds': 20,
        'dp': False,
        'personalization': True,
    },
}

