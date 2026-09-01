# models/dual_personalizer.py — Dual personalization: device-type × attack-type
# Key differentiator vs Sun et al. (2025)

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 8-class UNB taxonomy (runtime remapped from 34-class)
# After remapping by utils/mmap_dataset.REMAP_34_TO_8:
#   [0] Benign      — was BenignTraffic (34-class ID 1)
#   [1] BruteForce  — was DictionaryBruteForce (34-class ID 17)
#   [2] DDoS        — was 12 DDoS sub-types (34-class IDs 4-15)
#   [3] DoS         — was 4 DoS sub-types (34-class IDs 18-21)
#   [4] Mirai       — was 3 Mirai sub-types (34-class IDs 23-25)
#   [5] Recon       — was 5 Recon sub-types (34-class IDs 26-29,32)
#   [6] Spoofing    — was DNS_Spoofing + MITM (34-class IDs 16,22)
#   [7] Web-based   — was 6 Web sub-types (34-class IDs 0,2,3,30,31,33)
# ============================================================

# Category ID → list of 8-class IDs in that category
# In 8-class taxonomy, each category maps to exactly one class.
# This structure is preserved for DualPersonalizedHead compatibility.
CATEGORY_CLASSES = {
    0: [2],   # DDoS      → 8-class ID 2
    1: [3],   # DoS       → 8-class ID 3
    2: [5],   # Recon     → 8-class ID 5
    3: [7],   # Web-based → 8-class ID 7
    4: [1],   # BruteForce → 8-class ID 1
    5: [6],   # Spoofing  → 8-class ID 6
    6: [4],   # Mirai     → 8-class ID 4
}
BENIGN_CLASS_ID = 0  # 8-class alphabetical: [0]Benign
NUM_CATEGORIES = len(CATEGORY_CLASSES) + 1  # 7 attack categories + benign = 8


class DualPersonalizedHead(nn.Module):
    """
    Two-stage attack classification head.
    Stage 1: Detect attack CATEGORY (8 classes: 7 categories + benign)
    Stage 2: Classify within each category (fine-grained)
    """

    def __init__(self, input_dim=64, num_attack_categories=7, num_classes=8):
        super().__init__()
        self.num_classes = num_classes

        # Stage 1: Category detection
        self.category_detector = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_attack_categories + 1),  # +1 for benign
        )

        # Stage 2: Fine-grained classifier per category
        self.fine_classifiers = nn.ModuleDict()
        for cat_id, class_ids in CATEGORY_CLASSES.items():
            self.fine_classifiers[str(cat_id)] = nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.ReLU(),
                nn.Linear(16, len(class_ids)),
            )

    def forward(self, x):
        """
        Two-stage classification:
        1. Detect attack category (or benign)
        2. Classify within category

        Args: x: (batch, input_dim) — attended feature vector
        Returns:
            fine_logits: (batch, C) — class logits (C=8 in 8-class taxonomy)
            cat_logits: (batch, 8) — category-level logits

        Note: In 8-class taxonomy, each category maps to exactly one class,
        so fine_logits and cat_logits are structurally equivalent. The dual
        head architecture is preserved for ablation analysis (S6-S10).
        """
        cat_logits = self.category_detector(x)  # (B, 8)

        # Build fine-grained logits by combining category + fine predictions
        fine_logits = torch.zeros(x.size(0), self.num_classes, device=x.device)
        fine_logits[:, BENIGN_CLASS_ID] = cat_logits[:, 0]  # Benign score

        for cat_id, class_ids in CATEGORY_CLASSES.items():
            cat_fine = self.fine_classifiers[str(cat_id)](x)  # (B, n_sub_classes)
            for i, cls_id in enumerate(class_ids):
                fine_logits[:, cls_id] = cat_logits[:, cat_id + 1] + cat_fine[:, i]

        return fine_logits, cat_logits


class DualFairPFLModel(nn.Module):
    """
    FairPFL with dual personalization:
    - Dimension 1: Device-type (via PersonalizedLayer LSTM)
    - Dimension 2: Attack-type (via DualPersonalizedHead)
    Used for S8/S9 ablation.
    """

    def __init__(self, input_dim=46, personal_dim=128, shared_dim=64,
                 num_classes=8, num_heads=4, dropout=0.3):
        super().__init__()

        # Dimension 1: Device-type personalization
        self.device_personal = nn.LSTM(
            input_dim, personal_dim, batch_first=True, num_layers=1)

        # Shared global feature extractor
        self.shared_lstm = nn.LSTM(
            personal_dim, shared_dim, batch_first=True, num_layers=1)
        self.shared_attention = nn.MultiheadAttention(
            shared_dim, num_heads, batch_first=True)
        self.shared_norm = nn.LayerNorm(shared_dim)

        # Dimension 2: Attack-type personalization
        self.attack_personal = DualPersonalizedHead(shared_dim, 7, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.device_personal(x)
        out = self.dropout(out)
        out, _ = self.shared_lstm(out)
        attn_out, attn_w = self.shared_attention(out, out, out)
        out = self.shared_norm(attn_out + out).mean(dim=1)
        fine_logits, cat_logits = self.attack_personal(out)
        return fine_logits, attn_w, cat_logits

    def get_device_personal_params(self):
        return dict(self.device_personal.named_parameters())

    def get_attack_personal_params(self):
        return dict(self.attack_personal.named_parameters())

    def get_personal_params(self):
        """All personalized params with FULL MODEL-SCOPED keys.

        device_personal keys are prefixed with 'device_personal.' and
        attack_personal keys with 'attack_personal.' to match state_dict()
        for correct disk save/load across Flower rounds.
        """
        params = {f'device_personal.{k}': v
                  for k, v in self.device_personal.named_parameters()}
        for k, v in self.attack_personal.named_parameters():
            params[f'attack_personal.{k}'] = v
        return params

    def get_shared_params(self):
        """Shared params sent to server for aggregation."""
        params = {}
        for k, v in self.shared_lstm.named_parameters():
            params[f'shared_lstm.{k}'] = v
        for k, v in self.shared_attention.named_parameters():
            params[f'shared_attention.{k}'] = v
        for k, v in self.shared_norm.named_parameters():
            params[f'shared_norm.{k}'] = v
        return params


class DualFairPFLModelWithLLM(nn.Module):
    """
    Final model: ALL differentiators combined.
    - Device-type personalization (LSTM)
    - Attack-type personalization (DualPersonalizedHead)
    - LLM semantic enrichment (DistilBERT embeddings, 64-dim)
    - Shared global feature extractor

    Input: raw_features (46) + llm_embeddings (64) = 110-dim
    This is the model for FINAL paper results.
    """

    def __init__(self, raw_dim=46, llm_dim=64, personal_dim=128,
                 shared_dim=64, num_classes=8, num_heads=4, dropout=0.3):
        super().__init__()
        input_dim = raw_dim + llm_dim  # 110

        self.device_personal = nn.LSTM(
            input_dim, personal_dim, batch_first=True, num_layers=1)
        self.shared_lstm = nn.LSTM(
            personal_dim, shared_dim, batch_first=True, num_layers=1)
        self.shared_attention = nn.MultiheadAttention(
            shared_dim, num_heads, batch_first=True)
        self.shared_norm = nn.LayerNorm(shared_dim)
        self.attack_personal = DualPersonalizedHead(shared_dim, 7, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_enriched):
        """Args: x_enriched: (B, seq_len, 110) — raw + LLM embeddings."""
        out, _ = self.device_personal(x_enriched)
        out = self.dropout(out)
        out, _ = self.shared_lstm(out)
        attn_out, attn_w = self.shared_attention(out, out, out)
        out = self.shared_norm(attn_out + out).mean(dim=1)
        fine_logits, cat_logits = self.attack_personal(out)
        return fine_logits, attn_w, cat_logits

    def get_personal_params(self):
        """All personalized params with FULL MODEL-SCOPED keys.

        device_personal keys prefixed with 'device_personal.',
        attack_personal keys prefixed with 'attack_personal.'
        to match state_dict() for correct disk save/load.
        """
        params = {f'device_personal.{k}': v
                  for k, v in self.device_personal.named_parameters()}
        for k, v in self.attack_personal.named_parameters():
            params[f'attack_personal.{k}'] = v
        return params

    def get_shared_params(self):
        params = {}
        for k, v in self.shared_lstm.named_parameters():
            params[f'shared_lstm.{k}'] = v
        for k, v in self.shared_attention.named_parameters():
            params[f'shared_attention.{k}'] = v
        for k, v in self.shared_norm.named_parameters():
            params[f'shared_norm.{k}'] = v
        return params


def labels_to_categories(labels):
    """Convert 8-class labels to category labels for DualPersonalizedHead.

    8-class mapping:
      Label 0 (Benign)     → Category 0 (Benign)
      Label 1 (BruteForce) → Category 5 (cat_id+1=5)
      Label 2 (DDoS)       → Category 1 (cat_id+1=1)
      Label 3 (DoS)        → Category 2 (cat_id+1=2)
      Label 4 (Mirai)      → Category 7 (cat_id+1=7)
      Label 5 (Recon)      → Category 3 (cat_id+1=3)
      Label 6 (Spoofing)   → Category 6 (cat_id+1=6)
      Label 7 (Web-based)  → Category 4 (cat_id+1=4)

    Returns tensor of same shape as labels with category IDs.
    """
    cat_map = torch.zeros(8, dtype=torch.long)
    # Build from CATEGORY_CLASSES (cat_id+1 because 0=benign)
    cat_map[BENIGN_CLASS_ID] = 0
    for cat_id, class_ids in CATEGORY_CLASSES.items():
        for cls_id in class_ids:
            cat_map[cls_id] = cat_id + 1  # +1 because 0=benign

    device = labels.device if isinstance(labels, torch.Tensor) else 'cpu'
    cat_map = cat_map.to(device)

    if isinstance(labels, torch.Tensor):
        return cat_map[labels]
    else:
        return cat_map[torch.tensor(labels)]


def dual_loss(fine_logits, cat_logits, labels, category_labels, alpha=0.3):
    """
    Combined loss for dual personalization.
    L = (1-α) * L_fine + α * L_category

    Args:
        fine_logits: (B, C) class logits (C=8 in 8-class taxonomy)
        cat_logits: (B, 8) category-level logits
        labels: (B,) ground truth class labels (0-7)
        category_labels: (B,) category labels (0-7)
        alpha: trade-off parameter (0.3 recommended)
    """
    l_fine = F.cross_entropy(fine_logits, labels)
    l_cat = F.cross_entropy(cat_logits, category_labels)
    return (1 - alpha) * l_fine + alpha * l_cat
