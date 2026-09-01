# utils/checkpoint.py — Model checkpointing and experiment logging

import torch
import os
import json
import time


def save_checkpoint(model, optimizer, round_num, metrics, path):
    """Save model checkpoint with metrics."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'round': round_num,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'metrics': metrics,
        'timestamp': time.time(),
    }, path)
    print(f"Checkpoint saved: round {round_num} → {path}")


def load_checkpoint(model, optimizer, path, device='cpu'):
    """Load model checkpoint. Returns round number and metrics."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and checkpoint.get('optimizer_state_dict'):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['round'], checkpoint.get('metrics', {})


class ExperimentLogger:
    """Log all metrics per round to JSON for analysis."""

    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.history = []
        self.start_time = time.time()

    def log_round(self, round_num, metrics_dict):
        """Log one round's metrics incrementally."""
        entry = {
            'round': round_num,
            'elapsed_seconds': time.time() - self.start_time,
            **metrics_dict
        }
        self.history.append(entry)

        # Save incrementally
        with open(os.path.join(self.log_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f, indent=2, default=str)

    def log_final(self, summary_dict):
        """Log final experiment summary."""
        summary_dict['total_elapsed_seconds'] = time.time() - self.start_time
        with open(os.path.join(self.log_dir, 'summary.json'), 'w') as f:
            json.dump(summary_dict, f, indent=2, default=str)
        print(f"Final summary saved to {self.log_dir}/summary.json")
