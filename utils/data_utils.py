# utils/data_utils.py — PyTorch Dataset and DataLoader creation

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class IoTDataset(Dataset):
    """PyTorch Dataset for cluster-level IoT intrusion detection data."""

    def __init__(self, X, y, seq_len=1):
        """
        Args:
            X: numpy array (N, num_features) features
            y: numpy array (N,) integer labels
            seq_len: sliding window size (1 = no temporal structure)
        """
        if seq_len > 1:
            self.X, self.y = self._create_sequences(X, y, seq_len)
        else:
            # Add sequence dimension: (N, 1, F)
            self.X = X.reshape(-1, 1, X.shape[1])
            self.y = y

        self.X = torch.FloatTensor(self.X)
        self.y = torch.LongTensor(self.y)

    def _create_sequences(self, X, y, seq_len):
        """Create sliding window sequences."""
        seqs, labels = [], []
        for i in range(len(X) - seq_len + 1):
            seqs.append(X[i:i + seq_len])
            labels.append(y[i + seq_len - 1])
        return np.array(seqs), np.array(labels)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_dataloaders(cluster_data, feature_cols, batch_size=64, seq_len=1,
                       num_workers=0, pin_memory=True):
    """
    Create DataLoaders for all clusters.

    Args:
        cluster_data: dict {cluster_id: {'train_X', 'train_y', 'val', 'test'}}
        feature_cols: list of feature column names
        batch_size: batch size for training
        seq_len: sliding window size
        num_workers: DataLoader workers (0 for low-memory)
        pin_memory: pin to GPU memory

    Returns:
        dict: {cluster_id: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}}
    """
    loaders = {}
    for k, splits in cluster_data.items():
        # Training data (already numpy from SMOTE or preprocessing)
        train_ds = IoTDataset(splits['train_X'], splits['train_y'], seq_len)

        # Validation data (from DataFrame)
        val_X = splits['val'][feature_cols].values.astype(np.float32)
        val_y = splits['val']['label_encoded'].values
        val_ds = IoTDataset(val_X, val_y, seq_len)

        # Test data (from DataFrame)
        test_X = splits['test'][feature_cols].values.astype(np.float32)
        test_y = splits['test']['label_encoded'].values
        test_ds = IoTDataset(test_X, test_y, seq_len)

        loaders[k] = {
            'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=pin_memory),
            'val': DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory),
            'test': DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False,
                               num_workers=num_workers, pin_memory=pin_memory),
        }
        print(f"Cluster {k}: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    return loaders
