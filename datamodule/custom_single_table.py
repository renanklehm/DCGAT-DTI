from __future__ import annotations

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class PairDataset(Dataset):
    def __init__(self, drug_features, target_features, pairs: pd.DataFrame):
        self.drug_features = drug_features.to_numpy(dtype=np.float32, copy=False)
        self.target_features = target_features.to_numpy(dtype=np.float32, copy=False)
        self.drug_indices = pairs["Drug_ID"].to_numpy(dtype=np.int64, copy=False)
        self.target_indices = pairs["Prot_ID"].to_numpy(dtype=np.int64, copy=False)
        self.labels = pairs["label"].to_numpy(dtype=np.float32, copy=False)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        drug_index = int(self.drug_indices[index])
        target_index = int(self.target_indices[index])
        x1 = torch.from_numpy(self.drug_features[drug_index])
        x2 = torch.from_numpy(self.target_features[target_index])
        y = torch.tensor(self.labels[index], dtype=torch.float32)
        return x1, x2, y, torch.tensor(drug_index), torch.tensor(target_index)


class ConcatDataset(Dataset):
    def __init__(self, drug_features, target_features, pairs: pd.DataFrame):
        self.drug_features = drug_features.to_numpy(dtype=np.float32, copy=False)
        self.target_features = target_features.to_numpy(dtype=np.float32, copy=False)
        self.drug_indices = pairs["Drug_ID"].to_numpy(dtype=np.int64, copy=False)
        self.target_indices = pairs["Prot_ID"].to_numpy(dtype=np.int64, copy=False)
        self.labels = pairs["label"].to_numpy(dtype=np.float32, copy=False)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        drug_index = int(self.drug_indices[index])
        target_index = int(self.target_indices[index])
        x = torch.from_numpy(np.concatenate([self.drug_features[drug_index], self.target_features[target_index]], axis=0))
        y = torch.tensor(self.labels[index], dtype=torch.float32)
        return x, y


class SingleTableDTIDataModule(pl.LightningDataModule):
    def __init__(self, config, dataset, dm_cfg, splitting, serializer):
        super().__init__()
        self.config = config
        self.X_drug = dataset["X_drug"]
        self.X_target = dataset["X_target"]
        self.test_ind = dataset["test"]
        self.batch_size = dm_cfg["batch_size"]
        self.num_workers = dm_cfg["num_workers"]
        self.module_target = config["module"]["_target_"]

    def _dataloader_kwargs(self):
        kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": True,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = True
        return kwargs

    def setup(self, stage=None):
        if self.module_target == "module.MLP.Net":
            self.test_dataset = ConcatDataset(self.X_drug, self.X_target, self.test_ind)
        else:
            self.test_dataset = PairDataset(self.X_drug, self.X_target, self.test_ind)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, drop_last=False, **self._dataloader_kwargs())
