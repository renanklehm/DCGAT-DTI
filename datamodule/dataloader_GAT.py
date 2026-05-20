import torch,sys
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from utils import utils
import numpy as np 

# pytorch datalaoder
class MyDataset(Dataset):
    def __init__(self, drug, target, DTI):
        self.drug = drug.to_numpy(dtype=np.float32, copy=False)
        self.target = target.to_numpy(dtype=np.float32, copy=False)
        dti_values = DTI.iloc[:, :3].to_numpy()
        self.drug_indices = dti_values[:, 0].astype(np.int64, copy=False)
        self.target_indices = dti_values[:, 1].astype(np.int64, copy=False)
        self.labels = dti_values[:, 2].astype(np.float32, copy=False)
        
    def __getitem__(self, index):
        drug_index = int(self.drug_indices[index])
        target_index = int(self.target_indices[index])
        x1 = torch.from_numpy(self.drug[drug_index])
        x2 = torch.from_numpy(self.target[target_index])
        y = torch.tensor(self.labels[index], dtype=torch.float32)
        return x1, x2, y, drug_index, target_index

    def __len__(self):
        return len(self.labels)


class UNIDataModule(pl.LightningDataModule):
    def __init__(self,config,dataset,dm_cfg,splitting,serializer):
        super().__init__()
        self.X_drug = dataset['X_drug']
        self.X_target = dataset['X_target']
        self.train_ind = dataset['train']
        self.val_ind = dataset['val']
        self.test_ind = dataset['test']    
        self.batch_size = dm_cfg['batch_size']
        self.num_workers = dm_cfg['num_workers']
        self.config = config

    def _dataloader_kwargs(self):
        kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": True,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = True
        return kwargs
        
    def setup(self, stage):
        self.train_dataset = MyDataset(self.X_drug, self.X_target, self.train_ind)
        self.val_dataset = MyDataset(self.X_drug, self.X_target, self.val_ind)
        self.test_dataset = MyDataset(self.X_drug, self.X_target, self.test_ind)

    def train_dataloader(self):
        # REQUIRED
        return DataLoader(self.train_dataset, shuffle=False, drop_last=True, **self._dataloader_kwargs())

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(self.val_dataset, shuffle=False, drop_last=False, **self._dataloader_kwargs())

    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(self.test_dataset, shuffle=False, drop_last=False, **self._dataloader_kwargs())
