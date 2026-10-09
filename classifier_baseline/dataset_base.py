import torch
import lightning as pl

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from sklearn.model_selection import train_test_split


class OriginalDataset(Dataset):
    
    def __init__(
        self,
        data_path,
        masks_path,
        labels_path=None
    ):
        
        
        self.data = torch.load(data_path)
        self.masks = torch.load(masks_path)

        self.labels = None

        if labels_path is not None:
            self.labels = torch.load(labels_path)


        assert len(self.data) == len(self.masks)

        if self.labels is not None:
            assert len(self.data) == len(self.labels)
        
        self.data = self.data.clone()


    def __len__(self):

        return len(self.data)


    def __getitem__(self, idx):

        item = {
            "data": self.data[idx],
            "mask": self.masks[idx],
        }

        if self.labels is not None:
            item["label"] = self.labels[idx]

        return item


class OriginalDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_path: str,
        masks_path: str,
        labels_path: str,
        batch_size: int = 32,
        num_workers: int = 0,
        test_size: float = 0.20,
        seed: int = 56,
    ):
        super().__init__()

        self.data_path = data_path
        self.masks_path = masks_path
        self.labels_path = labels_path


        self.batch_size = batch_size
        self.num_workers = num_workers

        self.test_size = test_size

        self.seed = seed

    def setup(self, stage=None):

        dataset = OriginalDataset(
            data_path=self.data_path,
            masks_path=self.masks_path,
            labels_path=self.labels_path,
        )

        labels = dataset.labels.numpy()

        indices = torch.arange(len(dataset)).numpy()

        # -------------------------
        # train + val / test
        # -------------------------

        train_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_size,
            stratify=labels,
            random_state=self.seed,
            shuffle=True,
        )

        self.train_dataset = Subset(dataset, train_idx)
        self.test_dataset = Subset(dataset, test_idx)


    def train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )