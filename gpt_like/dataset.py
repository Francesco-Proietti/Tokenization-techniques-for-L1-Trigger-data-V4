import torch
import lightning as pl

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from sklearn.model_selection import train_test_split


class TokenDataset(Dataset):
    
    def __init__(
        self,
        tokens_path,
        masks_path,
        cb_size:int = 512,
        labels_path=None
    ):
        
        self.pad_token = cb_size
        
        self.tokens = torch.load(tokens_path)
        self.masks = torch.load(masks_path)

        self.labels = None

        if labels_path is not None:
            self.labels = torch.load(labels_path)


        assert len(self.tokens) == len(self.masks)

        if self.labels is not None:
            assert len(self.tokens) == len(self.labels)
        
        self.tokens = self.tokens.clone()

        self.tokens[self.tokens == -1] = self.pad_token


    def __len__(self):

        return len(self.tokens)


    def __getitem__(self, idx):

        item = {
            "tokens": self.tokens[idx],
            "mask": self.masks[idx],
        }

        if self.labels is not None:
            item["label"] = self.labels[idx]

        return item


class TokenDataModule(pl.LightningDataModule):

    def __init__(
        self,
        tokens_path: str,
        masks_path: str,
        labels_path: str,
        cb_size: int = 512,
        batch_size: int = 32,
        num_workers: int = 0,
        test_size: float = 0.10,
        val_size: float = 0.20,
        seed: int = 42,
    ):
        super().__init__()

        self.tokens_path = tokens_path
        self.masks_path = masks_path
        self.labels_path = labels_path

        self.cb_size = cb_size

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.test_size = test_size
        self.val_size = val_size

        self.seed = seed

    def setup(self, stage=None):

        dataset = TokenDataset(
            tokens_path=self.tokens_path,
            masks_path=self.masks_path,
            labels_path=self.labels_path,
            cb_size=self.cb_size,
        )

        labels = dataset.labels.numpy()

        indices = torch.arange(len(dataset)).numpy()

        # -------------------------
        # train + val / test
        # -------------------------

        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_size,
            stratify=labels,
            random_state=self.seed,
            shuffle=True,
        )

        # -------------------------
        # train / val
        # -------------------------

        train_val_labels = labels[train_val_idx]

        val_fraction = self.val_size / (1.0 - self.test_size)

        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_fraction,
            stratify=train_val_labels,
            random_state=self.seed,
            shuffle=True,
        )

        self.train_dataset = Subset(dataset, train_idx)
        self.val_dataset = Subset(dataset, val_idx)
        self.test_dataset = Subset(dataset, test_idx)


    def train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
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