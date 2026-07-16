"""
Data-loading Implementation

It consists of an IterableDataset and a Lightning DataModule
"""

from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
from torch.utils.data import IterableDataset, get_worker_info
import lightning as pl
import random

from pathlib import Path

# Label mapping for classification
LABEL_MAP = {
    "minbias": 0,
    "ggHbb": 1
}

# Function to extract label from file path (name of the folder)
def extract_label(file_path):
    process = Path(file_path).parent.name
    return LABEL_MAP[process]


class EventJetsL1TriggerDataset(IterableDataset):
    """
    IterableDataset for L1-trigger data from parquet files.

    Streams data lazily from parquet files instead of loading all into memory.
    Each event contains PUPPI particles with features: pT, eta, phi.
    """

    def __init__(
        self,
        parquet_dirs: List[str],
        max_jets: int = 8,
        features: List[str] = ["L1T_JetPuppiAK4_PT", "L1T_JetPuppiAK4_Eta", "L1T_JetPuppiAK4_Phi"],
        preprocessing: bool = True,
        shuffling: bool = False,
        labels: bool = False,
    ):
        """
        Initialize the dataset.

        Args:
            parquet_dirs: List of directories containing parquet files.
            max_jets: Maximum number of jets per event.
            features: List of feature to extract.
            preprocessing: Whether to apply preprocessing.
        """
        super().__init__()

        self.dataset = ds.dataset(parquet_dirs, format="parquet")
        self.max_jets = max_jets
        self.features = features
        self.preprocessing = preprocessing
        self.shuffling = shuffling
        self.labels = labels

    def _process_event(self, row: pd.Series) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single event row into padded features and mask.

        Returns:
            features: [max_jets, n_feats] tensor
            mask: [max_jets] boolean tensor
        """
        n_feats = len(self.features)

        pt = np.array(row["L1T_JetPuppiAK4_PT"])
        eta = np.array(row["L1T_JetPuppiAK4_Eta"])
        phi = np.array(row["L1T_JetPuppiAK4_Phi"])

        feats = np.zeros((self.max_jets, n_feats), dtype=np.float32)
        mask = np.zeros(self.max_jets, dtype=bool)

        n_jets = min(len(pt), self.max_jets)

        feats[:n_jets, 0] = pt[:n_jets]
        feats[:n_jets, 1] = eta[:n_jets]
        feats[:n_jets, 2] = phi[:n_jets]

        mask[:n_jets] = True

        # Preprocessing 
        if self.preprocessing:
            
            pt = np.log(pt + 1e-8) - 1.8  
            eta = eta / 3
            phi_sin = np.sin(phi)
            phi_cos = np.cos(phi)   
            
            feats = np.zeros((self.max_jets, n_feats + 1), dtype=np.float32)

            feats[:n_jets, 0] = pt[:n_jets]
            feats[:n_jets, 1] = eta[:n_jets]
            feats[:n_jets, 2] = phi_sin[:n_jets]   
            feats[:n_jets, 3] = phi_cos[:n_jets]     

        return torch.FloatTensor(feats), torch.BoolTensor(mask)

    def __iter__(self) -> Iterator[Tuple]:
        """
        Iterate over all events using pyarrow.dataset scanner.
        """

        worker_info = get_worker_info()

        files = self.dataset.files

        if self.shuffling:
            random.shuffle(files)

        if worker_info is None:
            assigned_files = files
        else:
            assigned_files = files[worker_info.id::worker_info.num_workers]
        
        buffer = []
        buffer_size = 5000
        
        for file_path in assigned_files:

            label = extract_label(file_path)

            dataset = ds.dataset(file_path, format="parquet")

            scanner = dataset.scanner(
                columns=self.features,
                use_threads=True,
            )

            for batch in scanner.to_batches():
                
                df = batch.to_pandas()

                for i in range(len(df)):
                    
                    event = df.iloc[i]

                    if len(df.iloc[i]["L1T_JetPuppiAK4_PT"]) == 0:
                        continue

                    if self.shuffling:

                        buffer.append((event,label))

                        if len(buffer) >= buffer_size:
                            
                            idx = random.randint(0, len(buffer)-1)

                            event, label = buffer.pop(idx)

                            if self.labels:
                                yield self._process_event(event), label
                            else:
                                yield self._process_event(event)

                    else:
                        if self.labels:
                            yield self._process_event(event), label
                        else:
                            yield self._process_event(event)

        if self.shuffling:        
            # Remaining events in buffer
            while buffer:

                idx = random.randint(0, len(buffer)-1)
                event, label = buffer.pop(idx)
                
                if self.labels:
                    yield self._process_event(event), label
                else:
                    yield self._process_event(event)


class EventJetsL1TriggerDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for L1-trigger data.
    """

    def __init__(
        self,
        cfg,
        batch_size: int = 32
    ):
        """
        Initialize the DataModule.

        Args:
            parquet_dirs_train: Directories containing training parquet files.
            parquet_dirs_val: Directories containing validation data.
            parquet_dirs_test: Directories containing test data.
            max_jets: Maximum jets per event.
            batch_size: Batch size for dataloaders.
            num_workers: Workers for dataloaders.
            features: Features to extract.
            preprocessing: Whether to apply preprocessing.
        """
        super().__init__()

        self.train_dirs = cfg.train_path
        self.val_dirs = cfg.val_path or []
        self.test_dirs = cfg.test_path or []
        self.max_jets = cfg.max_jets
        self.batch_size = batch_size
        self.num_workers = cfg.num_workers
        self.features = list(cfg.features)
        self.preprocessing = cfg.preprocessing

    def train_dataloader(self):
        """Return training dataloader."""
        self.train_dataset = EventJetsL1TriggerDataset(
            parquet_dirs=self.train_dirs,
            max_jets=self.max_jets,
            features=self.features,
            preprocessing=self.preprocessing,
            shuffling=True
        )
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
            #drop_last=True,
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        self.val_dataset = EventJetsL1TriggerDataset(
            parquet_dirs=self.val_dirs,
            max_jets=self.max_jets,
            features=self.features,
            preprocessing=self.preprocessing
        )
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            #drop_last=True,
        )

    def test_dataloader(self):
        """Return test dataloader."""
        self.test_dataset = EventJetsL1TriggerDataset(
            parquet_dirs=self.test_dirs,
            max_jets=self.max_jets,
            features=self.features,
            preprocessing=self.preprocessing
        )
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            #drop_last=True,
        )