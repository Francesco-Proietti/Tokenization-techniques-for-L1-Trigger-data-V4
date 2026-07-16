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


class JetConstL1TriggerDataset(IterableDataset):
    """
    IterableDataset for L1-trigger data from parquet files.

    Streams data lazily from parquet files instead of loading all into memory.
    Each event contains PUPPI particles and jets.
    """

    def __init__(
        self,
        parquet_dirs: List[str],
        max_particles: int = 128,
        features: List[str] = ["L1T_PUPPIPart_PT", "L1T_PUPPIPart_Eta", "L1T_PUPPIPart_Phi", "L1T_PUPPIPart_PuppiW"],
        preprocessing: bool = True,
        shuffling: bool = False,
        labels: bool = False
    ):
        """
        Initialize the dataset.

        Args:
            parquet_dirs: List of directories containing parquet files.
            max_particles: Maximum number of particles per jet.
            features: List of feature to extract.
            preprocessing: Whether to apply preprocessing.
        """
        super().__init__()

        self.dataset = ds.dataset(parquet_dirs, format="parquet")
        self.max_particles = max_particles
        self.features = features
        self.kin_coord_num = 3
        self.preprocessing = preprocessing
        self.shuffling = shuffling
        self.labels = labels

    def _process_event(self, row: pd.Series) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single event row into padded features and mask per jet. (constituent-level)
        Each event can contain more than one jet.

        Returns:
            features: [max_particles, kin_features (3)] tensor
            mask: [max_particles] boolean tensor
        """

        kin_coord_num = self.kin_coord_num

        const_pt = np.array(row["L1T_PUPPIPart_PT"])
        const_eta = np.array(row["L1T_PUPPIPart_Eta"])
        const_phi = np.array(row["L1T_PUPPIPart_Phi"])
        const_idx = np.array(row["L1T_JetPuppiAK4_ConstituentsIdx"])

        jet_pt = np.array(row["L1T_JetPuppiAK4_PT"])
        jet_eta = np.array(row["L1T_JetPuppiAK4_Eta"])
        jet_phi = np.array(row["L1T_JetPuppiAK4_Phi"])
        jet_mass = np.array(row["L1T_JetPuppiAK4_Mass"])
        
        # For loop among jets of the same event
        for i, j in enumerate(const_idx):
            
            #Jet-level features
            jet_features = torch.FloatTensor([
                jet_pt[i],
                jet_eta[i],
                jet_phi[i],
                jet_mass[i],
            ])

            # Apply constituents' mask
            j_const_pt = const_pt[j]
            j_const_eta = const_eta[j]
            j_const_phi = const_phi[j]

            # Preprocessing 
            if self.preprocessing:
                
                # Relative features WRT the jet axis
                j_const_eta = j_const_eta - jet_eta[i]
                j_const_phi = j_const_phi - jet_phi[i]
                j_const_phi = (j_const_phi + np.pi) % (2 * np.pi) - np.pi
                
                # Scaling
                j_const_pt = np.log(j_const_pt + 1e-8) - 1.8
                j_const_eta = j_const_eta / 3
                j_const_phi = j_const_phi / 3

            n_particles = min(len(j_const_pt), self.max_particles)

            feats = np.zeros((self.max_particles, kin_coord_num), dtype=np.float32)
            mask = np.zeros(self.max_particles, dtype=bool)

            feats[:n_particles, 0] = j_const_pt[:n_particles]
            feats[:n_particles, 1] = j_const_eta[:n_particles]
            feats[:n_particles, 2] = j_const_phi[:n_particles]

            mask[:n_particles] = True
            
            yield (
                torch.FloatTensor(feats),
                torch.BoolTensor(mask),
                jet_features,
            )

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

                    if len(event["L1T_JetPuppiAK4_ConstituentsIdx"]) == 0:
                        continue

                    if self.shuffling:

                        buffer.append((event, label))

                        if len(buffer) >= buffer_size:

                            idx = random.randint(0, len(buffer)-1)
                            
                            event, label = buffer.pop(idx)
                            
                            if self.labels:

                                for data in self._process_event(event):
                                    yield data, label
                            else:
                                for data in self._process_event(event):
                                    yield data
                    else:
                        if self.labels:
                            for data in self._process_event(event):
                                yield data, label
                        else:
                            for data in self._process_event(event):
                                yield data
                        

        if self.shuffling:        
            # Remaining events in buffer
            while buffer:

                idx = random.randint(0, len(buffer)-1)
                event, label = buffer.pop(idx)

                if self.labels:
                    for data in self._process_event(event):
                        yield data, label
                else:
                    for data in self._process_event(buffer.pop(idx)):
                        yield data                 


class JetConstL1TriggerDataModule(pl.LightningDataModule):
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
            max_particles: Maximum particles per event.
            batch_size: Batch size for dataloaders.
            num_workers: Workers for dataloaders.
            features: Features to extract.
            puppiw_threshold: Minimum PUPPI weight.
        """
        super().__init__()

        self.train_dirs = cfg.train_path
        self.val_dirs = cfg.val_path or []
        self.test_dirs = cfg.test_path or []
        self.max_particles = cfg.max_particles
        self.batch_size = batch_size
        self.num_workers = cfg.num_workers
        self.features = list(cfg.features)
        self.preprocessing = cfg.preprocessing

    def train_dataloader(self):
        """Return training dataloader."""
        self.train_dataset = JetConstL1TriggerDataset(
            parquet_dirs=self.train_dirs,
            max_particles=self.max_particles,
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
        self.val_dataset = JetConstL1TriggerDataset(
            parquet_dirs=self.val_dirs,
            max_particles=self.max_particles,
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
        self.test_dataset = JetConstL1TriggerDataset(
            parquet_dirs=self.test_dirs,
            max_particles=self.max_particles,
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
