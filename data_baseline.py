 #!/usr/bin/env python3


#Import libraries
import os
import re
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm

import lightning as pl

from src.data.event_jets_data_loading import EventJetsL1TriggerDataset, EventJetsL1TriggerDataModule


def generate_dataset(
    dataset,
    max_batches: int = None
) -> Tuple[List[torch.Tensor], List[str], List[int]]:
    """
    Generate tokens for a dataset.
    
    Args:
        dataset: Iterable dataset
        max_batches: Optional limit on number of batches
        
    Returns:
        data_list: List of data tensors (one per event/jet)
        labels_list: List of process labels
        file_indices: List of file indices for tracking
    """


    data_list = []
    masks_list = []
    labels_list = []
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        num_workers=0,
        pin_memory=True
    )

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating tokens")):
        
        if max_batches and batch_idx >= max_batches:
            break

        x_mask, l = batch
        x, mask = x_mask

        # Store tokens, masks and labels 
        data_list.append(x.cpu())
        masks_list.append(mask.cpu())
        labels_list.append(l.cpu())

    return data_list, masks_list, labels_list

def generate_and_save_data(
    parquet_dirs: List[str],
    output_dir: str,
    batch_size: int = 32,
    max_events: int = None
):
    """
    Main function to generate and save data.
    
    Args:
        parquet_dirs: List of directories with parquet files
        output_dir: Directory to save data
        max_particles: Max particles per event
        batch_size: Batch size for inference
        max_events: Max events to process (None for all)
    """

    # Create dataset
    print(f"Creating dataset for...")

    
    dataset = EventJetsL1TriggerDataset(
        parquet_dirs=parquet_dirs,
        max_jets=14,
        features=["L1T_JetPuppiAK4_PT", "L1T_JetPuppiAK4_Eta", "L1T_JetPuppiAK4_Phi"],
        preprocessing=True,
        shuffling=False,
        labels=True
    )
    

    # Generate tokens
    print("Generating tokens...")
    data_list, masks_list, labels_list = generate_dataset(
        dataset, max_batches=None
    )

    # Save tokens
    os.makedirs(output_dir, exist_ok=True)

    # Combine all batches
    all_data = torch.cat(data_list, dim=0)  # [total_events, max_particles/jets]
    all_masks = torch.cat(masks_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    
    # Save as PyTorch tensor
    data_path = os.path.join(output_dir, "data.pt")
    torch.save(all_data, data_path)
    print(f"Saved data to {data_path}")

    # Save masks
    masks_path = os.path.join(output_dir, "masks.pt")
    torch.save(all_masks, masks_path)
    print(f"Saved masks to {masks_path}")

    # Save labels
    labels_path = os.path.join(output_dir, "labels.pt")
    torch.save(all_labels, labels_path)
    print(f"Saved labels to {labels_path}")
    

    # Save config
    config = {
        "num_events": all_data.shape[0],
        "unique_labels": torch.unique(all_labels).tolist(),
    }
    config_path = os.path.join(output_dir, "config.pt")
    torch.save(config, config_path)
    print(f"Saved config to {config_path}")

    # Print summary
    print("\n" + "="*50)
    print("DATASET GENERATION SUMMARY")
    print("="*50)
    print(f"Total events: {all_data.shape[0]}")
    print(f"Unique labels: {config['unique_labels']}")


    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate dataset for classification baseline")
    parser.add_argument("--parquet-dir", required=True, help="Directory with parquet files")
    parser.add_argument("--output-dir", required=True, help="Directory to save data")
    parser.add_argument("--max-particles", type=int, default=128,
                        help="Max particles per event (default: 128)")

    args = parser.parse_args()

    generate_and_save_data(
        parquet_dirs=args.parquet_dir,
        output_dir=args.output_dir,
    )