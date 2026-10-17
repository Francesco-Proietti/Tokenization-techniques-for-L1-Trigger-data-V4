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

from src.models.transformer_vqvae import TransformerVQVAE


def generate_tokens_for_dataset(
    dataset,
    model: pl.LightningModule,
    max_batches: int = None
) -> Tuple[List[torch.Tensor], List[str], List[int]]:
    """
    Generate recon dataset for classification.
    
    Args:
        dataset: Iterable dataset
        model: Trained VQ-VAE model
        max_batches: Optional limit on number of batches
        
    Returns:
        data_reco_list: List of data_reco tensors (one per event/jet)
        labels_list: List of process labels
        file_indices: List of file indices for tracking
    """

    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    data_reco_list = []
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
        x, mask = x.to(device), mask.to(device)
        l = l.to(device)

        # Get data from model
        with torch.no_grad():
            output = model(x, mask)
            recon = output[0]  # [B, N] - quantization indices

        # Store data, masks and labels 
        data_reco_list.append(recon.cpu())
        masks_list.append(mask.cpu())
        labels_list.append(l.cpu())

    return data_reco_list, masks_list, labels_list

def generate_and_save_tokens(
    checkpoint_path: str,
    parquet_dirs: List[str],
    output_dir: str,
    max_particles: int = 128,
    batch_size: int = 32,
    max_events: int = None
):
    """
    Main function to generate and save tokens.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        parquet_dirs: List of directories with parquet files
        output_dir: Directory to save tokens
        max_particles: Max particles per event
        batch_size: Batch size for inference
        max_events: Max events to process (None for all)
    """
    print(f"Loading model from {checkpoint_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = TransformerVQVAE.load_from_checkpoint(checkpoint_path, weights_only=False)

    model.to(device)
    model.eval()

    # Get hyperparameters from checkpoint
    ckpt = torch.load(checkpoint_path, weights_only=False)
    cfg = ckpt["hyper_parameters"]["cfg"]
    codebook_size = cfg.get("codebook_size", 256)

    # Create dataset
    print(f"Creating dataset...")

    
    dataset = EventJetsL1TriggerDataset(
        parquet_dirs=parquet_dirs,
        max_jets=14,
        features=["L1T_JetPuppiAK4_PT", "L1T_JetPuppiAK4_Eta", "L1T_JetPuppiAK4_Phi"],
        preprocessing=True,
        shuffling=False,
        labels=True
    )
    

    # Generate data
    print("Generating data...")
    data_reco_list, masks_list, labels_list = generate_tokens_for_dataset(
        dataset, model, max_batches=None
    )

    # Save data
    os.makedirs(output_dir, exist_ok=True)

    # Combine all batches
    all_data_reco = torch.cat(data_reco_list, dim=0)  # [total_events, max_particles/jets, features]
    all_masks = torch.cat(masks_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    
    # Save as PyTorch tensor
    data_reco_path = os.path.join(output_dir, "data_reco.pt")
    torch.save(all_data_reco, data_reco_path)
    print(f"Saved data to {data_reco_path}")

    # Save masks
    masks_path = os.path.join(output_dir, "masks.pt")
    torch.save(all_masks, masks_path)
    print(f"Saved masks to {masks_path}")

    # Save labels
    labels_path = os.path.join(output_dir, "labels.pt")
    torch.save(all_labels, labels_path)
    print(f"Saved labels to {labels_path}")
    
    # Save codebook
    cb_path = os.path.join(output_dir, "codebook.pt")
    torch.save(model.quantizer.codebook, cb_path)
    print(f"Saved codebook to {cb_path}")

    # Save config
    config = {
        "checkpoint": checkpoint_path,
        "codebook_size": codebook_size,
        "max_particles": 14,
        "num_events": all_data_reco.shape[0],
        "unique_labels": torch.unique(all_labels).tolist(),
    }
    config_path = os.path.join(output_dir, "config.pt")
    torch.save(config, config_path)
    print(f"Saved config to {config_path}")

    # Print summary
    print("\n" + "="*50)
    print("DATA GENERATION SUMMARY")
    print("="*50)
    print(f"Total events: {all_data_reco.shape[0]}")
    print(f"Codebook size: {codebook_size}")
    print(f"Unique labels: {config['unique_labels']}")


    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate tokens from VQ-VAE")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--parquet-dir", required=True, help="Directory with parquet files")
    parser.add_argument("--output-dir", required=True, help="Directory to save tokens")
    parser.add_argument("--max-particles", type=int, default=128,
                        help="Max particles per event (default: 128)")

    args = parser.parse_args()

    generate_and_save_tokens(
        checkpoint_path=args.checkpoint,
        parquet_dirs=args.parquet_dir,
        output_dir=args.output_dir,
        max_particles=args.max_particles
    )