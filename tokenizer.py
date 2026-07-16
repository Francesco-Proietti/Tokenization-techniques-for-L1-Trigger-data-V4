 #!/usr/bin/env python3

"""
Generate tokens from a trained VQ-VAE model.
Saves tokens for downstream GPT training.
"""

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
from src.data.event_particles_data_loading import EventPartL1TriggerDataset, EventPartL1TriggerDataModule
from src.data.jet_constituents_data_loading import JetConstL1TriggerDataset, JetConstL1TriggerDataModule
from src.models.mlp_vqvae import MLPVQVAE
from src.models.transformer_vqvae import TransformerVQVAE


def generate_tokens_for_dataset(
    dataset,
    model: pl.LightningModule,
    data_type: str,
    max_batches: int = None
) -> Tuple[List[torch.Tensor], List[str], List[int]]:
    """
    Generate tokens for a dataset.
    
    Args:
        dataset: Iterable dataset
        model: Trained VQ-VAE model
        device: Device to run on ('cuda' or 'cpu')
        max_batches: Optional limit on number of batches
        
    Returns:
        tokens_list: List of token tensors (one per event/jet)
        labels_list: List of process labels
        file_indices: List of file indices for tracking
    """

    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    tokens_list = []
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

        if data_type == "jet_const":
            x_mask_j, l = batch
            x, mask, j = x_mask_j
            x, mask, j = x.to(device), mask.to(device), j.to(device)
            l = l.to(device)
        else:
            x_mask, l = batch
            x, mask = x_mask
            x, mask = x.to(device), mask.to(device)
            l = l.to(device)

        # Get tokens from model
        with torch.no_grad():
            output = model(x, mask)
            tokens = output[2]  # [B, N] - quantization indices

        # Store tokens, masks and labels 
        tokens_list.append(tokens.cpu())
        masks_list.append(mask.cpu())
        labels_list.append(l.cpu())

    return tokens_list, masks_list, labels_list

def generate_and_save_tokens(
    checkpoint_path: str,
    parquet_dirs: List[str],
    data_type: str,
    output_dir: str,
    model_type: str = "mlp",
    max_particles: int = 128,
    batch_size: int = 32,
    max_events: int = None
):
    """
    Main function to generate and save tokens.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        parquet_dirs: List of directories with parquet files
        data_type: 'jet_const', 'event_part', or 'event_jets'
        output_dir: Directory to save tokens
        model_type: 'mlp' or 'transformer'
        max_particles: Max particles per event
        batch_size: Batch size for inference
        max_events: Max events to process (None for all)
    """
    print(f"Loading model from {checkpoint_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    if model_type == "mlp":
        model = MLPVQVAE.load_from_checkpoint(checkpoint_path, weights_only=False)
    else:
        model = TransformerVQVAE.load_from_checkpoint(checkpoint_path, weights_only=False)

    model.to(device)
    model.eval()

    # Get hyperparameters from checkpoint
    ckpt = torch.load(checkpoint_path, weights_only=False)
    cfg = ckpt["hyper_parameters"]["cfg"]
    codebook_size = cfg.get("codebook_size", 256)

    # Create dataset
    print(f"Creating dataset for {data_type}...")

    if data_type == "jet_const":
        dataset = JetConstL1TriggerDataset(
            parquet_dirs=parquet_dirs,
            max_particles=max_particles,
            features=[
                "L1T_PUPPIPart_PT", "L1T_PUPPIPart_Eta", "L1T_PUPPIPart_Phi",
                "L1T_PUPPIPart_PuppiW", "L1T_JetPuppiAK4_PT",
                "L1T_JetPuppiAK4_Eta", "L1T_JetPuppiAK4_Phi",
                "L1T_JetPuppiAK4_Mass", "L1T_JetPuppiAK4_ConstituentsIdx"
            ],
            preprocessing=True,
            shuffling=False,
            labels=True
        )
    elif data_type == "event_part":
        dataset = EventPartL1TriggerDataset(
            parquet_dirs=parquet_dirs,
            max_particles=max_particles,
            features=[
                "L1T_PUPPIPart_PT", "L1T_PUPPIPart_Eta",
                "L1T_PUPPIPart_Phi", "L1T_PUPPIPart_PuppiW"
            ],
            puppiw_threshold=0.05,
            preprocessing=True,
            shuffling=False,
            labels=True
        )
    elif data_type == "event_jets":
        dataset = EventJetsL1TriggerDataset(
            parquet_dirs=parquet_dirs,
            max_jets=16,
            features=["L1T_JetPuppiAK4_PT", "L1T_JetPuppiAK4_Eta", "L1T_JetPuppiAK4_Phi"],
            preprocessing=True,
            shuffling=False,
            labels=True
        )
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    # Generate tokens
    print("Generating tokens...")
    tokens_list, masks_list, labels_list = generate_tokens_for_dataset(
        dataset, model, data_type, max_batches=None
    )

    # Save tokens
    os.makedirs(output_dir, exist_ok=True)

    # Combine all batches
    all_tokens = torch.cat(tokens_list, dim=0)  # [total_events, max_particles/jets]
    all_masks = torch.cat(masks_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    
    # Save as PyTorch tensor
    tokens_path = os.path.join(output_dir, "tokens.pt")
    torch.save(all_tokens, tokens_path)
    print(f"Saved tokens to {tokens_path}")

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
        "data_type": data_type,
        "codebook_size": codebook_size,
        "max_particles": max_particles if data_type != "event_jets" else 16,
        "num_events": all_tokens.shape[0],
        "unique_labels": torch.unique(all_labels).tolist(),
    }
    config_path = os.path.join(output_dir, "config.pt")
    torch.save(config, config_path)
    print(f"Saved config to {config_path}")

    # Print summary
    print("\n" + "="*50)
    print("TOKEN GENERATION SUMMARY")
    print("="*50)
    print(f"Total events: {all_tokens.shape[0]}")
    print(f"Codebook size: {codebook_size}")
    print(f"Unique labels: {config['unique_labels']}")

    # Calculate codebook usage
    unique_tokens = torch.unique(all_tokens)
    usage = len(unique_tokens) / codebook_size * 100
    print(f"Unique tokens used: {len(unique_tokens)} ({usage:.2f}% of codebook)")
    print("="*50)

    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate tokens from VQ-VAE")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--parquet-dir", required=True, help="Directory with parquet files")
    parser.add_argument("--data-type", required=True, choices=["jet_const", "event_part", "event_jets"],
                        help="Data type: jet_const, event_part, or event_jets")
    parser.add_argument("--output-dir", required=True, help="Directory to save tokens")
    parser.add_argument("--model-type", default="mlp", choices=["mlp", "transformer"],
                        help="Model type: mlp or transformer")
    parser.add_argument("--max-particles", type=int, default=128,
                        help="Max particles per event (default: 128)")

    args = parser.parse_args()

    generate_and_save_tokens(
        checkpoint_path=args.checkpoint,
        parquet_dirs=args.parquet_dir,
        data_type=args.data_type,
        output_dir=args.output_dir,
        model_type=args.model_type,
        max_particles=args.max_particles
    )