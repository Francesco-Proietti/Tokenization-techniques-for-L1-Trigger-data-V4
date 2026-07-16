# Tokenization Techniques for L1 Trigger Data - VQ-VAE Implementation

This project implements Vector Quantized Variational Autoencoders (VQ-VAEs) for tokenizing and reconstructing L1 trigger data from high-energy physics experiments. The codebase provides two architecture variants—MLP-based and Transformer-based—configured via Hydra for easy experimentation.

## Overview

The project uses PyTorch Lightning and aims to:
- Learn discrete token representations of particle flow data
- Compare MLP and Transformer encoder/decoder architectures
- Evaluate reconstruction quality using VQ-VAE with rotation trick or Straight-Through Estimation (STE)

## Project Structure

### Main Files

| File | Description |
|------|-------------|
| `train.py` | Training script that orchestrates data loading, model initialization, and PyTorch Lightning training |
| `environment.yaml` | Conda environment specification with all dependencies |

### Configuration Files (`configs/`)

The configuration system uses Hydra's compositional config pattern:

| File | Description |
|------|-------------|
| `configs/config.yaml` | Main config file combining model, data, and trainer configs |
| `configs/model/mlp_vqvae.yaml` | MLP VQ-VAE hyperparameters (hidden dims, latent dim, codebook size) |
| `configs/model/transformer_vqvae.yaml` | Transformer VQ-VAE hyperparameters (n_heads, depth, dropout) |
| `configs/data/default.yaml` | Data paths, features, max particles, preprocessing settings |
| `configs/trainer/default.yaml` | Training hyperparameters (epochs, batch size, learning rate, seed) |

### Source Code (`src/`)

#### Data Loading (`src/data/`)

| File | Description |
|------|-------------|
| `src/data/data_loading.py` | `L1TriggerDataset` (IterableDataset for streaming parquet files) and `L1TriggerDataModule` (LightningDataModule for train/val/test splits) |

#### Models (`src/models/`)

| File | Description |
|------|-------------|
| `src/models/registry.py` | Model registry mapping names ("mlp", "transformer") to classes |
| `src/models/mlp_vqvae.py` | MLP-based VQ-VAE with encoder/decoder and VectorQuantize layer |
| `src/models/transformer_vqvae.py` | Transformer-based VQ-VAE using NormFormer blocks |

## Data Format

The dataset expects Parquet files containing L1T PUPPI particles with features:
- `L1T_PUPPIPart_PT`: Transverse momentum
- `L1T_PUPPIPart_Eta`: Pseudorapidity
- `L1T_PUPPIPart_Phi`: Azimuthal angle
- `L1T_PUPPIPart_PuppiW`: PUPPI weight (used for filtering by threshold)

Particles are padded to `max_particles` (default: 128) with a mask indicating valid entries.

## Usage

```bash
# Train with default config
python train.py

# Change data paths in configs/data/default.yaml before running
```

## Dependencies

Install via: `conda env create -f environment.yaml`
