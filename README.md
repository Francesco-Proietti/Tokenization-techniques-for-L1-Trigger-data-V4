# Tokenization Techniques for L1 Trigger Data - VQ-VAE Implementation

This project implements Vector Quantized Variational Autoencoders (VQ-VAEs) for tokenizing and reconstructing L1 trigger data from high-energy physics experiments. The codebase provides two architecture variants—MLP-based and Transformer-based—configured via Hydra for easy experimentation.
After training the VQ-VAE, a Transformer encoder-only model is adopted for classification.

## Overview

The project uses PyTorch Lightning and aims to:
- Learn discrete token representations of L1-trigger kinematic features
- Compare MLP- and Transformer-based VQ-VAEs
- Evaluate reconstruction quality using VQ-VAE with Rotation Trick or Straight-Through Estimation (STE)
- Use the learned token with a Transformer encoder-only classifier

## Project Structure

### Main Files

| File | Description |
|------|-------------|
| `train_VQ_VAE.py` | Training script that orchestrates data loading, model initialization, and PyTorch Lightning training |
| `environment.yaml` | Conda environment specification with all dependencies |
| `tokenizer.py` | Tokenization script that, given a lightning checkpoint, produces the tokens |

### Configuration Files (`configs/`)

The configuration system uses Hydra's compositional config pattern:

| File | Description |
|------|-------------|
| `configs/config.yaml` | Main config file combining model, data, and trainer configs |
| `configs/model/mlp_vqvae.yaml` | MLP VQ-VAE hyperparameters (hidden dims, latent dim, codebook size) |
| `configs/model/transformer_vqvae.yaml` | Transformer VQ-VAE hyperparameters (n_heads, depth, dropout) |
| `configs/data/event_jets_data_loading.yaml` | Data paths, features, max particles, preprocessing settings of the jet-level view |
| `configs/data/event_particles_data_loading.yaml` | Data paths, features, max particles, preprocessing settings of the particle-level view |
| `configs/data/event_jet_constituents_loading.yaml` | Data paths, features, max particles, preprocessing settings of the jet-constituents-level view |
| `configs/trainer/default.yaml` | Training hyperparameters (epochs, batch size, learning rate, seed) |

### Source Code (`src/`)

#### Data Loading (`src/data/`)

| File | Description |
|------|-------------|
| `src/data/event_jets_data_loading.py` | `L1TriggerDataset` (IterableDataset for streaming parquet files) and `L1TriggerDataModule` (LightningDataModule for train/val/test splits) |
| `src/data/event_particles_data_loading.py` | `L1TriggerDataset` (IterableDataset for streaming parquet files) and `L1TriggerDataModule` (LightningDataModule for train/val/test splits) |
| `src/data/jet_constituents_data_loading.py` | `L1TriggerDataset` (IterableDataset for streaming parquet files) and `L1TriggerDataModule` (LightningDataModule for train/val/test splits) |

#### Models (`src/models/`)

| File | Description |
|------|-------------|
| `src/models/registry.py` | Model registry mapping names ("mlp", "transformer") to classes |
| `src/models/mlp_vqvae.py` | MLP-based VQ-VAE |
| `src/models/transformer_vqvae.py` | Transformer-based VQ-VAE |

## Data Format

### Event particles

The dataset expects Parquet files containing L1T PUPPI particles with features:
- `L1T_PUPPIPart_PT`: Transverse momentum
- `L1T_PUPPIPart_Eta`: Pseudorapidity
- `L1T_PUPPIPart_Phi`: Azimuthal angle
- `L1T_PUPPIPart_PuppiW`: PUPPI weight (used for filtering by threshold)

Particles are padded to `max_particles` (default: 128) with a mask indicating valid entries.

### Event jets

The dataset expects Parquet files containing L1T PUPPI jets with features:
- `L1T_JetPuppiAK4_PT`: Transverse momentum
- `L1T_JetPuppiAK4_Eta`: Psudorapidity
- `L1T_JetPuppiAK4_Phi`: Azimuthal angle

Jets are padded to `max_jets` (default:14) with a mask indicating valid entries.

### Jet constituents

The dataset expects Parquet files containing L1T PUPPI particles and jets with features:
- `L1T_PUPPIPart_PT`: Particles transverse momentum
- `L1T_PUPPIPart_Eta`: Particles pseudorapidity
- `L1T_PUPPIPart_Phi`: Particles azimuthal angle
- `L1T_PUPPIPart_PuppiW`: Particles PUPPI weight (used for filtering by threshold)
- `L1T_JetPuppiAK4_PT`: Jets transverse momentum
- `L1T_JetPuppiAK4_Eta`: Jets pseudorapidity
- `L1T_JetPuppiAK4_Phi`: Jets azimuthal angle
- `L1T_JetPuppiAK4_ConstituentsIdx`: Indices of the constituents position

Constituents are padded to `max_particles` (default:64) with a mask indicating valid entries.

## Usage

```bash
# Train with default config
python train_VQ_VAE.py

# Change data paths in configs/data/default.yaml before running
```

```bash
# Command example for running the tokenizer
python tokenizer.py --checkpoint checkpoints/transformer-event_jets-True-1024/v0-epoch=09-val_loss=0.0454-transformer-event_jets-True-1024.ckpt --parquet-dir /path/to/data/ --data-type event_jets --output-dir classifier/tokens --model-type transformer --max-particles 14
```

## Dependencies

Install via: `conda env create -f environment.yaml`
