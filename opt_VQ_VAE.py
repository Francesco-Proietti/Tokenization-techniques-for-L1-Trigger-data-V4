#!/usr/bin/env python3
"""Hyperparameter optimization with Optuna."""

import copy
import hydra
import optuna
import lightning as pl

from omegaconf import DictConfig

from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping
)

from optuna.integration import PyTorchLightningPruningCallback

from src.models.model_registry import MODEL_REGISTRY
from src.data.data_registry import DATA_REGISTRY


# Objective function
def objective(trial, cfg):

    cfg = copy.deepcopy(cfg)
    
    # Learning Rate
    cfg.trainer.lr = trial.suggest_float(
        "lr",
        1e-5,
        1e-2,
        log=True
    )
    
    # Batch size
    cfg.trainer.batch_size = trial.suggest_categorical(
        "batch_size",
        [32,64,128]
    )
    
    # Latent dimension
    cfg.model.latent_dim = trial.suggest_categorical(
        "latent_dim",
        [4, 8, 16]
    )
    
    # EMA decay
    cfg.model.decay = trial.suggest_float(
        "decay",
        0.70,
        0.9999
    )
    
    # Beta (commitment weight)
    cfg.model.beta = trial.suggest_float(
        "beta",
        0.60,
        0.95
    )


    # Seed
    pl.seed_everything(cfg.trainer.seed, workers=True)


    # DataModule
    DataModuleClass = DATA_REGISTRY[cfg.data.name]

    data_module = DataModuleClass(
        cfg.data,
        batch_size=cfg.trainer.batch_size
    )


    # Model
    ModelClass = MODEL_REGISTRY[cfg.model.name]

    model = ModelClass(
        cfg.model,
        lr=cfg.trainer.lr
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"optuna_checkpoints/trial_{trial.number}",
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=10,
        verbose=False
    )

    pruning_callback = PyTorchLightningPruningCallback(
        trial,
        monitor="val_loss"
    )


    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=False,
        callbacks=[
            checkpoint_callback,
            early_stopping,
            pruning_callback
        ]
    )


    # Training
    trainer.fit(model, datamodule=data_module)


    return checkpoint_callback.best_model_score.item()


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="config"
)
def main(cfg: DictConfig):

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=5
        )
    )

    study.optimize(
        lambda trial: objective(trial, cfg),
        n_trials=50
    )

    print("\n==============================")
    print("Best trial")
    print("==============================")

    print(f"Best validation Loss: {study.best_value:.6f}")

    print("\nBest hyperparameters:")

    for key, value in study.best_params.items():
        print(f"{key}: {value}")

    print(f"\nBest trial number: {study.best_trial.number}")

    print(
        f"\nBest checkpoint: "
        f"optuna_checkpoints/trial_{study.best_trial.number}/best.ckpt"
    )
    

if __name__ == "__main__":
    main()