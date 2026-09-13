#!/usr/bin/env python3
"""Testing script"""

import hydra
from omegaconf import DictConfig

import lightning as pl
import torch
from lightning.pytorch.loggers import TensorBoardLogger

# Import model and data registries
from src.models.model_registry import MODEL_REGISTRY
from src.data.data_registry import DATA_REGISTRY

from src.callbacks.histogram_plotter import HistogramPlotter, HistogramTestPlotter

@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="config"
)
def main(cfg: DictConfig):

    # Set seed for reproducibility
    pl.seed_everything(cfg.trainer.seed, workers=True)

    # ---------------------------------------------------------
    # DataModule
    # ---------------------------------------------------------
    data_name = cfg.data.name
    DataModuleClass = DATA_REGISTRY[data_name]

    data_module = DataModuleClass(
        cfg.data,
        batch_size=cfg.trainer.batch_size
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    #checkpoint_path = "checkpoints/mlp-event_jets-True-1024/v0-epoch=09-val_loss=0.0454-transformer-event_jets-True-1024.ckpt"
    #checkpoint_path = "checkpoints/mlp-event_part-True-1024/v0-epoch=09-val_loss=0.0454-transformer-event_jets-True-1024.ckpt"
    #checkpoint_path = "checkpoints/mlp-jet_const-True-1024/v0-epoch=09-val_loss=0.0454-transformer-event_jets-True-1024.ckpt"
    checkpoint_path = "checkpoints/transformer-event_jets-True-1024/v0-epoch=09-val_loss=0.0454-transformer-event_jets-True-1024.ckpt"
    #checkpoint_path = "checkpoints/transformer-event_part-True-1024/v0-epoch=09-val_loss=0.0454-transformer-event_jets-True-1024.ckpt"
    #checkpoint_path = "checkpoints/transformer-jet_const-True-1024/v0-epoch=09-val_loss=0.0454-transformer-event_jets-True-1024.ckpt"
    
    model_name = cfg.model.name
    ModelClass = MODEL_REGISTRY[model_name]

    model = ModelClass.load_from_checkpoint(checkpoint_path, weights_only=False)

    # ---------------------------------------------------------
    # Logger
    # ---------------------------------------------------------
    logger = TensorBoardLogger(
        save_dir=cfg.paths.logs_dir + "test",
        name=cfg.experiment.name + "test"
    )

    histogram_callback = HistogramTestPlotter(
        data_loading=cfg.data.name,
        cb_size=cfg.model.codebook_size,
        model_name=cfg.model.name,
        rotation=cfg.model.rotation_trick,
        output_dir=(
            f"{cfg.paths.logs_dir + "test"}/"
            f"{cfg.experiment.name + "test"}/"
            f"test_plots"
        ),
        max_samples=None,
    )

    # ---------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------
    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=logger,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        callbacks=[histogram_callback]
    )

    # ---------------------------------------------------------
    # Test
    # ---------------------------------------------------------
    
    print(f"Loading checkpoint: {checkpoint_path}")

    trainer.test(
        model=model,
        datamodule=data_module,
    )

    print("\nTest results:")


if __name__ == "__main__":
    main()

