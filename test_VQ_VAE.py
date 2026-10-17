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
    #pl.seed_everything(cfg.trainer.seed, workers=True)

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

    #checkpoint_path = "checkpoints/mlp-event_jets-True-2048/v2-epoch=07-val_loss=0.0498-mlp-event_jets-True-2048.ckpt"
    #checkpoint_path = "checkpoints/mlp-event_part-True-2048/v2-epoch=00-val_loss=0.0271-mlp-event_part-True-2048.ckpt"
    #checkpoint_path = "checkpoints/mlp-jet_const-True-2048/v3-epoch=06-val_loss=0.0107-mlp-jet_const-True-2048.ckpt"
    #checkpoint_path = "checkpoints/transformer-event_jets-True-2048/v2-epoch=08-val_loss=0.0455-transformer-event_jets-True-2048.ckpt"
    #checkpoint_path = "checkpoints/transformer-event_part-True-2048/v2-epoch=07-val_loss=0.0206-transformer-event_part-True-2048.ckpt"
    checkpoint_path = "checkpoints/transformer-jet_const-True-2048/v3-epoch=03-val_loss=0.0098-transformer-jet_const-True-2048.ckpt"
    
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

