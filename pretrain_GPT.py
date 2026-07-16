#!/usr/bin/env python3
"""Training script"""

import lightning as pl
import torch

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger


def main():

    # Set seed for reproducibility
    pl.seed_everything(56, workers=True)

    # DataModule
    

    # Model
    model_name = cfg.model.name
    ModelClass = MODEL_REGISTRY[model_name]

    model = ModelClass(cfg.model, lr=cfg.trainer.lr)

    # Logger
    logger = TensorBoardLogger(
        save_dir=cfg.paths.logs_dir,
        name=cfg.experiment.name
    )

    # Checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{cfg.paths.checkpoint_dir}/{cfg.experiment.name}",
        filename=f"v{logger.version}" + "-{epoch:02d}-{val_loss:.4f}" + f"-{cfg.experiment.name}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True
    )

    # Add histogram plotter callback
    histogram_callback = HistogramPlotter(
        data_loading=cfg.data.name,
        cb_size=cfg.model.codebook_size,
        model_name=cfg.model.name,
        rotation=cfg.model.rotation_trick,
        output_dir=f"{cfg.paths.logs_dir}/{cfg.experiment.name}/version_{logger.version}/validation_plots",
        log_every_n_epochs=1,  # Plot every epoch
        #max_samples=1000  # Optional: limit for memory
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        logger=logger,
        callbacks=[checkpoint_callback, histogram_callback]
    )
    
    # Training
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()