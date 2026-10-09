#!/usr/bin/env python3
"""Training script"""

import lightning as pl
import torch

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from classifier_baseline.dataset_base import OriginalDataModule
from classifier_baseline.model_baseline import EncoderClassificationModule


def main():

    # Set seed for reproducibility
    pl.seed_everything(15, workers=True)

    # DataModule
    data_module = OriginalDataModule(
        data_path="classifier_baseline/data/data.pt",
        masks_path="classifier_baseline/data/masks.pt",
        labels_path="classifier_baseline/data/labels.pt",
        batch_size=64,
        num_workers=0,
        test_size=0.20,
        seed=15,
    )

    # Model
    model = EncoderClassificationModule(
        max_seq_len=14,
        cls_token=0,
        d_model=256,
        n_layers=4,
        n_heads=8,
        dropout=0.1,
        lr=3e-4,
        weight_decay=0.01,
    )

    # Logger
    logger = TensorBoardLogger(
        save_dir="logs_base_classifier",
        name="base_class"
    )

    check_dir = "checkpoints_base_classifier"
    exp_name = "base_class"

    # Checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{check_dir}/{exp_name}",
        filename=f"v{logger.version}" + "-{epoch:02d}-{train_acc:.4f}" + f"-{exp_name}",
        monitor="train_acc",
        mode="max",
        save_top_k=3,
        save_last=True
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
        logger=logger,
        callbacks=checkpoint_callback
    )
    
    # Training
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()