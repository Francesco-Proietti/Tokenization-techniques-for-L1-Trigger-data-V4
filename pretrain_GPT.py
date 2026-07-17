#!/usr/bin/env python3
"""Training script"""

import lightning as pl
import torch

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from gpt_like.dataset import TokenDataModule
from gpt_like.model import GPTPretrainModule


def main():

    # Set seed for reproducibility
    pl.seed_everything(56, workers=True)

    # DataModule
    data_module = TokenDataModule(
        tokens_path="gpt_like/tokens/tokens.pt",
        masks_path="gpt_like/tokens/masks.pt",
        labels_path="gpt_like/tokens/labels.pt",
        cb_size=512,
        batch_size=32,
        num_workers=0,
        test_size=0.10,
        val_size=0.20,
        seed=42,
    )

    # Model
    

    model = GPTPretrainModule(
        vocab_size=514,
        max_seq_len=129,
        pad_token=512,
        bos_token=513,
        d_model=256,
        n_layers=4,
        n_heads=8,
        dropout=0.1,
        lr=3e-4,
        weight_decay=0.01,
    )

    # Logger
    logger = TensorBoardLogger(
        save_dir="logs_GPT",
        name="GPT"
    )

    check_dir = "checkpoints_GPT"
    exp_name = "GPT"

    # Checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{check_dir}/{exp_name}",
        filename=f"v{logger.version}" + "-{epoch:02d}-{val_loss:.4f}" + f"-{exp_name}",
        monitor="val_loss",
        mode="min",
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