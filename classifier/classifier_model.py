import torch
import torch.nn as nn
import torchmetrics
import lightning as pl


# ============================================================
# Transformer Encoder Block
# ============================================================

class EncoderBlock(nn.Module):
    """
    Transformer Encoder block (Pre-LayerNorm).

    Structure:
        x = x + SelfAttention(LN(x))
        x = x + MLP(LN(x))
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # -----------------------
        # Self Attention
        # -----------------------

        self.ln1 = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout1 = nn.Dropout(dropout)

        # -----------------------
        # Feed Forward
        # -----------------------

        self.ln2 = nn.LayerNorm(d_model)

        hidden_dim = mlp_ratio * d_model

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x,
        key_padding_mask=None,
    ):
        """
        Parameters
        ----------
        x : (B, L, D)

        key_padding_mask : (B, L)
            True  = PAD
            False = token valido
        """

        # -----------------------
        # Self Attention
        # -----------------------

        h = self.ln1(x)

        h, _ = self.attn(
            query=h,
            key=h,
            value=h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        x = x + self.dropout1(h)

        # -----------------------
        # Feed Forward
        # -----------------------

        h = self.ln2(x)

        h = self.mlp(h)

        x = x + h

        return x


# ============================================================
# Transformer Encoder Backbone
# ============================================================

class EncoderBackbone(nn.Module):
    """
    Encoder-only Transformer backbone.

    Input:
        tokens: (B, L)

    Output:
        hidden states: (B, L+1, D)

    The +1 comes from the [CLS] token.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        cls_token: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.cls_token = cls_token

        # -----------------------------
        # Token embedding
        # -----------------------------

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
        )

        # -----------------------------
        # Positional embedding
        # -----------------------------

        # +1 because of [CLS]
        self.position_embedding = nn.Embedding(
            max_seq_len + 1,
            d_model,
        )

        self.dropout = nn.Dropout(dropout)

        # -----------------------------
        # CLS embedding
        # -----------------------------

        self.cls_embedding = nn.Parameter(
            torch.zeros(1, 1, d_model)
        )

        nn.init.normal_(
            self.cls_embedding,
            mean=0.0,
            std=0.02,
        )

        # -----------------------------
        # Transformer Encoder blocks
        # -----------------------------

        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        # -----------------------------
        # Final normalization
        # -----------------------------

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tokens,
        attention_mask=None,
    ):
        """
        Parameters
        ----------
        tokens:
            (B, L)

        attention_mask:
            (B, L)

            True  = token valido
            False = PAD

        Returns
        -------
        hidden:
            (B, L+1, D)
        """

        B, L = tokens.shape

        device = tokens.device

        if L > self.max_seq_len:
            raise ValueError(
                f"Sequence length {L} exceeds "
                f"max_seq_len={self.max_seq_len}"
            )

        # -----------------------------------
        # Token embeddings
        # -----------------------------------

        x = self.token_embedding(tokens)

        # -----------------------------------
        # Add CLS token
        # -----------------------------------

        cls = self.cls_embedding.expand(B, -1, -1)

        x = torch.cat(
            [cls, x],
            dim=1,
        )

        # -----------------------------------
        # Positional embeddings
        # -----------------------------------

        positions = torch.arange(
            L + 1,
            device=device,
        ).unsqueeze(0)

        x = x + self.position_embedding(
            positions
        )

        x = self.dropout(x)

        # -----------------------------------
        # Attention mask
        # -----------------------------------

        if attention_mask is None:

            key_padding_mask = None

        else:

            # CLS is always valid
            cls_mask = torch.ones(
                (B, 1),
                dtype=torch.bool,
                device=device,
            )

            attention_mask = torch.cat(
                [
                    cls_mask,
                    attention_mask,
                ],
                dim=1,
            )

            # MultiheadAttention:
            # True = ignore / padding
            # False = valid
            key_padding_mask = ~attention_mask

        # -----------------------------------
        # Transformer Encoder
        # -----------------------------------

        for block in self.blocks:

            x = block(
                x,
                key_padding_mask=key_padding_mask,
            )

        # -----------------------------------
        # Final normalization
        # -----------------------------------

        x = self.norm(x)

        return x


# ============================================================
# Encoder for Binary Classification
# ============================================================

class EncoderForClassification(nn.Module):
    """
    Encoder-only Transformer for binary classification.

    Input:
        tokens: (B, L)

    Output:
        logits: (B,)
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        cls_token: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.backbone = EncoderBackbone(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            cls_token=cls_token,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # -----------------------------------
        # Classification head
        # -----------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        tokens,
        attention_mask=None,
    ):

        hidden = self.backbone(
            tokens,
            attention_mask,
        )

        # -----------------------------------
        # CLS representation
        # -----------------------------------

        cls = hidden[:, 0]

        # -----------------------------------
        # Binary classification
        # -----------------------------------

        logits = self.classifier(cls)

        return logits.squeeze(-1)


# ============================================================
# Lightning Module
# ============================================================

class EncoderClassificationModule(pl.LightningModule):

    def __init__(
        self,
        vocab_size,
        max_seq_len,
        cls_token,
        d_model=256,
        n_layers=4,
        n_heads=8,
        mlp_ratio=4,
        dropout=0.1,
        lr=3e-4,
        weight_decay=1e-2,
    ):
        super().__init__()

        self.save_hyperparameters()

        # -----------------------------------
        # Model
        # -----------------------------------

        self.model = EncoderForClassification(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            cls_token=cls_token,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # -----------------------------------
        # Loss
        # -----------------------------------

        self.loss_fn = nn.BCEWithLogitsLoss()

        # -----------------------------------
        # Metrics
        # -----------------------------------

        self.train_acc = torchmetrics.classification.BinaryAccuracy()
        self.val_acc = torchmetrics.classification.BinaryAccuracy()
        self.test_acc = torchmetrics.classification.BinaryAccuracy()

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        tokens,
        mask,
    ):

        return self.model(
            tokens,
            mask,
        )

    # ========================================================
    # Training
    # ========================================================

    def training_step(
        self,
        batch,
        batch_idx,
    ):

        tokens = batch["tokens"]
        mask = batch["mask"]

        labels = batch["label"].float()

        logits = self(
            tokens,
            mask,
        )

        loss = self.loss_fn(
            logits,
            labels,
        )

        # Predictions
        preds = torch.sigmoid(logits)

        self.train_acc(
            preds,
            labels.int(),
        )

        # Logging
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=tokens.size(0),
        )

        self.log(
            "train_acc",
            self.train_acc,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=tokens.size(0),
        )

        return loss

    # ========================================================
    # Validation
    # ========================================================

    def validation_step(
        self,
        batch,
        batch_idx,
    ):

        tokens = batch["tokens"]
        mask = batch["mask"]

        labels = batch["label"].float()

        logits = self(
            tokens,
            mask,
        )

        loss = self.loss_fn(
            logits,
            labels,
        )

        preds = torch.sigmoid(logits)

        self.val_acc(
            preds,
            labels.int(),
        )

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            batch_size=tokens.size(0),
        )

        self.log(
            "val_acc",
            self.val_acc,
            prog_bar=True,
            on_epoch=True,
            batch_size=tokens.size(0),
        )

    # ========================================================
    # Test
    # ========================================================

    def test_step(
        self,
        batch,
        batch_idx,
    ):

        tokens = batch["tokens"]
        mask = batch["mask"]

        labels = batch["label"].float()

        logits = self(
            tokens,
            mask,
        )

        loss = self.loss_fn(
            logits,
            labels,
        )

        preds = torch.sigmoid(logits)

        self.test_acc(
            preds,
            labels.int(),
        )

        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            batch_size=tokens.size(0),
        )

        self.log(
            "test_acc",
            self.test_acc,
            prog_bar=True,
            on_epoch=True,
            batch_size=tokens.size(0),
        )

    # ========================================================
    # Optimizer
    # ========================================================

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        return optimizer
