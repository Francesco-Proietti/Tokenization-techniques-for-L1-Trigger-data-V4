import torch
import torch.nn as nn
import torchmetrics
import lightning as pl


# GPT block---------------------------
class GPTBlock(nn.Module):
    """
    Decoder-only Transformer block (Pre-LayerNorm).

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

        self.ln1 = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout1 = nn.Dropout(dropout)

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
        attn_mask=None,
        key_padding_mask=None,
    ):
        """
        Parameters
        ----------
        x : (B, L, D)

        attn_mask :
            causal mask (L,L)

        key_padding_mask :
            (B,L)
            True = PAD
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
            attn_mask=attn_mask,
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


#GPT backbone--------------------------
class GPTBackbone(nn.Module):
    """
    Decoder-only Transformer backbone.

    Output:
        hidden states (B, L, D)

    no final head
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # -----------------------------
        # Embeddings
        # -----------------------------

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
        )

        self.position_embedding = nn.Embedding(
            max_seq_len,
            d_model,
        )

        self.dropout = nn.Dropout(dropout)

        # -----------------------------
        # Transformer blocks
        # -----------------------------

        self.blocks = nn.ModuleList(
            [
                GPTBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        # -----------------------------

        self.norm = nn.LayerNorm(d_model)

    def _build_causal_mask(
        self,
        seq_len,
        device,
    ):

        return torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

    def forward(
        self,
        tokens,
        attention_mask=None,
    ):
        """
        Parameters
        ----------

        tokens:
            (B,L)

        attention_mask:
            (B,L)
        """

        B, L = tokens.shape

        device = tokens.device

        # -----------------------------------

        positions = torch.arange(
            L,
            device=device,
        ).unsqueeze(0)

        x = (
            self.token_embedding(tokens)
            +
            self.position_embedding(positions)
        )

        x = self.dropout(x)

        # -----------------------------------

        causal_mask = self._build_causal_mask(
            L,
            device,
        )

        if attention_mask is None:

            key_padding_mask = None

        else:

            key_padding_mask = ~attention_mask

        # -----------------------------------

        for block in self.blocks:

            x = block(
                x,
                attn_mask=causal_mask,
                key_padding_mask=key_padding_mask,
            )

        x = self.norm(x)

        return x


#GPT for pretraining--------------------------
class GPTForPretraining(nn.Module):

    def __init__(
        self,
        vocab_size,
        max_seq_len,
        d_model=256,
        n_layers=4,
        n_heads=8,
        mlp_ratio=4,
        dropout=0.1,
    ):
        super().__init__()

        self.backbone = GPTBackbone(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False,
        )

        # Weight tying (GPT-2)
        self.lm_head.weight = self.backbone.token_embedding.weight

    def forward(
        self,
        tokens,
        attention_mask=None,
    ):

        hidden = self.backbone(
            tokens,
            attention_mask,
        )

        logits = self.lm_head(hidden)

        return logits
    

#GPT for classification------------------------
class GPTForClassification(nn.Module):

    def __init__(
        self,
        vocab_size,
        max_seq_len,
        num_classes=1,
        d_model=256,
        n_layers=4,
        n_heads=8,
        mlp_ratio=4,
        dropout=0.1,
    ):
        super().__init__()

        self.backbone = GPTBackbone(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(

            nn.Linear(d_model, d_model),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(d_model, num_classes)
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

        cls = hidden[:, 0]

        logits = self.classifier(cls)

        return logits.squeeze(-1)


#GPT for pretraining lightining
class GPTPretrainModule(pl.LightningModule):

    def __init__(
        self,
        vocab_size,
        max_seq_len,
        pad_token,
        bos_token,
        d_model=256,
        n_layers=4,
        n_heads=8,
        dropout=0.1,
        lr=3e-4,
        weight_decay=0.01,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = GPTForPretraining(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
        )

        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=pad_token
        )

    def forward(
        self,
        tokens,
        padding_mask,
    ):

        return self.model(
            tokens,
            padding_mask,
        )
    
    def _prepare_batch(self, batch):

        tokens = batch["tokens"]
        mask = batch["mask"]

        B = tokens.size(0)

        bos = torch.full(
            (B, 1),
            self.hparams.bos_token,
            dtype=tokens.dtype,
            device=tokens.device,
        )

        input_tokens = torch.cat(
            [bos, tokens],
            dim=1,
        )

        target_tokens = input_tokens[:, 1:]
        input_tokens = input_tokens[:, :-1]

        bos_mask = torch.ones(
            (B, 1),
            dtype=torch.bool,
            device=mask.device,
        )

        input_mask = torch.cat(
            [bos_mask, mask],
            dim=1,
        )

        input_mask = input_mask[:, :-1]

        return input_tokens, target_tokens, input_mask
    
    def training_step(
        self,
        batch,
        batch_idx,
    ):

        x, y, mask = self._prepare_batch(batch)

        logits = self(
            x,
            mask,
        )

        loss = self.loss_fn(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=x.size(0),
        )

        return loss
    
    def validation_step(
        self,
        batch,
        batch_idx,
    ):

        x, y, mask = self._prepare_batch(batch)

        logits = self(
            x,
            mask,
        )

        loss = self.loss_fn(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            batch_size=x.size(0),
        )

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        return optimizer


# GPT classifier lightning
class GPTClassificationModule(pl.LightningModule):

    def __init__(
        self,
        vocab_size,
        max_seq_len,
        pad_token,
        bos_token,
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

        self.model = GPTForClassification(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.train_acc = torchmetrics.classification.BinaryAccuracy()
        self.val_acc = torchmetrics.classification.BinaryAccuracy()
        self.test_acc = torchmetrics.classification.BinaryAccuracy()

    def forward(
        self,
        tokens,
        mask,
    ):
        return self.model(tokens, mask)
    
    def _prepend_bos(self, tokens, attention_mask):
        
        B = tokens.size(0)

        bos = torch.full(
            (B, 1),
            self.hparams.bos_token,
            device=tokens.device,
            dtype=tokens.dtype,
        )

        bos_mask = torch.ones(
            (B, 1),
            device=attention_mask.device,
            dtype=torch.bool,
        )

        tokens = torch.cat(
            [bos, tokens],
            dim=1,
        )

        attention_mask = torch.cat(
            [bos_mask, attention_mask],
            dim=1,
        )

        return tokens, attention_mask
    
    def training_step(
        self,
        batch,
        batch_idx,
    ):

        tokens = batch["tokens"]
        mask = batch["mask"]

        tokens, mask = self._prepend_bos(
            tokens,
            mask,
        )

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

        self.train_acc(
            preds,
            labels.int(),
        )

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

        self.log(
            "train_acc",
            self.train_acc,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

        return loss
    
    def validation_step(
        self,
        batch,
        batch_idx,
    ):

        tokens = batch["tokens"]
        mask = batch["mask"]

        tokens, mask = self._prepend_bos(
            tokens,
            mask,
        )

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
        )

        self.log(
            "val_acc",
            self.val_acc,
            prog_bar=True,
            on_epoch=True,
        )

    def test_step(
        self,
        batch,
        batch_idx,
    ):

        tokens = batch["tokens"]
        mask = batch["mask"]

        tokens, mask = self._prepend_bos(
            tokens,
            mask,
        )

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
            on_epoch=True,
        )

        self.log(
            "test_acc",
            self.test_acc,
            on_epoch=True,
        )

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        return optimizer