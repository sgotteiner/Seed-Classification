import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class BandAttentionModel(pl.LightningModule):
    def __init__(self, in_channels, embed_dim=64, num_heads=4, num_classes=1, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.embedder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, embed_dim, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # [B, D, 1, 1]
            nn.Flatten(start_dim=1)   # [B, D]
        )

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        if num_classes == 1:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def get_sinusoidal_pe(self, length, dim, device):
        pe = torch.zeros(length, 1, dim, device=device)
        position = torch.arange(0, length, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):  # x: [B, C, H, W]
        B, C, H, W = x.shape
        tokens = []

        for i in range(C):
            band = x[:, i:i+1, :, :]        # [B, 1, H, W]
            tok = self.embedder(band)       # [B, D]
            tokens.append(tok)

        band_tokens = torch.stack(tokens, dim=0)  # [C, B, D]

        # Generate sinusoidal positional encoding
        pe = self.get_sinusoidal_pe(C, band_tokens.size(-1), band_tokens.device)  # [C, 1, D]

        # Calculate embedding range per batch and dim
        emb_min, _ = band_tokens.min(dim=0, keepdim=True)  # [1, B, D]
        emb_max, _ = band_tokens.max(dim=0, keepdim=True)  # [1, B, D]
        emb_range = emb_max - emb_min  # [1, B, D]

        # Compute scale scalar (mean over batch and embedding dim)
        scale = emb_range.mean(dim=(1, 2), keepdim=True)  # [1, 1, 1]

        # Scale positional encoding accordingly
        scaled_pe = pe * scale.squeeze()  # scalar scale broadcasted

        # Add scaled PE to embeddings
        band_tokens = band_tokens + scaled_pe

        attn_out, attn_weights = self.attn(band_tokens, band_tokens, band_tokens)
        fused = attn_out.mean(dim=0)               # [B, D]
        logits = self.classifier(fused)            # [B, num_classes]

        return logits, attn_weights

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, _ = self(x)

        if self.hparams.num_classes == 1:
            loss = self.loss_fn(logits.squeeze(1), y.float())
        else:
            loss = self.loss_fn(logits, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, _ = self(x)

        if self.hparams.num_classes == 1:
            loss = self.loss_fn(logits.squeeze(1), y.float())
        else:
            loss = self.loss_fn(logits, y)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        # Optional accuracy metric
        preds = (torch.sigmoid(logits) > 0.5).int() if self.hparams.num_classes == 1 else logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)