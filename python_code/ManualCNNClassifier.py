#!/data/bin/miniconda2/envs/seed-v1.0/bin/python
# coding: utf-8

import copy
import torch.nn as nn
import pytorch_lightning as pl
import torch


class HyperspectralCNN(pl.LightningModule):
    def __init__(self, num_channels, lr=1e-3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        x, y = batch
        
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        
        logits = self(x).squeeze()
        loss = self.loss_fn(logits, y)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        acc = (preds == y).float().mean()
        
        self.log(f'{stage}_loss', loss, prog_bar=True)
        self.log(f'{stage}_acc', acc, prog_bar=True)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    

class HyperspectralMultiCNN(pl.LightningModule):
    class SingleBandCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()  # [B, 128]
            )

        def forward(self, x):  # x: [B, 1, H, W]
            return self.net(x)
    
    def __init__(self, num_bands, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.num_bands = num_bands

        # One CNN per band
        base_cnn = self.SingleBandCNN()
        self.band_cnns = nn.ModuleList([
            copy.deepcopy(base_cnn) for _ in range(num_bands)
        ])

        # Classifier takes concatenated features
        self.classifier = nn.Sequential(
            nn.Linear(128 * num_bands, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):  # x: [B, C=num_bands, H, W]
        features = []
        for i in range(self.num_bands):
            band = x[:, i:i+1, :, :]                # [B, 1, H, W]
            feat = self.band_cnns[i](band)          # [B, 128]
            features.append(feat)

        fused = torch.cat(features, dim=1)          # [B, 128 * num_bands]
        return self.classifier(fused)               # [B, 1]

    def _shared_step(self, batch, stage):
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        logits = self(x).squeeze()
        loss = self.loss_fn(logits, y)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        acc = (preds == y).float().mean()

        self.log(f'{stage}_loss', loss, prog_bar=True)
        self.log(f'{stage}_acc', acc, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
