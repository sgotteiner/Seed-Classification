#!/data/bin/miniconda2/envs/seed-v1.0/bin/python
# coding: utf-8

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
            nn.Sigmoid()
        )
        self.loss_fn = nn.BCELoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        x, y = batch
        
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        
        logits = self(x).squeeze()
        loss = self.loss_fn(logits, y)
        preds = (logits > 0.5).float()
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
