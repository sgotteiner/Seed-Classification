import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F


class SpectralAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpectralAttention, self).__init__()
        self.weights = nn.Parameter(torch.randn(in_channels), requires_grad=True)

    def forward(self, x):
        # x: [B, C, H, W]
        w = self.weights.view(1, -1, 1, 1)  # shape: [1, C, 1, 1]
        return x * w


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_map = self.cnn(x)  # [B, 1, H, W]
        return x * attention_map


class SpectralSpatialModuleModel(pl.LightningModule):
    def __init__(self, in_channels, num_classes=1, lr=1e-3):
        super(SpectralSpatialModuleModel, self).__init__()
        self.save_hyperparameters()

        self.spectral = SpectralAttention(in_channels)
        self.spatial = SpatialAttention(in_channels)

        self.feature_cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

        self.loss_fn = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.spectral(x)
        x = self.spatial(x)
        x = self.feature_cnn(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze(1 if self.hparams.num_classes == 1 else 0)
        loss = self.loss_fn(logits, y.float() if self.hparams.num_classes == 1 else y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze(1 if self.hparams.num_classes == 1 else 0)
        loss = self.loss_fn(logits, y.float() if self.hparams.num_classes == 1 else y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)