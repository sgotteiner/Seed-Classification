import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class DynamicSpectralAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # squeeze spatial, project to per‑band scores
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),            # [B,C,1,1]
            nn.Flatten(1),                      # [B,C]
            nn.Linear(in_channels, in_channels),
            nn.Softmax(dim=1)                   # per‐sample band distribution
        )

    def forward(self, x):
        # x: [B, C, H, W]
        w = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * w, w                        # return weighted feature + weights


class DynamicSpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # channel‑pooled spatial map → conv → sigmoid
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, H, W]
        # channel‐wise max & avg pooling
        mx = torch.max(x, dim=1, keepdim=True)[0]
        av = torch.mean(x, dim=1, keepdim=True)
        attn = self.conv(torch.cat([mx, av], dim=1))  # [B,1,H,W]
        return x * attn, attn                         # weighted feat + map


class SpectralSpatialFusion(nn.Module):
    def __init__(self, in_channels, embed_dim=64, num_heads=4, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # project band‐pooled vector → embed
        self.spec_proj = nn.Linear(in_channels, embed_dim)

        # patchify & embed spatial maps
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

        # cross‐attention: spec queries spat
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)

        # back to channel space
        self.out_proj = nn.Linear(embed_dim, in_channels)

    def forward(self, x):
        # x: [B,C,H,W]  after both attentions
        B,C,H,W = x.shape

        # 1) Spectral tokens (one per band)
        band_vec = F.adaptive_avg_pool2d(x,1).view(B, C)        # [B,C]
        spec_tok = self.spec_proj(band_vec)                    # [B,embed]
        spec_tok = spec_tok.unsqueeze(0)                       # [1,B,embed]

        # 2) Spatial tokens (one per patch)
        p = self.patch_size
        spat = self.patch_embed(x)                             # [B,embed,H/p,W/p]
        P = (H//p)*(W//p)
        spat_tok = spat.flatten(2).permute(2,0,1)              # [P,B,embed]

        # 3) Cross‐attention
        attn_out, attn_w = self.mha(
            query=spec_tok, key=spat_tok, value=spat_tok
        )
        # attn_w: [B, heads, 1 (spec_tokens), P]

        # 4) Fuse: use attended spatial → reproject to channels
        fused = attn_out.squeeze(0)                            # [B,embed]
        ch_w = self.out_proj(fused)                            # [B,C]
        ch_w = ch_w.view(B, C, 1, 1)
        return x * ch_w, attn_w


class SpectralSpatialAttentionModel(pl.LightningModule):
    def __init__(self, in_channels, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # 1) Dynamic modules
        self.spec_attn   = DynamicSpectralAttention(in_channels)
        self.spat_attn   = DynamicSpatialAttention(in_channels)

        # 2) Simple feature extractor
        self.backbone    = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.ReLU(),
        )

        # 3) Fusion on backbone output
        self.fusion      = SpectralSpatialFusion(
            in_channels=128,
            embed_dim=64,
            num_heads=4,
            patch_size=8
        )

        # 4) Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.loss_fn = (nn.BCEWithLogitsLoss())

    def forward(self, x):
        # (1) spectral
        x, spec_w = self.spec_attn(x)
        # (2) spatial
        x, spat_map = self.spat_attn(x)
        # (3) backbone
        x = self.backbone(x)
        # (4) fusion
        x, fusion_w = self.fusion(x)
        # (5) classify
        logits = self.classifier(x)
        return logits, spec_w, spat_map, fusion_w

    def training_step(self, batch, idx):
        x,y = batch
        logits, *_ = self(x)
        loss = self.loss_fn(logits.squeeze(1), y.float())
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x,y = batch
        logits, *_ = self(x)
        loss = self.loss_fn(logits.squeeze(1), y.float())
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
