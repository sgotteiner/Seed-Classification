import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class DynamicSpectralAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(in_channels, in_channels // 2), nn.ReLU(),
            nn.Linear(in_channels // 2, in_channels),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        w = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * w, w


class DynamicSpatialSingleHeatmapAttention(nn.Module):
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


class SpectralSpatialSingleHeatmapFusion(nn.Module):
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


class SpectralSpatialSingleHeatmapAttentionModel(pl.LightningModule):
    def __init__(self, in_channels, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # 1) Dynamic modules
        self.spec_attn   = DynamicSpectralAttention(in_channels)
        self.spat_attn   = DynamicSpatialSingleHeatmapAttention(in_channels)

        # 2) Simple feature extractor
        self.backbone    = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.ReLU(),
        )

        # 3) Fusion on backbone output
        self.fusion      = SpectralSpatialSingleHeatmapFusion(
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
    


#################################################################################################################################



# 1) Dynamic spectral attention: per-sample normalized band weights


# 2) Per-band spatial attention: generates distinct heatmap per channel
class DynamicSpatialMultiHeatmapAttention(nn.Module):
    class SingleBandAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 1, kernel_size=1),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.net(x)

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        base_attn = self.SingleBandAttention()
        self.band_attns = nn.ModuleList([copy.deepcopy(base_attn) for _ in range(in_channels)])

    def forward(self, x):
        # x shape: [B, C=in_channels, H, W]
        attn_maps = []
        weighted_bands = []
        for i in range(self.in_channels):
            band = x[:, i:i+1, :, :]              # [B,1,H,W]
            attn_map = self.band_attns[i](band)  # [B,1,H,W]
            weighted_band = band * attn_map
            attn_maps.append(attn_map)
            weighted_bands.append(weighted_band)

        attn = torch.cat(attn_maps, dim=1)       # [B,C,H,W]
        weighted = torch.cat(weighted_bands, dim=1)  # [B,C,H,W]
        return weighted, attn


# 3) Spectral-spatial fusion via multi-head cross-attention
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralSpatialMultiHeatmapFusion(nn.Module):
    def __init__(self, in_channels, embed_dim=64, num_heads=4):
        super().__init__()
        # project spectral summary to embedding
        self.spec_proj = nn.Linear(in_channels, embed_dim)
        # project spatial pixels (per location vector across bands)
        self.spat_proj = nn.Linear(in_channels, embed_dim)
        # multi-head attention: queries=spec tokens, keys/values=spatial tokens
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)
        # project fusion output back to channel weights
        self.out_proj = nn.Linear(embed_dim, in_channels)

    def forward(self, x):  # x: [B, C, H, W]
        B, C, H, W = x.shape
        # spectral tokens: one per band, summarized spatially
        band_vec = F.adaptive_avg_pool2d(x, 1).view(B, C)    # [B, C]
        spec_tok = self.spec_proj(band_vec).unsqueeze(0)     # [1, B, embed]

        # spatial tokens: each pixel location is a vector of C values
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)   # [B, HW, C]
        spat_tok = self.spat_proj(x_flat)                    # [B, HW, embed]
        spat_tok = spat_tok.permute(1, 0, 2)                 # [HW, B, embed]

        # cross-attention: spec queries spatial
        attn_out, attn_w = self.mha(query=spec_tok, key=spat_tok, value=spat_tok)
        fused = attn_out.squeeze(0)                          # [B, embed]
        ch_w = self.out_proj(fused).view(B, C, 1, 1)         # [B, C, 1, 1]

        return x * ch_w, attn_w


# 4) Complete model integrating all modules and classification head
class SpectralSpatialMultiHeatmapAttentionModel(pl.LightningModule):
    def __init__(self, in_channels, num_classes=1, lr=1e-3, sparsity_coef=1e-4):
        super().__init__()
        self.save_hyperparameters()
        # dynamic spectral and per-band spatial attention
        self.spec_attn = DynamicSpectralAttention(in_channels)
        self.band_spat_attn = DynamicSpatialMultiHeatmapAttention(in_channels)
        # feature extractor
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.ReLU()
        )
        # fusion
        self.fusion = SpectralSpatialMultiHeatmapFusion(
            in_channels=128, embed_dim=64, num_heads=4, patch_size=8
        )
        # classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )
        self.loss_fn = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()

    def forward(self, x):
        x, spec_w = self.spec_attn(x)
        x, band_maps = self.band_spat_attn(x)
        x = self.backbone(x)
        x, fusion_w = self.fusion(x)
        logits = self.classifier(x)
        return logits, spec_w, band_maps, fusion_w

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, spec_w, band_maps, fusion_w = self(x)
        loss = self.loss_fn(logits.squeeze(1) if self.hparams.num_classes==1 else logits,
                            y.float() if self.hparams.num_classes==1 else y)
        # add sparsity regularization on per-band maps
        loss += self.hparams.sparsity_coef * band_maps.abs().mean()
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, spec_w, band_maps, fusion_w = self(x)
        loss = self.loss_fn(logits.squeeze(1) if self.hparams.num_classes==1 else logits,
                            y.float() if self.hparams.num_classes==1 else y)
        # add sparsity regularization on per-band maps
        loss += self.hparams.sparsity_coef * band_maps.abs().mean()
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    

#################################################################################################################################


class DynamicSpatialPatchAttention(nn.Module):
    def __init__(self, in_channels, patch_size=10):
        super().__init__()
        self.patch_size = patch_size
        # self.attn_net = nn.Sequential(
        #     nn.Conv2d(in_channels, 8 * in_channels, kernel_size=3, padding=1, groups=in_channels),  # depthwise
        #     nn.ReLU(),
        #     nn.Conv2d(8 * in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),  # optional: still depthwise
        #     nn.Sigmoid()
        # )

        self.attn_net = nn.Sequential(
            nn.Conv2d(in_channels, 32 * in_channels, kernel_size=3, padding=1, groups=in_channels),  # Depthwise conv
            nn.BatchNorm2d(32 * in_channels),
            nn.ReLU(),
        
            nn.Conv2d(32 * in_channels, 64 * in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(64 * in_channels),
            nn.ReLU(),
        
            nn.Conv2d(64 * in_channels, 128 * in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(128 * in_channels),
            nn.ReLU(),
        
            nn.Conv2d(128 * in_channels, in_channels, kernel_size=1, groups=1),  # Compress back to C channels
            nn.Sigmoid()  # Output per-band spatial attention mask
        )

    def forward(self, x):
        B, C, H, W = x.shape
        ps = self.patch_size

        # Crop to patchable size (if needed)
        new_H = (H // ps) * ps
        new_W = (W // ps) * ps
        x = x[:, :, :new_H, :new_W]

        # Compute full-resolution heatmaps per band in parallel
        heatmaps = self.attn_net(x)  # [B, C, H, W]

        # Apply per-band attention
        x = x * heatmaps  # [B, C, H, W]

        return x, heatmaps


class SpectralSpatialMultiHeatmapPatchFusion(nn.Module):
    def __init__(self, in_channels, embed_dim=64, num_heads=4, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        # project spectral summary to embedding
        self.spec_proj = nn.Linear(in_channels, embed_dim)
        # patchify spatial feature maps
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        # multi-head attention: queries=spec tokens, keys/values=spatial tokens
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)
        # project fusion output back to channel weights
        self.out_proj = nn.Linear(embed_dim, in_channels)

    def forward(self, x):  # x: [B, C, H, W]
        B, C, H, W = x.shape
        # spectral tokens: one per band
        band_vec = F.adaptive_avg_pool2d(x, 1).view(B, C)  # [B,C]
        spec_tok = self.spec_proj(band_vec).unsqueeze(0)   # [1,B,embed]
        # spatial tokens: flattened patches
        p = self.patch_size
        spat = self.patch_embed(x)                        # [B,embed,H/p,W/p]
        P = (H // p) * (W // p)
        spat_tok = spat.flatten(2).permute(2, 0, 1)        # [P,B,embed]
        # cross-attend: spec queries spat
        attn_out, attn_w = self.mha(query=spec_tok, key=spat_tok, value=spat_tok)
        # fuse: reproject to channel weights
        fused = attn_out.squeeze(0)                       # [B,embed]
        ch_w = self.out_proj(fused).view(B, C, 1, 1)      # [B,C,1,1]
        return x * ch_w, attn_w
        

class SpectralSpatialPatchHeatmapAttentionModel(pl.LightningModule):
    def __init__(self, in_channels, num_classes=1, lr=1e-3, sparsity_coef=1e-4):
        super().__init__()
        self.save_hyperparameters()
        # dynamic spectral and per-band spatial attention
        self.spec_attn = DynamicSpectralAttention(in_channels)
        self.band_spat_attn = DynamicSpatialPatchAttention(in_channels)
        # feature extractor
        self.backbone = nn.Sequential(nn.Conv2d(in_channels, 128, 3, padding=1, stride=4), nn.ReLU())
        # fusion
        self.fusion = SpectralSpatialMultiHeatmapPatchFusion(
            in_channels=128, embed_dim=64, num_heads=4, patch_size=8
        )
        # classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )
        self.loss_fn = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()

    def forward(self, x):
        x, spec_w = self.spec_attn(x)
        x, band_maps = self.band_spat_attn(x)
        x = self.backbone(x)
        x, fusion_w = self.fusion(x)
        logits = self.classifier(x)
        return logits, spec_w, band_maps, fusion_w

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, spec_w, band_maps, fusion_w = self(x)
        loss = self.loss_fn(logits.squeeze(1) if self.hparams.num_classes==1 else logits,
                            y.float() if self.hparams.num_classes==1 else y)
        # add sparsity regularization on per-band maps
        loss += self.hparams.sparsity_coef * band_maps.abs().mean()
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, spec_w, band_maps, fusion_w = self(x)
        loss = self.loss_fn(logits.squeeze(1) if self.hparams.num_classes==1 else logits,
                            y.float() if self.hparams.num_classes==1 else y)
        # add sparsity regularization on per-band maps
        loss += self.hparams.sparsity_coef * band_maps.abs().mean()
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)