"""
TiDE Model Architecture for multi-horizon ozone forecasting.
Google TiDE (Time-series Dense Encoder) adapted for SIMA Monterrey O3.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.norm2(out)
        out = self.act(out + residual)
        return out

class ConvPastEncoder(nn.Module):
    """
    1D Temporal Convolution past encoder:
    Transforms (B, L, C_past) -> (B, hidden_dim).
    Captures multi-scale trends and diurnal patterns across 96 hours.
    """
    def __init__(self, in_channels, seq_len=96, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=5, padding=2)
        self.norm1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)  # 96 -> 48
        self.norm2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=5, stride=2, padding=2)  # 48 -> 24
        self.norm3 = nn.BatchNorm1d(128)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        
        flat_dim = 128 * 24
        self.fc_out = nn.Linear(flat_dim, hidden_dim)
        self.norm_out = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (B, L, C) -> transpose to (B, C, L)
        x = x.transpose(1, 2)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.drop(out)
        out = self.act(self.norm2(self.conv2(out)))
        out = self.drop(out)
        out = self.act(self.norm3(self.conv3(out)))
        out = out.flatten(1)
        out = self.act(self.norm_out(self.fc_out(out)))
        return out

class TiDEModel(nn.Module):
    def __init__(
        self,
        past_channels=36,
        past_len=96,
        future_channels=18,
        future_len=48,
        spatial_dim=37,
        num_stations=13,
        station_emb_dim=16,
        future_proj_dim=16,
        hidden_dim=384,
        decoder_dim=64,
        num_encoder_blocks=3,
        num_heads=1,  # 1 for MSE point prediction, 3 for quantiles (0.1, 0.5, 0.9)
        dropout=0.15
    ):
        super().__init__()
        self.past_len = past_len
        self.future_len = future_len
        self.num_heads = num_heads
        
        # 1. Station embedding
        self.station_emb = nn.Embedding(num_stations, station_emb_dim)
        
        # 2. Past sequence encoder
        self.past_encoder = ConvPastEncoder(past_channels, past_len, hidden_dim=256, dropout=dropout)
        # Direct projection of state at t (lag 0)
        self.lag0_proj = nn.Linear(past_channels, 64)
        
        # 3. Spatial context projection
        self.spatial_proj = nn.Linear(spatial_dim, 64)
        
        # 4. Future covariates projection (shared across horizons)
        self.future_proj = nn.Sequential(
            nn.Linear(future_channels, future_proj_dim),
            nn.LayerNorm(future_proj_dim),
            nn.SiLU()
        )
        
        # 5. Global linear residual skip: maps past O3 history directly to future horizons
        self.residual_skip = nn.Linear(past_len, future_len)
        
        # 6. Dense Encoder
        # In dimension: past_encoder (256) + lag0 (64) + spatial (64) + station_emb (16) + future_proj (48 * future_proj_dim)
        enc_in_dim = 256 + 64 + 64 + station_emb_dim + (future_len * future_proj_dim)
        self.encoder_in = nn.Sequential(
            nn.Linear(enc_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        self.encoder_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_encoder_blocks)
        ])
        
        # 7. Dense Decoder
        self.decoder_proj = nn.Linear(hidden_dim, future_len * decoder_dim)
        self.decoder_dim = decoder_dim
        self.future_proj_dim = future_proj_dim
        
        # Per-horizon output head: takes decoder_dim + future_proj_dim
        head_in_dim = decoder_dim + future_proj_dim
        self.head = nn.Sequential(
            nn.Linear(head_in_dim, decoder_dim),
            nn.LayerNorm(decoder_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(decoder_dim, num_heads)
        )

    def forward(self, past_x, future_x, spatial_x, station_idx, clim_future, sigma_res=12.4):
        """
        past_x:        (B, past_len, past_channels)
        future_x:      (B, future_len, future_channels)
        spatial_x:     (B, spatial_dim)
        station_idx:   (B,)
        clim_future:   (B, future_len)
        sigma_res:     float
        """
        B = past_x.shape[0]
        
        # 1. Embeddings & encoders
        st_emb = self.station_emb(station_idx)                       # (B, station_emb_dim)
        h_past = self.past_encoder(past_x)                           # (B, 256)
        h_lag0 = F.silu(self.lag0_proj(past_x[:, -1, :]))           # (B, 64)
        h_spatial = F.silu(self.spatial_proj(spatial_x))             # (B, 64)
        
        # Future covariates projection
        # future_x: (B, 48, future_channels) -> (B, 48, future_proj_dim)
        p_future = self.future_proj(future_x)
        p_future_flat = p_future.flatten(1)                          # (B, 48 * future_proj_dim)
        
        # Concatenate encoder input
        enc_inp = torch.cat([h_past, h_lag0, h_spatial, st_emb, p_future_flat], dim=1)
        feat = self.encoder_in(enc_inp)
        for blk in self.encoder_blocks:
            feat = blk(feat)
            
        # Decoder projection
        dec_feat = self.decoder_proj(feat).view(B, self.future_len, self.decoder_dim)  # (B, 48, 64)
        
        # Combine per horizon with projected future covariate
        step_inp = torch.cat([dec_feat, p_future], dim=-1)           # (B, 48, 64 + 16 = 80)
        out_heads = self.head(step_inp)                              # (B, 48, num_heads)
        
        # Global linear skip from past O3 channel (past_x[:, :, 0])
        skip_res = self.residual_skip(past_x[:, :, 0])               # (B, 48)
        
        if self.num_heads == 1:
            # Single MSE head: predicts residual in standardized scale
            r_pred = out_heads.squeeze(-1) + skip_res                # (B, 48)
            y_pred = clim_future + r_pred * sigma_res
            y_pred = torch.clamp(y_pred, min=0.0)
            return y_pred
        else:
            # Multi-quantile head (q10, q50, q90):
            # Head 0: base residual for q50
            # Head 1: delta for q10 (softplus)
            # Head 2: delta for q90 (softplus)
            r_50 = out_heads[:, :, 0] + skip_res
            delta_10 = F.softplus(out_heads[:, :, 1])
            delta_90 = F.softplus(out_heads[:, :, 2])
            
            r_10 = r_50 - delta_10
            r_90 = r_50 + delta_90
            
            # Reconstruct y in ppb
            y_50 = torch.clamp(clim_future + r_50 * sigma_res, min=0.0)
            y_10 = torch.clamp(clim_future + r_10 * sigma_res, min=0.0)
            y_90 = torch.clamp(clim_future + r_90 * sigma_res, min=0.0)
            
            return torch.stack([y_10, y_50, y_90], dim=-1)          # (B, 48, 3)
