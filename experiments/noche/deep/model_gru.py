"""
experiments/noche/deep/model_gru.py
-----------------------------------
Recurrent GRU Encoder + Multi-Horizon Decoder with future covariates.
"""
import os, sys
ROOT = r"D:\Codingggg\Notas\Multivariados\MA2003B_Eq1"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
import torch
import torch.nn as nn
import torch.nn.functional as F
from experiments.noche.deep.model_tide import ResidualBlock

class GRUTiDEModel(nn.Module):
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
        hidden_dim=256,
        decoder_dim=64,
        dropout=0.15
    ):
        super().__init__()
        self.past_len = past_len
        self.future_len = future_len
        
        # 1. Station embedding
        self.station_emb = nn.Embedding(num_stations, station_emb_dim)
        
        # 2. GRU Past Encoder
        self.gru_in = nn.Sequential(
            nn.Linear(past_channels, 128),
            nn.LayerNorm(128),
            nn.SiLU()
        )
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        # Temporal attention over GRU hidden states
        self.attn = nn.Sequential(
            nn.Linear(128, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        # 3. Spatial projection
        self.spatial_proj = nn.Linear(spatial_dim, 64)
        
        # 4. Future projection
        self.future_proj = nn.Sequential(
            nn.Linear(future_channels, future_proj_dim),
            nn.LayerNorm(future_proj_dim),
            nn.SiLU()
        )
        
        # 5. Global linear skip
        self.residual_skip = nn.Linear(past_len, future_len)
        
        # 6. Dense transition blocks
        # GRU attn pool (128) + spatial (64) + station_emb (16) + future_flat (48 * future_proj_dim)
        enc_in_dim = 128 + 64 + station_emb_dim + (future_len * future_proj_dim)
        self.encoder_in = nn.Sequential(
            nn.Linear(enc_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        self.encoder_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout) for _ in range(2)
        ])
        
        # 7. Decoder
        self.decoder_proj = nn.Linear(hidden_dim, future_len * decoder_dim)
        self.decoder_dim = decoder_dim
        
        head_in_dim = decoder_dim + future_proj_dim
        self.head = nn.Sequential(
            nn.Linear(head_in_dim, decoder_dim),
            nn.LayerNorm(decoder_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(decoder_dim, 1)
        )

    def forward(self, past_x, future_x, spatial_x, station_idx, clim_future, sigma_res=12.4):
        B = past_x.shape[0]
        st_emb = self.station_emb(station_idx)
        
        # GRU encoding
        x_proj = self.gru_in(past_x)
        gru_out, _ = self.gru(x_proj)                         # (B, L, 128)
        attn_w = F.softmax(self.attn(gru_out), dim=1)        # (B, L, 1)
        h_past = (gru_out * attn_w).sum(dim=1)               # (B, 128)
        
        h_spatial = F.silu(self.spatial_proj(spatial_x))
        
        p_future = self.future_proj(future_x)
        p_future_flat = p_future.flatten(1)
        
        enc_inp = torch.cat([h_past, h_spatial, st_emb, p_future_flat], dim=1)
        feat = self.encoder_in(enc_inp)
        for blk in self.encoder_blocks:
            feat = blk(feat)
            
        dec_feat = self.decoder_proj(feat).view(B, self.future_len, self.decoder_dim)
        step_inp = torch.cat([dec_feat, p_future], dim=-1)
        out_res = self.head(step_inp).squeeze(-1)
        
        skip_res = self.residual_skip(past_x[:, :, 0])
        r_pred = out_res + skip_res
        
        y_pred = torch.clamp(clim_future + r_pred * sigma_res, min=0.0)
        return y_pred
