"""
experiments/noche2/deep2/model_seq_corrected.py
================================================
Arquitectura secuencial honesta corregida:
1. Target anclado a persistencia física: y_pred(t+h) = ReLU(O3(t) + delta(t, h)).
2. Pérdida multi-horizonte normalizada por la varianza de cada horizonte (1 / sigma_h^2).
3. Encoder GRU temporal sobre ventana pasada (48h) de la red de estaciones SIMA.
4. Proyección de contexto espacial contemporáneo en t (37 canales).
5. Proyección de covariables meteorológicas futuras numéricas honestas t+1..t+48 (18 canales).
6. Embedding continuo de estación (13 estaciones).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, hidden_dim)
        w = F.softmax(self.net(x), dim=1) # (B, L, 1)
        return (x * w).sum(dim=1)         # (B, hidden_dim)


class SeqCorrectedModel(nn.Module):
    def __init__(
        self,
        past_channels: int = 36,
        past_len: int = 48,
        future_channels: int = 18,
        future_len: int = 48,
        spatial_dim: int = 37,
        num_stations: int = 13,
        station_emb_dim: int = 16,
        gru_hidden: int = 128,
        future_proj_dim: int = 32,
        hidden_dim: int = 256,
        dropout: float = 0.10
    ):
        super().__init__()
        self.past_len = past_len
        self.future_len = future_len

        # 1. Station embedding
        self.station_emb = nn.Embedding(num_stations, station_emb_dim)

        # 2. Past GRU Encoder
        self.past_proj = nn.Sequential(
            nn.Linear(past_channels, gru_hidden),
            nn.LayerNorm(gru_hidden),
            nn.SiLU()
        )
        self.gru = nn.GRU(
            input_size=gru_hidden,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        self.attn = TemporalAttention(gru_hidden)

        # 3. Spatial projection at t
        self.spatial_proj = nn.Sequential(
            nn.Linear(spatial_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU()
        )

        # 4. Future projection for t+1..t+48
        self.future_proj = nn.Sequential(
            nn.Linear(future_channels, future_proj_dim),
            nn.LayerNorm(future_proj_dim),
            nn.SiLU()
        )

        # 5. Transition encoder
        enc_in = gru_hidden + 64 + station_emb_dim + (future_len * future_proj_dim)
        self.encoder_mlp = nn.Sequential(
            nn.Linear(enc_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        # 6. Multi-Horizon Decoder Head
        self.step_proj = nn.Linear(hidden_dim, future_len * 64)
        self.head = nn.Sequential(
            nn.Linear(64 + future_proj_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1)
        )

    def forward(
        self,
        past_x: torch.Tensor,
        future_x: torch.Tensor,
        spatial_x: torch.Tensor,
        station_idx: torch.Tensor,
        base_o3: torch.Tensor
    ) -> torch.Tensor:
        B = past_x.shape[0]

        # Station embedding
        st_emb = self.station_emb(station_idx)

        # GRU past
        p_in = self.past_proj(past_x)
        gru_out, _ = self.gru(p_in)
        h_past = self.attn(gru_out) # (B, gru_hidden)

        # Spatial
        h_spat = self.spatial_proj(spatial_x) # (B, 64)

        # Future
        p_fut = self.future_proj(future_x) # (B, 48, future_proj_dim)
        p_fut_flat = p_fut.flatten(1)       # (B, 48 * future_proj_dim)

        # Joint encoding
        joint = torch.cat([h_past, h_spat, st_emb, p_fut_flat], dim=1)
        feat = self.encoder_mlp(joint) # (B, hidden_dim)

        # Multi-horizon steps
        step_feat = self.step_proj(feat).view(B, self.future_len, 64) # (B, 48, 64)
        dec_in = torch.cat([step_feat, p_fut], dim=-1)                # (B, 48, 64 + future_proj_dim)
        delta = self.head(dec_in).squeeze(-1)                         # (B, 48)

        # Persistence anchor: y_pred = ReLU(base_o3 + delta)
        # base_o3 is (B, 1)
        y_pred = F.relu(base_o3 + delta)
        return y_pred
