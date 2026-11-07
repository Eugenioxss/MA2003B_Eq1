"""
experiments/noche2/deep2/model.py
=================================
Arquitecturas neuronales profundas para datos tabulares multi-estación.
- TabResNet: Red Residual profunda para datos tabulares con embeddings de estación y hora.
- TabResNetQuantile: Variante con cabezal monotónico garantizado para cuantiles [q10, q50, q90].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TabResNetBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.05):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.bn1(x)
        out = self.act(self.fc1(out))
        out = self.drop1(out)
        out = self.bn2(out)
        out = self.act(self.fc2(out))
        out = self.drop2(out)
        return res + out


class TabResNet(nn.Module):
    def __init__(
        self,
        num_stations: int,
        num_features: int,
        embed_dim: int = 16,
        hr_embed_dim: int = 8,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.05
    ):
        super().__init__()
        self.st_emb = nn.Embedding(num_stations, embed_dim)
        self.hr_emb = nn.Embedding(24, hr_embed_dim)
        in_dim = num_features + embed_dim + hr_embed_dim
        self.input_layer = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([TabResNetBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.head_bn = nn.BatchNorm1d(hidden_dim)
        self.head_act = nn.SiLU()
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, st_idx: torch.Tensor, hr_idx: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        emb_st = self.st_emb(st_idx)
        emb_hr = self.hr_emb(hr_idx)
        x = torch.cat([x_num, emb_st, emb_hr], dim=1)
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        h = self.head_act(self.head_bn(h))
        return self.head(h)


class MonotoneQuantileHead(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.fc_pred = nn.Linear(in_features, 1)
        self.fc_d10 = nn.Linear(in_features, 1)
        self.fc_d90 = nn.Linear(in_features, 1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred = self.fc_pred(h)
        d10 = F.softplus(self.fc_d10(h))
        d90 = F.softplus(self.fc_d90(h))
        q10 = pred - d10
        q90 = pred + d90
        return pred, q10, q90


class TabResNetQuantile(nn.Module):
    def __init__(
        self,
        num_stations: int,
        num_features: int,
        embed_dim: int = 16,
        hr_embed_dim: int = 8,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.05
    ):
        super().__init__()
        self.st_emb = nn.Embedding(num_stations, embed_dim)
        self.hr_emb = nn.Embedding(24, hr_embed_dim)
        in_dim = num_features + embed_dim + hr_embed_dim
        self.input_layer = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([TabResNetBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.head_bn = nn.BatchNorm1d(hidden_dim)
        self.head_act = nn.SiLU()
        self.qhead = MonotoneQuantileHead(hidden_dim)

    def forward(self, st_idx: torch.Tensor, hr_idx: torch.Tensor, x_num: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb_st = self.st_emb(st_idx)
        emb_hr = self.hr_emb(hr_idx)
        x = torch.cat([x_num, emb_st, emb_hr], dim=1)
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        h = self.head_act(self.head_bn(h))
        return self.qhead(h)


def pinball_loss(y_pred: torch.Tensor, y_true: torch.Tensor, q: float) -> torch.Tensor:
    err = y_true - y_pred
    return torch.mean(torch.max(q * err, (q - 1.0) * err))
