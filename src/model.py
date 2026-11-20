"""PyTorch port of the LSTM + attention network in the lab's CMEpredict.py (Keras).

Keras original: LSTM(10, return_sequences, dropout=0.5, recurrent_dropout=0.3)
-> attention_3d_block -> Dense(100, relu) -> Dropout(0.25) -> Dense(1, sigmoid).
Differences: PyTorch's LSTM has no recurrent dropout, so only input dropout is used;
the tiny L2 activity regularizer on the output (1e-4) is omitted.
"""
import torch
import torch.nn as nn


class AttentionBlock(nn.Module):
    """Same computation as attention_3d_block in CMEpredict.py."""

    def __init__(self, hidden: int, series_len: int):
        super().__init__()
        self.score = nn.Linear(series_len, series_len, bias=False)   # Dense(series_len) on (B,H,T)
        self.out = nn.Linear(2 * hidden, hidden, bias=False)          # Dense(hidden, tanh)

    def forward(self, h):                       # h: (B, T, H)
        h_t = h.transpose(1, 2)                 # (B, H, T)
        s = self.score(h_t).transpose(1, 2)     # (B, T, H)
        last = h[:, -1, :]                      # (B, H)
        score = (s * last.unsqueeze(1)).sum(-1)  # (B, T)
        w = torch.softmax(score, dim=1)
        context = (h * w.unsqueeze(-1)).sum(1)   # (B, H)
        return torch.tanh(self.out(torch.cat([context, last], dim=1)))


class LSTMAttention(nn.Module):
    def __init__(self, n_features: int, series_len: int = 20, hidden: int = 10):
        super().__init__()
        self.in_drop = nn.Dropout(0.5)
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.att = AttentionBlock(hidden, series_len)
        self.head = nn.Sequential(nn.Linear(hidden, 100), nn.ReLU(), nn.Dropout(0.25), nn.Linear(100, 1))

    def forward(self, x):                       # x: (B, T, F) -> probability (B, 1)
        h, _ = self.lstm(self.in_drop(x))
        return torch.sigmoid(self.head(self.att(h)))
