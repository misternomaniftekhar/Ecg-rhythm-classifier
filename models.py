"""models.py - 1D-CNN, BiLSTM and Transformer beat classifiers (PyTorch).

Every model takes (beat waveform [B,200], RR features [B,4]) -> logits [B,5].
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ecg_pipeline import BEAT_LEN, CLASSES, N_RR


def _head(in_dim, n_classes):
    return nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.3),
                         nn.Linear(64, n_classes))


class CNN1D(nn.Module):
    def __init__(self, n_classes=len(CLASSES), n_rr=N_RR):
        super().__init__()

        def block(i, o, k):
            return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o),
                                 nn.ReLU(), nn.MaxPool1d(2))

        self.features = nn.Sequential(block(1, 32, 7), block(32, 64, 5),
                                      block(64, 128, 3), block(128, 128, 3))
        self.head = _head(256 + n_rr, n_classes)

    def forward(self, x, rr):
        h = self.features(x.unsqueeze(1))
        h = torch.cat([h.mean(-1), h.amax(-1)], dim=1)
        return self.head(torch.cat([h, rr], dim=1))


class LSTMNet(nn.Module):
    def __init__(self, n_classes=len(CLASSES), n_rr=N_RR):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2))
        self.lstm = nn.LSTM(16, 64, batch_first=True, bidirectional=True)
        self.head = _head(256 + n_rr, n_classes)

    def forward(self, x, rr):
        h = self.stem(x.unsqueeze(1)).transpose(1, 2)
        out, _ = self.lstm(h)
        h = torch.cat([out.mean(1), out.amax(1)], dim=1)
        return self.head(torch.cat([h, rr], dim=1))


class TransformerNet(nn.Module):
    def __init__(self, n_classes=len(CLASSES), n_rr=N_RR, d=64, patch=10):
        super().__init__()
        self.embed = nn.Conv1d(1, d, patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, BEAT_LEN // patch, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, 4, 128, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = _head(d + n_rr, n_classes)

    def forward(self, x, rr):
        h = self.embed(x.unsqueeze(1)).transpose(1, 2) + self.pos
        h = self.encoder(h).mean(1)
        return self.head(torch.cat([h, rr], dim=1))


MODEL_REGISTRY = {"cnn": CNN1D, "lstm": LSTMNet, "transformer": TransformerNet}


@torch.no_grad()
def predict_proba(model, X, R, device="cpu", batch=1024):
    model.eval().to(device)
    out = []
    for i in range(0, len(X), batch):
        xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=device)
        rb = torch.as_tensor(R[i:i + batch], dtype=torch.float32, device=device)
        out.append(torch.softmax(model(xb, rb), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(CLASSES)), np.float32)


def load_model(key, path):
    model = MODEL_REGISTRY[key]()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()
