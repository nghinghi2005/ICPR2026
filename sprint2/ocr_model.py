from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionFusion(nn.Module):
    """Temporal attention across frames for CRNN features."""

    def __init__(self, channels: int, frames: int = 5):
        super().__init__()
        self.frames = int(frames)
        self.score_net = nn.Sequential(
            nn.Conv2d(channels, max(1, channels // 8), kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(max(1, channels // 8), 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*T, C, 1, W]
        bt, c, h, w = x.size()
        assert h == 1, "CRNN backbone should end with height=1"
        b = bt // self.frames

        scores = self.score_net(x).view(b, self.frames, 1, w)
        x_view = x.view(b, self.frames, c, w)
        weights = F.softmax(scores, dim=1)
        fused = torch.sum(x_view * weights, dim=1)  # [B, C, W]
        return fused


class MultiFrameCRNN(nn.Module):
    """Multi-frame CRNN for license plate recognition (CTC)."""

    def __init__(self, num_classes: int, frames: int = 5, hidden_size: int = 256):
        super().__init__()
        self.frames = int(frames)

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(512, 512, 2, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

        self.fusion = AttentionFusion(channels=512, frames=self.frames)
        self.rnn = nn.LSTM(
            512,
            hidden_size,
            bidirectional=True,
            batch_first=True,
            num_layers=2,
            dropout=0.25,
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, 3, H, W]
        b, t, c, h, w = x.size()
        assert t == self.frames, f"expected {self.frames} frames, got {t}"

        x = x.view(b * t, c, h, w)
        feat = self.cnn(x)  # [B*T, 512, 1, W']
        fused = self.fusion(feat)  # [B, 512, W']

        seq = fused.permute(0, 2, 1)  # [B, W', 512]
        out, _ = self.rnn(seq)  # [B, W', 2H]
        logits = self.fc(out)  # [B, W', C]
        return logits.log_softmax(2)
