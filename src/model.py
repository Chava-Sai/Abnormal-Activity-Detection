"""
Violence Event Detection Model.
Components:
  1. MIL Anomaly Scorer      — segment-level anomaly scores
  2. Event Classifier        — 7-class (6 violence + normal)
  3. Temporal Refinement Net — transformer-based temporal context
  4. Boundary Head           — start/end boundary confidence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MILScorer(nn.Module):
    """
    MIL anomaly scorer: 2048 → 512 → 128 → 1 (sigmoid).
    Operates per segment independently.
    """
    def __init__(self, input_dim=2048, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, T, 2048) → scores: (B, T)
        return self.net(x).squeeze(-1)


class EventClassifier(nn.Module):
    """
    Per-segment event classifier: 2048 → 7 classes (6 violence + normal).
    """
    def __init__(self, input_dim=2048, num_classes=7, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # x: (B, T, 2048) → logits: (B, T, 7)
        return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TemporalRefinementNet(nn.Module):
    """
    Transformer-based temporal refinement.
    Input: (B, T, 2048) features
    Output: refined scores (B, T) and hidden states (B, T, d_model)
    """
    def __init__(self, input_dim=2048, d_model=512, nhead=8, num_layers=2,
                 dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.score_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, T, 2048)
        h = self.pos_enc(self.input_proj(x))   # (B, T, d_model)
        h = self.transformer(h)                 # (B, T, d_model)
        scores = self.score_head(h).squeeze(-1) # (B, T)
        return scores, h


class BoundaryHead(nn.Module):
    """
    Boundary confidence from adjacent segment pairs.
    Uses: [h_t ; h_{t+1} ; |s_t - s_{t+1}|] → b_t
    """
    def __init__(self, d_model=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 2 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden, scores):
        # hidden: (B, T, d_model), scores: (B, T)
        h_t   = hidden[:, :-1, :]
        h_tp1 = hidden[:, 1:, :]
        score_diff = (scores[:, :-1] - scores[:, 1:]).abs().unsqueeze(-1)
        inp = torch.cat([h_t, h_tp1, score_diff], dim=-1)
        return self.net(inp).squeeze(-1)  # (B, T-1)


class ViolenceDetector(nn.Module):
    """
    Full violence detection model.

    Forward returns a dict with:
      mil_scores   : (B, T)    — raw MIL anomaly scores
      trn_scores   : (B, T)    — TRN-refined anomaly scores
      cls_logits   : (B, T, 7) — per-segment class logits
      boundary     : (B, T-1)  — boundary confidence
    """
    def __init__(
        self,
        input_dim=2048,
        num_classes=7,
        d_model=512,
        nhead=8,
        trn_layers=2,
        dropout_mil=0.5,
        dropout_cls=0.3,
        dropout_trn=0.1,
    ):
        super().__init__()
        self.mil_scorer  = MILScorer(input_dim, dropout=dropout_mil)
        self.classifier  = EventClassifier(input_dim, num_classes, dropout=dropout_cls)
        self.trn         = TemporalRefinementNet(input_dim, d_model, nhead, trn_layers,
                                                  d_model * 2, dropout_trn)
        self.boundary    = BoundaryHead(d_model)

    def forward(self, x):
        # x: (B, T, 2048)
        mil_scores  = self.mil_scorer(x)
        cls_logits  = self.classifier(x)
        trn_scores, hidden = self.trn(x)
        boundary    = self.boundary(hidden, trn_scores)

        return {
            'mil_scores':  mil_scores,
            'trn_scores':  trn_scores,
            'cls_logits':  cls_logits,
            'boundary':    boundary,
        }

    def param_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


if __name__ == '__main__':
    model = ViolenceDetector()
    total, trainable = model.param_count()
    print(f"Total params    : {total/1e6:.2f}M")
    print(f"Trainable params: {trainable/1e6:.2f}M")

    B, T, D = 4, 32, 2048
    x = torch.randn(B, T, D)
    out = model(x)
    print(f"\nForward pass (B={B}, T={T}):")
    for k, v in out.items():
        print(f"  {k:15s}: {tuple(v.shape)}")
