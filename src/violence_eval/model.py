import torch
import torch.nn as nn
import math

class MILScorer(nn.Module):
    def __init__(self, input_dim=2048, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

class EventClassifier(nn.Module):
    def __init__(self, input_dim=2048, num_classes=7, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
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
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TemporalRefinementNet(nn.Module):
    def __init__(self, input_dim=2048, d_model=512, nhead=8, num_layers=2,
                 dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.score_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x):
        h = self.pos_enc(self.input_proj(x))
        h = self.transformer(h)
        scores = self.score_head(h).squeeze(-1)
        return scores, h

class BoundaryHead(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model*2 + 1, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, hidden, scores):
        h_t   = hidden[:, :-1, :]
        h_tp1 = hidden[:, 1:, :]
        score_diff = (scores[:, :-1] - scores[:, 1:]).abs().unsqueeze(-1)
        inp = torch.cat([h_t, h_tp1, score_diff], dim=-1)
        return self.net(inp).squeeze(-1)

class ViolenceDetector(nn.Module):
    def __init__(self, input_dim=2048, num_classes=7, d_model=512,
                 nhead=8, trn_layers=2, dropout_mil=0.5, dropout_cls=0.3, dropout_trn=0.1):
        super().__init__()
        self.mil_scorer = MILScorer(input_dim, dropout_mil)
        self.classifier = EventClassifier(input_dim, num_classes, dropout_cls)
        self.trn = TemporalRefinementNet(input_dim, d_model, nhead, trn_layers, d_model*2, dropout_trn)
        self.boundary = BoundaryHead(d_model)
    def forward(self, x):
        mil_scores = self.mil_scorer(x)
        cls_logits = self.classifier(x)
        trn_scores, hidden = self.trn(x)
        boundary = self.boundary(hidden, trn_scores)
        return {'mil_scores': mil_scores, 'trn_scores': trn_scores,
                'cls_logits': cls_logits, 'boundary': boundary}