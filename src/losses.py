"""
Loss functions for the violence detection framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def mil_ranking_loss(anom_scores, norm_scores, topk_ratio=0.1):
    """
    MIL ranking loss from Sultani et al. 2018.
    Anomalous video max score should exceed normal video max score by margin 1.

    Args:
        anom_scores: (B, T) scores for anomalous videos
        norm_scores: (B, T) scores for normal videos
        topk_ratio:  fraction of top segments to use (more stable than hard max)
    Returns:
        scalar loss
    """
    k = max(1, int(anom_scores.shape[1] * topk_ratio))

    anom_top = anom_scores.topk(k, dim=1).values.mean(dim=1)  # (B,)
    norm_top  = norm_scores.topk(k, dim=1).values.mean(dim=1) # (B,)

    loss = F.relu(1.0 - anom_top + norm_top)
    return loss.mean()


def classification_loss(cls_logits, anom_scores, cat_ids, topk=5):
    """
    Pseudo-label classification loss.
    For each anomalous video, top-k segments by MIL score get the video's category label.

    Args:
        cls_logits: (B, T, 7)  per-segment class logits (anomalous batch only)
        anom_scores:(B, T)     MIL scores (anomalous batch only)
        cat_ids:    (B,)       video-level category id (1-6)
        topk:       number of pseudo-positive segments per video
    Returns:
        scalar loss
    """
    B, T, C = cls_logits.shape
    k = min(topk, T)

    topk_idx = anom_scores.topk(k, dim=1).indices  # (B, k)

    losses = []
    for b in range(B):
        seg_logits = cls_logits[b, topk_idx[b], :]  # (k, 7)
        target = cat_ids[b].expand(k)               # (k,)
        losses.append(F.cross_entropy(seg_logits, target))

    return torch.stack(losses).mean()


def boundary_consistency_loss(boundary_conf, trn_scores):
    """
    Boundary confidence should correlate with score change magnitude.

    Args:
        boundary_conf: (B, T-1) predicted boundary confidence
        trn_scores:    (B, T)   refined anomaly scores
    Returns:
        scalar loss
    """
    score_diff = (trn_scores[:, :-1] - trn_scores[:, 1:]).abs()  # (B, T-1)
    return F.mse_loss(boundary_conf, score_diff)


def smoothness_loss(trn_scores):
    """
    Temporal smoothness: penalize large consecutive score changes.

    Args:
        trn_scores: (B, T)
    Returns:
        scalar loss
    """
    diff = trn_scores[:, 1:] - trn_scores[:, :-1]
    return (diff ** 2).mean()


class ViolenceLoss(nn.Module):
    """
    Combined loss with progressive stage control.

    Stages:
      1 — MIL ranking only
      2 — MIL + classification
      3 — MIL + classification + TRN ranking + boundary consistency

    Key design: Stage 3 adds a RANKING loss on trn_scores (same formulation as
    MIL ranking but applied to TRN outputs). This is the critical supervision
    signal the TRN needs to learn discrimination. Without it, the TRN only
    receives smoothness/boundary signals and cannot learn to distinguish
    anomalous from normal — causing the AUC to collapse.

    Smoothness loss is intentionally NOT used in stage 3 because it pushes
    consecutive TRN scores toward uniformity, which actively harms discrimination.
    """
    def __init__(self, lambda_cls=0.5, lambda_bnd=0.1, lambda_trn=1.0,
                 topk_ratio=0.1, topk_cls=5):
        super().__init__()
        self.lambda_cls    = lambda_cls
        self.lambda_bnd    = lambda_bnd
        self.lambda_trn    = lambda_trn   # weight for TRN ranking loss
        self.topk_ratio    = topk_ratio
        self.topk_cls      = topk_cls

    def forward(self, out_anom, out_norm, cat_ids, stage=3):
        """
        Args:
            out_anom:  model output dict for anomalous batch
            out_norm:  model output dict for normal batch
            cat_ids:   (B,) category ids for anomalous videos
            stage:     training stage (1, 2, or 3)
        Returns:
            total loss (scalar), dict of individual losses
        """
        losses = {}

        # MIL ranking loss — always active, trains MIL scorer
        l_mil = mil_ranking_loss(
            out_anom['mil_scores'], out_norm['mil_scores'], self.topk_ratio
        )
        losses['mil'] = l_mil
        total = l_mil

        if stage >= 2:
            # Classification loss — pseudo-labels from top MIL segments
            l_cls = classification_loss(
                out_anom['cls_logits'], out_anom['mil_scores'],
                cat_ids, self.topk_cls
            )
            losses['cls'] = l_cls
            total = total + self.lambda_cls * l_cls

        if stage >= 3:
            # TRN ranking loss — direct discrimination signal for the TRN.
            # Same formulation as MIL ranking but applied to trn_scores.
            # This is what was MISSING in the original Stage 3, causing AUC collapse.
            l_trn = mil_ranking_loss(
                out_anom['trn_scores'], out_norm['trn_scores'], self.topk_ratio
            )
            losses['trn'] = l_trn
            total = total + self.lambda_trn * l_trn

            # Boundary consistency — predict high confidence at large score transitions
            l_bnd = boundary_consistency_loss(
                out_anom['boundary'], out_anom['trn_scores']
            )
            losses['bnd'] = l_bnd
            total = total + self.lambda_bnd * l_bnd
            # NOTE: smoothness loss intentionally removed — it pushes TRN scores
            # toward uniformity which destroys the discrimination signal.

        losses['total'] = total
        return total, losses


if __name__ == '__main__':
    B, T = 4, 32
    out_anom = {
        'mil_scores': torch.rand(B, T),
        'trn_scores': torch.rand(B, T),
        'cls_logits': torch.randn(B, T, 7),
        'boundary':   torch.rand(B, T - 1),
    }
    out_norm = {
        'mil_scores': torch.rand(B, T),
        'trn_scores': torch.rand(B, T),
        'cls_logits': torch.randn(B, T, 7),
        'boundary':   torch.rand(B, T - 1),
    }
    cat_ids = torch.randint(1, 7, (B,))

    criterion = ViolenceLoss()
    for stage in [1, 2, 3]:
        total, breakdown = criterion(out_anom, out_norm, cat_ids, stage=stage)
        print(f"Stage {stage}: total={total:.4f}  " +
              "  ".join(f"{k}={v:.4f}" for k, v in breakdown.items() if k != 'total'))
