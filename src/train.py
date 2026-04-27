"""
Training script for Violence Event Detection.
Progressive 3-stage training:
  Stage 1 (Epochs 1-30):   MIL scorer only
  Stage 2 (Epochs 31-60):  MIL + event classifier
  Stage 3 (Epochs 61-100): MIL + classifier + TRN + boundary head

Evaluation uses a stratified val split carved from the training set.
This avoids the issue where the standard 109-video test set is missing
most violence categories (Fighting, Shooting, Explosion, Abuse, Riot).

Frame-level AUC is computed using segment scores × 16 frames/segment
with proxy GT (all frames in anomalous video = 1). This is lower-bound
of real frame-level AUC but tracks temporal localization progress.
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            print("tensorboard not installed; scalar logging disabled")

        def add_scalar(self, *args, **kwargs):
            return None

        def close(self):
            return None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ViolenceDetector
from losses import ViolenceLoss
from dataset import UCFCrimeDataset, MILDataset, build_dataloaders

CAT_ID_TO_NAME = {
    0: 'Normal', 1: 'Abuse', 2: 'Fighting',
    3: 'Shooting', 4: 'Explosion', 5: 'Robbery', 6: 'Riot',
}


def get_stage(epoch):
    if epoch <= 30:
        return 1
    elif epoch <= 60:
        return 2
    else:
        return 3


def training_schedule_name(args) -> str:
    if args.joint:
        return 'joint'
    if args.max_stage >= 3:
        return 'progressive'
    return f'progressive_stage{args.max_stage}'


def resolve_run_root(checkpoint_dir: str, log_dir: str) -> Path:
    ckpt = Path(checkpoint_dir).expanduser().resolve()
    log = Path(log_dir).expanduser().resolve()
    try:
        common = Path(os.path.commonpath([str(ckpt), str(log)]))
    except ValueError:
        common = ckpt.parent
    return common


def resolve_device(requested: str) -> torch.device:
    if requested == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    if requested == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError("Requested --device cuda but CUDA is not available")
        return torch.device('cuda')

    if requested == 'mps':
        if not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available():
            raise ValueError("Requested --device mps but MPS is not available")
        return torch.device('mps')

    if requested == 'cpu':
        return torch.device('cpu')

    raise ValueError(f"unsupported device: {requested}")


def save_run_summary(args, best_epoch, best_metrics):
    run_root = resolve_run_root(args.checkpoint_dir, args.log_dir)
    payload = {
        'training_schedule': training_schedule_name(args),
        'joint': bool(args.joint),
        'seed': int(args.seed),
        'num_segments': int(args.num_segments),
        'input_dim': int(args.input_dim),
        'frames_per_segment': int(args.frames_per_segment),
        'epochs': int(args.epochs),
        'best_epoch': int(best_epoch),
        'best_metrics': best_metrics,
        'eval_split': 'external_test' if args.no_val_split else 'val_split',
        'checkpoint_dir': str(Path(args.checkpoint_dir).expanduser().resolve()),
        'log_dir': str(Path(args.log_dir).expanduser().resolve()),
    }
    out_path = run_root / 'run_summary.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Run summary: {out_path}")


def set_stage_params(model, optimizer, stage, lr, reset_lr_on_stage3=True):
    """
    Freeze/unfreeze components based on training stage and reset LR.

    Stage 3 resets LR to the base rate because the TRN is initialized randomly
    and joins training at epoch 61 — by which point the LR scheduler has already
    decayed the LR by 100x. Without a reset, TRN would train at LR=1e-6 from
    random init, causing slow/no convergence.
    """
    for name, param in model.named_parameters():
        if stage == 1:
            param.requires_grad = 'mil_scorer' in name
        elif stage == 2:
            param.requires_grad = ('mil_scorer' in name or 'classifier' in name)
        else:
            param.requires_grad = True

    active_params = [p for p in model.parameters() if p.requires_grad]
    optimizer.param_groups[0]['params'] = active_params

    # Reset LR at Stage 3 so the TRN can learn from random init
    if stage == 3 and reset_lr_on_stage3:
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        print(f"  [LR reset to {lr} for Stage 3 TRN training]")

    n_active = sum(p.numel() for p in active_params)
    return n_active


def train_one_epoch(model, loader, criterion, optimizer, device, stage, epoch):
    model.train()
    total_loss = 0.0
    loss_breakdown = {}
    n_batches = 0

    for batch in loader:
        anom_feat = batch['anom_features'].to(device)  # (B, T, 2048)
        norm_feat = batch['norm_features'].to(device)
        cat_ids   = batch['anom_cat'].to(device)

        optimizer.zero_grad()

        out_anom = model(anom_feat)
        out_norm = model(norm_feat)

        loss, breakdown = criterion(out_anom, out_norm, cat_ids, stage=stage)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        for k, v in breakdown.items():
            loss_breakdown[k] = loss_breakdown.get(k, 0.0) + v.item()
        n_batches += 1

    avg = {k: v / n_batches for k, v in loss_breakdown.items()}
    return avg


@torch.no_grad()
def evaluate(model, loader, device, frames_per_segment: int = 16, stage: int = 3):
    """
    Compute video-level AUC, frame-level AUC (proxy GT), and per-category AP.

    Score selection by stage:
      Stage 1-2: use mil_scores (TRN not yet trained, its scores are random)
      Stage 3:   use trn_scores (full model trained, TRN refines temporal context)

    Frame-level AUC: segment scores repeated frames_per_segment times, proxy GT
    where all frames in anomalous video = 1.
    """
    model.eval()

    all_vid_scores = []
    all_vid_labels = []
    all_frm_pred   = []
    all_frm_gt     = []
    by_cat         = {}

    for batch in loader:
        feat   = batch['features'].to(device)   # (1, T, 2048)
        label  = batch['label'].item()
        cat_id = batch['cat_id'].item()

        out = model(feat)
        # Use MIL scores until TRN is trained (stage 3)
        if stage < 3:
            seg_scores = out['mil_scores'].squeeze(0).cpu().numpy()
        else:
            seg_scores = out['trn_scores'].squeeze(0).cpu().numpy()
        T = len(seg_scores)

        vid_score = float(seg_scores.max())
        all_vid_scores.append(vid_score)
        all_vid_labels.append(label)

        # Frame-level expansion
        frame_scores = np.repeat(seg_scores, frames_per_segment)
        all_frm_pred.extend(frame_scores.tolist())
        all_frm_gt.extend([label] * len(frame_scores))

        # Per-category tracking
        if cat_id not in by_cat:
            by_cat[cat_id] = {'scores': [], 'labels': []}
        by_cat[cat_id]['scores'].append(vid_score)
        by_cat[cat_id]['labels'].append(label)

    vid_scores = np.array(all_vid_scores)
    vid_labels = np.array(all_vid_labels)
    frm_pred   = np.array(all_frm_pred)
    frm_gt     = np.array(all_frm_gt)

    n_classes = len(np.unique(vid_labels))
    if n_classes < 2:
        return {
            'video_auc': 0.0, 'video_ap': 0.0,
            'frame_auc': 0.0, 'frame_ap': 0.0,
            'per_cat':   {},
        }

    video_auc = float(roc_auc_score(vid_labels, vid_scores))
    video_ap  = float(average_precision_score(vid_labels, vid_scores))

    fpr, tpr, _ = roc_curve(frm_gt, frm_pred)
    frame_auc   = float(auc(fpr, tpr))
    frame_ap    = float(average_precision_score(frm_gt, frm_pred))

    # Per-category AP (anomaly cat vs all normal)
    normal_scores = vid_scores[vid_labels == 0]
    cat_ap = {}
    for cid, data in by_cat.items():
        if cid == 0:
            continue  # skip normal
        cat_s = np.array(data['scores'])
        scores_bin = np.concatenate([cat_s, normal_scores])
        labels_bin = np.concatenate([np.ones(len(cat_s)), np.zeros(len(normal_scores))])
        if len(np.unique(labels_bin)) > 1:
            cat_ap[CAT_ID_TO_NAME.get(cid, str(cid))] = float(
                average_precision_score(labels_bin, scores_bin)
            )

    return {
        'video_auc': video_auc,
        'video_ap':  video_ap,
        'frame_auc': frame_auc,
        'frame_ap':  frame_ap,
        'per_cat':   cat_ap,
    }


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, args, is_best=False):
    ckpt = {
        'epoch':      epoch,
        'state_dict': model.state_dict(),
        'optimizer':  optimizer.state_dict(),
        'scheduler':  scheduler.state_dict(),
        'metrics':    metrics,
        'args':       vars(args),
        'training_schedule': training_schedule_name(args),
    }
    path = os.path.join(args.checkpoint_dir, f'epoch_{epoch:03d}.pt')
    torch.save(ckpt, path)
    if is_best:
        best_path = os.path.join(args.checkpoint_dir, 'best.pt')
        torch.save(ckpt, best_path)
        print(f"  -> Saved best checkpoint (video_AUC={metrics['video_auc']:.4f})")


def main(args):
    # Seed everything for reproducibility
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed: {args.seed}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------ Data
    print("\nLoading datasets...")
    train_loader, eval_loader, train_ds, eval_ds = build_dataloaders(
        train_feature_dir=args.train_dir,
        test_feature_dir=args.test_dir,
        train_list=args.train_list,
        test_list=args.test_list,
        num_segments=args.num_segments,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        use_val_split=not args.no_val_split,
        seed=args.seed,
        pin_memory=(device.type == 'cuda'),
    )
    if not args.no_val_split:
        print(f"[Eval] Using val split (20% of training data, kept separate from training)")
    else:
        print(f"[Eval] Using external test set")

    inferred_input_dim = getattr(train_ds, 'input_dim', None)
    if args.input_dim is None:
        if inferred_input_dim is None:
            raise ValueError("Failed to infer feature dimensionality from the training set")
        args.input_dim = inferred_input_dim

    eval_input_dim = getattr(eval_ds, 'input_dim', None)
    if eval_input_dim is not None and eval_input_dim != args.input_dim:
        raise ValueError(
            f"Feature dimension mismatch: train={args.input_dim}, eval={eval_input_dim}"
        )

    # ----------------------------------------------------------------- Model
    model = ViolenceDetector(
        input_dim=args.input_dim,
        num_classes=7,
        d_model=args.d_model,
        nhead=args.nhead,
        trn_layers=args.trn_layers,
    ).to(device)

    total, trainable = model.param_count()
    print(f"\nModel: {total/1e6:.2f}M total, {trainable/1e6:.2f}M trainable")

    # --------------------------------------------------------------- Loss/Opt
    criterion = ViolenceLoss(
        lambda_cls=args.lambda_cls,
        lambda_bnd=args.lambda_bnd,
        lambda_trn=args.lambda_trn,
        topk_ratio=args.topk_ratio,
        topk_cls=args.topk_cls,
    )
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[30, 60, 80], gamma=0.1
    )

    writer = SummaryWriter(log_dir=args.log_dir)

    # ----------------------------------------------------------- Resume/Start
    start_epoch = 1
    best_auc = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_auc = ckpt['metrics'].get('video_auc',
                    ckpt['metrics'].get('auc', 0.0))
        print(f"Resumed from epoch {ckpt['epoch']}, best AUC={best_auc:.4f}")

    schedule_name = training_schedule_name(args)
    best_epoch = 0
    best_metrics = {
        'video_auc': 0.0,
        'video_ap': 0.0,
        'frame_auc': 0.0,
        'frame_ap': 0.0,
        'per_cat': {},
    }

    print(f"\nStarting training for {args.epochs} epochs")
    print(f"Training schedule: {schedule_name}")
    print(f"Feature dimension: {args.input_dim}")
    print(f"Temporal sequence length (num_segments): {args.num_segments}")
    print(f"{'Ep':>4} {'Stg':>3} {'Loss':>7} {'MIL':>7} {'Cls':>7} "
          f"{'TRN':>7} {'Bnd':>7} {'vAUC':>7} {'fAUC':>7} {'Time':>7}")
    print("-" * 73)

    # In joint mode all losses are active from epoch 1 (no staged gating)
    stage_transition_epochs = {1} if args.joint else {1, 31, 61}

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        stage = 3 if args.joint else get_stage(epoch)

        if not args.joint and stage > args.max_stage:
            print(f"\nReached max_stage={args.max_stage}. Stopping early at epoch {epoch-1}.")
            break

        if epoch == start_epoch or epoch in stage_transition_epochs:
            n_active = set_stage_params(
                model,
                optimizer,
                stage,
                args.lr,
                reset_lr_on_stage3=(epoch in stage_transition_epochs),
            )
            label = "joint" if args.joint else f"Stage {stage}"
            print(f"\n[{label} — {n_active/1e6:.2f}M active params]\n")

        losses = train_one_epoch(
            model, train_loader, criterion, optimizer, device, stage, epoch
        )

        metrics = {'video_auc': 0.0, 'video_ap': 0.0,
                   'frame_auc': 0.0, 'frame_ap': 0.0, 'per_cat': {}}
        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            metrics = evaluate(model, eval_loader, device,
                               frames_per_segment=args.frames_per_segment,
                               stage=stage)

        scheduler.step()
        elapsed = time.time() - t0

        # Logging
        for k, v in losses.items():
            writer.add_scalar(f'Loss/{k}', v, epoch)
        writer.add_scalar('Eval/VideoAUC',  metrics['video_auc'], epoch)
        writer.add_scalar('Eval/VideoAP',   metrics['video_ap'],  epoch)
        writer.add_scalar('Eval/FrameAUC',  metrics['frame_auc'], epoch)
        writer.add_scalar('Eval/FrameAP',   metrics['frame_ap'],  epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        for cat, ap in metrics.get('per_cat', {}).items():
            writer.add_scalar(f'Eval/AP_{cat}', ap, epoch)

        is_best = metrics['video_auc'] > best_auc
        if is_best:
            best_auc = metrics['video_auc']
            best_epoch = epoch
            best_metrics = metrics.copy()

        if epoch % args.save_freq == 0 or epoch == args.epochs or is_best:
            save_checkpoint(model, optimizer, scheduler, epoch, metrics, args, is_best)

        print(f"{epoch:>4} {stage:>3} "
              f"{losses.get('total', 0):>7.4f} "
              f"{losses.get('mil',   0):>7.4f} "
              f"{losses.get('cls',   0):>7.4f} "
              f"{losses.get('trn',   0):>7.4f} "
              f"{losses.get('bnd',   0):>7.4f} "
              f"{metrics['video_auc']:>7.4f} "
              f"{metrics['frame_auc']:>7.4f} "
              f"{elapsed:>6.1f}s")

        # Print per-category AP every 10 epochs
        if epoch % 10 == 0 and metrics['per_cat']:
            cats_str = '  '.join(
                f"{c[:3]}={v:.3f}" for c, v in sorted(metrics['per_cat'].items())
            )
            print(f"       Per-cat AP: {cats_str}")

    writer.close()
    save_run_summary(args, best_epoch=best_epoch, best_metrics=best_metrics)
    print(f"\nTraining complete. Best video AUC: {best_auc:.4f}")
    print(f"Best checkpoint: {args.checkpoint_dir}/best.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Violence Event Detection Training')

    # Data
    parser.add_argument('--train_dir',  required=True)
    parser.add_argument('--test_dir',   default=None)
    parser.add_argument('--train_list', default='auto')
    parser.add_argument('--test_list',  default='auto')
    parser.add_argument('--no_val_split', action='store_true',
                        help='Use external test set instead of val split')
    parser.add_argument('--val_ratio',  type=float, default=0.2)

    # Model
    parser.add_argument('--num_segments',       type=int,   default=32)
    parser.add_argument('--input_dim',          type=int,   default=None,
                        help='Feature dimension; defaults to auto-detect from training data')
    parser.add_argument('--d_model',            type=int,   default=512)
    parser.add_argument('--nhead',              type=int,   default=8)
    parser.add_argument('--trn_layers',         type=int,   default=2)
    parser.add_argument('--frames_per_segment', type=int,   default=16)

    # Training
    parser.add_argument('--epochs',       type=int,   default=100)
    parser.add_argument('--batch_size',   type=int,   default=32)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers',  type=int,   default=4)
    parser.add_argument('--device',       choices=('auto', 'cpu', 'cuda', 'mps'), default='auto')

    # Loss weights
    parser.add_argument('--lambda_cls',    type=float, default=0.5)
    parser.add_argument('--lambda_bnd',    type=float, default=0.1)
    parser.add_argument('--lambda_trn',    type=float, default=1.0)
    parser.add_argument('--topk_ratio',    type=float, default=0.1)
    parser.add_argument('--topk_cls',      type=int,   default=5)

    # Logging / checkpointing
    parser.add_argument('--checkpoint_dir', default='/scratch/saichava/violence_detection/checkpoints')
    parser.add_argument('--log_dir',        default='/scratch/saichava/violence_detection/logs/tensorboard')
    parser.add_argument('--eval_freq',      type=int, default=5)
    parser.add_argument('--save_freq',      type=int, default=10)
    parser.add_argument('--resume',         default=None)
    parser.add_argument('--seed',           type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--max_stage',      type=int, default=3,
                        help='Stop training after this stage (1, 2, or 3)')
    parser.add_argument('--joint',          action='store_true',
                        help='Joint training: all 3 losses active from epoch 1 (no staged gating)')

    args = parser.parse_args()
    if not args.no_val_split and args.test_dir is None:
        args.test_dir = args.train_dir
    if args.no_val_split and not args.test_dir:
        raise ValueError("--test_dir is required when --no_val_split is used")
    main(args)
