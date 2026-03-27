import os, sys, argparse, numpy as np, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from model import ViolenceDetector
from losses import mil_ranking_loss as MILRankingLoss
from xd_dataset import make_xd_train_val

def mil_collate(batch):
    anom   = [b for b in batch if b['label'].item() == 1]
    normal = [b for b in batch if b['label'].item() == 0]
    n = min(len(anom), len(normal))
    if n == 0: return None
    anom = anom[:n]; normal = normal[:n]
    return {
        'anom_feat':  torch.stack([b['features'] for b in anom]),
        'norm_feat':  torch.stack([b['features'] for b in normal]),
        'anom_cat':   torch.stack([b['cat_id']   for b in anom]),
    }

def evaluate(model, dataset, device):
    model.eval(); scores, labels = [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            out  = model(item['features'].unsqueeze(0).to(device))
            scores.append(float(out['trn_scores'].squeeze(0).max()))
            labels.append(int(item['label'].item()))
    return roc_auc_score(labels, scores)

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    train_ds, val_ds = make_xd_train_val(
        args.rgb_dir, args.flow_dir,
        num_segments=args.num_segments, val_ratio=0.2, seed=args.seed)

    loader = DataLoader(train_ds, batch_size=args.batch_size,
                        shuffle=True, collate_fn=mil_collate,
                        num_workers=2, drop_last=True)

    model = ViolenceDetector(input_dim=2048, num_classes=7,
                d_model=512, nhead=8, trn_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    pass  # mil_fn defined below
    ce_fn  = nn.CrossEntropyLoss()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_auc = 0.0

    for epoch in range(1, args.epochs + 1):
        use_cls = epoch > 30
        use_trn = epoch > 60
        model.train(); total_loss = 0.0; nb = 0

        for batch in loader:
            if batch is None: continue
            af = batch['anom_feat'].to(device)
            nf = batch['norm_feat'].to(device)
            ac = batch['anom_cat'].to(device)
            out_a = model(af); out_n = model(nf)

            loss = MILRankingLoss(out_a["mil_scores"], out_n["mil_scores"])
            if use_cls:
                loss += 0.5 * ce_fn(out_a['cls_logits'].mean(dim=1), ac)
            if use_trn:
                loss += 0.5 * MILRankingLoss(out_a["trn_scores"], out_n["trn_scores"])
                smooth = torch.mean(torch.abs(out_a['trn_scores'][:,1:] - out_a['trn_scores'][:,:-1]))
                loss += 0.1 * smooth
                if 'boundary_scores' in out_a:
                    loss += 0.1 * out_a['boundary'].mean()

            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item(); nb += 1

        scheduler.step()

        if epoch % 5 == 0 or epoch == args.epochs:
            val_auc = evaluate(model, val_ds, device)
            stage = 3 if use_trn else (2 if use_cls else 1)
            print(f'Epoch {epoch:3d} | Stage {stage} | loss={total_loss/max(nb,1):.4f} | vAUC={val_auc:.4f}', flush=True)
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                            'video_auc': val_auc},
                           os.path.join(args.checkpoint_dir, 'best.pt'))

    print(f'Training complete. Best XD-Violence vAUC: {best_auc:.4f}')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--rgb_dir',        required=True)
    p.add_argument('--flow_dir',       required=True)
    p.add_argument('--checkpoint_dir', required=True)
    p.add_argument('--log_dir',        required=True)
    p.add_argument('--epochs',   type=int,   default=100)
    p.add_argument('--batch_size',type=int,  default=32)
    p.add_argument('--num_segments',type=int,default=32)
    p.add_argument('--lr',       type=float, default=1e-3)
    p.add_argument('--seed',     type=int,   default=42)
    train(p.parse_args())
