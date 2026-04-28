import os
import glob
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from model import ViolenceDetector  # same folder

def load_model(checkpoint_path, device, input_dim=1024):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = ViolenceDetector(input_dim=input_dim, num_classes=7,
                             d_model=512, nhead=8, trn_layers=2)
    # Remove classifier weights if they cause mismatch (we only need trn_scores)
    state_dict = {k:v for k,v in ckpt['state_dict'].items() if not k.startswith('classifier.')}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --- CHANGE THESE PATHS ---
    checkpoint = "/projectnb/cs585/students/shrishty/shanghaitech/best.pt"
    feat_dir   = "/projectnb/cs585/students/shrishty/shanghaitech/shanghaitech_features"
    mask_dir   = "/projectnb/cs585/students/shrishty/shanghaitech/archive/test_frame_mask"
    frames_per_segment = 16

    model = load_model(checkpoint, device, input_dim=1024)

    all_scores = []
    all_labels = []
    for npz_file in glob.glob(os.path.join(feat_dir, "*.npz")):
        vid = os.path.splitext(os.path.basename(npz_file))[0]
        data = np.load(npz_file)
        feats = data['features']  # (T, 1024)
        x = torch.from_numpy(feats).float().unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)
            seg_scores = out['trn_scores'].squeeze(0).cpu().numpy()
        # Load ground truth mask (frame-level)
        mask_path = os.path.join(mask_dir, f"{vid}.npy")
        if not os.path.exists(mask_path):
            continue
        gt = np.load(mask_path)
        total_frames = len(gt)
        # Expand segment scores to frames
        frame_scores = np.repeat(seg_scores, frames_per_segment)[:total_frames]
        all_scores.extend(frame_scores)
        all_labels.extend(gt)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    auc = roc_auc_score(all_labels, all_scores)
    ap = average_precision_score(all_labels, all_scores)
    print(f"\nShanghaiTech Evaluation Results")
    print(f"Total frames: {len(all_labels):,}")
    print(f"Anomalous frames: {all_labels.sum():,} ({100*all_labels.mean():.1f}%)")
    print(f"Frame-level AUC: {auc:.4f}")
    print(f"Frame-level AP : {ap:.4f}")
    import json
results = {
    'total_frames': int(len(all_labels)),
    'anomalous_frames': int(all_labels.sum()),
    'anomalous_percent': float(100 * all_labels.mean()),
    'frame_auc': float(auc),
    'frame_ap': float(ap)
}
with open('shanghai_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved to shanghai_results.json")

if __name__ == "__main__":
    main()