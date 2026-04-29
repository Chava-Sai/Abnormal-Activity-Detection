import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import multiprocessing as mp
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

KINETICS_MEAN = [0.45, 0.45, 0.45]
KINETICS_STD  = [0.225, 0.225, 0.225]
SEG_LEN = 16

FRAME_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(KINETICS_MEAN, KINETICS_STD),
])

def build_i3d(device):
    from pytorchvideo.models.hub import i3d_r50
    model = i3d_r50(pretrained=True)

    # Proper pooled 2048-dim output
    model.blocks[-1].proj = nn.Identity()
    model.blocks[-1].output_pool = nn.AdaptiveAvgPool3d(1)

    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def extract_video(frame_dir, model, device):
    frame_paths = sorted(
        glob.glob(os.path.join(frame_dir, "*.jpg")) +
        glob.glob(os.path.join(frame_dir, "*.png"))
    )

    if len(frame_paths) < SEG_LEN:
        return None

    features = []

    for start in range(0, len(frame_paths) - SEG_LEN + 1, SEG_LEN):
        clip_paths = frame_paths[start:start + SEG_LEN]

        frames = []
        for fp in clip_paths:
            img = Image.open(fp).convert("RGB")
            frames.append(FRAME_TRANSFORM(img))

        clip = torch.stack(frames, dim=0)
        clip = clip.permute(1, 0, 2, 3).unsqueeze(0).to(device)

        feat = model(clip)              # (1, 2048)
        feat = feat.flatten(1)          # safety: force (1, 2048)

        features.append(feat.squeeze(0).cpu().numpy())

    return np.stack(features, axis=0).astype(np.float32)

def worker(gpu_id, video_dirs, out_dir, overwrite):
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    print(f"[GPU {gpu_id}] Loading I3D...")
    model = build_i3d(device)
    print(f"[GPU {gpu_id}] Loaded model. Processing {len(video_dirs)} videos.")

    success, skipped, already_done = 0, 0, 0

    for vdir in tqdm(video_dirs, desc=f"GPU {gpu_id}", position=gpu_id):
        vid_name = os.path.basename(vdir)
        out_path = os.path.join(out_dir, f"{vid_name}.npy")

        if os.path.exists(out_path) and not overwrite:
            already_done += 1
            continue

        feats = extract_video(vdir, model, device)

        if feats is None:
            skipped += 1
            continue

        np.save(out_path, feats)
        success += 1

    print(f"[GPU {gpu_id}] Done.")
    print(f"[GPU {gpu_id}] Extracted: {success}")
    print(f"[GPU {gpu_id}] Skipped: {skipped}")
    print(f"[GPU {gpu_id}] Already done: {already_done}")

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    video_dirs = sorted([
        d for d in glob.glob(os.path.join(args.frames_root, "*"))
        if os.path.isdir(d)
    ])

    if not video_dirs:
        raise FileNotFoundError(f"No video directories found in {args.frames_root}")

    print(f"Found {len(video_dirs)} video directories")

    gpus = args.gpus
    chunks = [video_dirs[i::len(gpus)] for i in range(len(gpus))]

    mp.set_start_method("spawn", force=True)

    processes = []
    for gpu_id, chunk in zip(gpus, chunks):
        p = mp.Process(
            target=worker,
            args=(gpu_id, chunk, args.out_dir, args.overwrite)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    files = sorted(glob.glob(os.path.join(args.out_dir, "*.npy")))
    print(f"\nTotal output files: {len(files)}")

    if files:
        arr = np.load(files[0])
        print(f"Sanity check: {os.path.basename(files[0])}, shape={arr.shape}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frames_root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    main(args)