import os
import glob
import numpy as np
import torch
from pytorchvideo.models.hub import i3d_r50
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

def extract_video(video_dir, output_dir, device):
    frame_paths = sorted(glob.glob(os.path.join(video_dir, "*.jpg")) +
                         glob.glob(os.path.join(video_dir, "*.png")))
    if len(frame_paths) < 16:
        return
    # Segment into 16-frame non-overlapping clips
    seg_len = 16
    features = []
    for start in range(0, len(frame_paths) - seg_len + 1, seg_len):
        clip = []
        for fpath in frame_paths[start:start+seg_len]:
            img = Image.open(fpath).convert('RGB')
            img = transforms.Resize(256)(img)
            img = transforms.CenterCrop(224)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])(img)
            clip.append(img)
        clip = torch.stack(clip, dim=0).permute(1,0,2,3).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(clip).squeeze().cpu().numpy()
        features.append(feat)
    if len(features) > 0:
        feats = np.stack(features, axis=0)  # (T, 1024)
        vid_name = os.path.basename(video_dir)
        np.savez(os.path.join(output_dir, f"{vid_name}.npz"), features=feats)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = i3d_r50(pretrained=True).to(device)
    model.head = torch.nn.Identity()  # remove classifier, keep pooled features (1024-dim)
    model.eval()

    frames_root = "/projectnb/cs585/students/shrishty/shanghaitech/shanghaitech_features"
    out_dir = "/projectnb/cs585/students/shrishty/shanghaitech_features"
    os.makedirs(out_dir, exist_ok=True)

    video_dirs = glob.glob(os.path.join(frames_root, "*"))
    for vdir in tqdm(video_dirs):
        extract_video(vdir, out_dir, device)