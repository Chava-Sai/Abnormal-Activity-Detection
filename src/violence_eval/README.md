cat > README.md <<'EOF'
# ShanghaiTech Violence Evaluation

Zero-shot ShanghaiTech robustness evaluation using a UCF-Crime-trained violence detector.

## Contents

- `evaluate_shanghaitech.py` -- evaluates frame-level AUC/AP on ShanghaiTech
- `extract_shanghai_test_features.py` -- extracts I3D features from ShanghaiTech test frames
- `model.py` -- violence detector model definition
- `run_shanghai_eval.sh` -- example evaluation script

## Results (ShanghaiTech Zero-Shot)



- Videos evaluated: 107  
- Total frames: 40,791  
- Anomalous frames: 17,326 (42.5%)

**Frame-level Performance**
- AUC: **0.4045**
- AP: **0.3730**
Performance is expected to be suboptimal/aggregate. 

## Notes

Large files are excluded from GitHub:

- model checkpoints: `*.pt`, `*.pth`
- extracted features: `*.npy`, `*.npz`
- ShanghaiTech frames/features folders
- logs/cache files

## Example

```bash
python evaluate_shanghaitech.py \
  --checkpoint violence_detector_best.pt \
  --feat_dir shanghai_test_features \
  --mask_dir /path/to/test_frame_mask \
  --num_segments 32 \
  --frames_per_segment 16 \
  --gpus 0 \
  --output_json shanghai_eval_results.json
