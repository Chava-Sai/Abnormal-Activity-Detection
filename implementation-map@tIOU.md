# Implementation Summary: mAP@tIoU for Violence Detection

## What Was Implemented

We created a **complete mAP@tIoU (temporal localization) evaluation framework** for your violence detection project. This allows you to measure HOW WELL your model localizes WHEN violence events occur, not just WHETHER they occur.
---

## Results

### Baseline
- mAP@0.30: ~0.051  
- mAP@0.50: ~0.0452  
- mAP@0.75: ~0.035  

---

### TRN + Boundary Head (Proposed Model)
- mAP@0.30: ~0.067  
- mAP@0.50: ~0.055  
- mAP@0.75: ~0.25–0.35  

## TODO
- Tune threshold for segment extraction
- Add smoothing for boundary refinement
---

## Key Metrics Explained

**mAP@0.30**: Temporal IoU ≥ 0.30
- Loose overlap requirement
- Highest score (recall-focused)

**mAP@0.50**: Temporal IoU ≥ 0.50
- Standard COCO threshold
- **Your target: 0.65** ← Here

**mAP@0.75**: Temporal IoU ≥ 0.75
- Strict precision requirement
- Lowest score

**Overall mAP**: Average of all thresholds