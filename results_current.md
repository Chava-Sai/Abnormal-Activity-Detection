# Current Results Summary

Last updated: 2026-04-24

## Overall Assessment

- `--num_segments` in the current codebase is a temporal sequence-length setting, not frame-level segment size.
- On the internal validation split, `Progressive, T=32` is best by video AUC.
- On the completed local held-out UCF 5cat+Normal test subset, `Joint, T=32` is clearly the best current checkpoint.
- For the progressive family, the sequence-length ranking is stable: `T=32 > T=16 > T=8`.
- Binary anomaly detection is usable but only moderate at the final calibrated operating point.
- Temporal localization is still weak and is currently the main weakness of the project.

## Data Basis

- Training features: `1092` videos
  - Abuse `48`, Explosion `29`, Fighting `45`, Robbery `145`, Shooting `27`, Normal `798`
- Local held-out test subset: `206` videos
  - Abuse `2`, Explosion `21`, Fighting `5`, Robbery `5`, Shooting `23`, Normal `150`
- Feature format: pooled I3D, `1024`-D, extracted with `16` frames per segment

## 1. Validation Results

### Step 1: Progressive vs Joint

| Setting | Video AUC | Video AP | Frame AUC | Frame AP |
|---|---:|---:|---:|---:|
| Progressive, `T=32` | 0.9182 | 0.8129 | 0.8284 | 0.6739 |
| Joint, `T=32` | 0.8987 | 0.7386 | 0.9039 | 0.7501 |

Conclusion:

- Validation favors `Progressive` on video-level ranking.
- Validation favors `Joint` on frame-level metrics.

### Step 2: Temporal Sequence-Length Ablation

| Setting | Video AUC | Video AP | Frame AUC | Frame AP |
|---|---:|---:|---:|---:|
| Progressive, `T=8` | 0.9059 | 0.7975 | 0.8386 | 0.6831 |
| Progressive, `T=16` | 0.9001 | 0.7735 | 0.8348 | 0.6757 |
| Progressive, `T=32` | 0.9182 | 0.8129 | 0.8284 | 0.6739 |

Conclusion:

- On validation, `T=32` is the best sequence length.
- This should be reported as a sequence-length ablation, not a segment-size ablation.

## 2. Held-out Local Test Results

| Setting | Video AUC | Video AP | Frame AUC | Frame AP |
|---|---:|---:|---:|---:|
| Progressive, `T=32` | 0.7157 | 0.4926 | 0.5254 | 0.3015 |
| Joint, `T=32` | 0.8411 | 0.6618 | 0.8391 | 0.6843 |
| Progressive, `T=8` | 0.5014 | 0.2967 | 0.5025 | 0.2727 |
| Progressive, `T=16` | 0.6236 | 0.3481 | 0.5229 | 0.2960 |
| Progressive, `T=32` | 0.7157 | 0.4926 | 0.5254 | 0.3015 |

Conclusion:

- The held-out test subset reverses the validation conclusion for Step 1.
- `Joint, T=32` is the strongest current model in this repository.
- The progressive model does not transfer well from validation to held-out test.
- Within the progressive family, longer temporal context helps consistently.

## 3. Final Operating Point

Threshold calibration was fixed to use the same refined-score space as the real prediction pipeline:

- smoothing + boundary refinement + max refined score

Adopted threshold:

- `0.68433` from validation `best_macro_f1`

Held-out local test binary result at that threshold:

| Threshold | Accuracy | Balanced Acc | Precision | Recall | Specificity | F1 |
|---|---:|---:|---:|---:|---:|---:|
| 0.68433 | 0.7136 | 0.6411 | 0.4737 | 0.4821 | 0.8000 | 0.4779 |

Confusion matrix:

- TP `27`
- TN `120`
- FP `30`
- FN `29`

Interpretation:

- Binary detection is acceptable but not strong.
- The model is reasonably conservative on normal videos (`specificity=0.8000`).
- Recall remains limited, so many anomalous videos are still missed.

## 4. Error Analysis

Expanded error taxonomy on the training split:

- False positives: `203`
- False negatives: `123`

Main failure pattern:

- all `203` false positives were predicted as `Robbery`

Interpretation:

- The model shows strong class collapse toward `Robbery`.
- This likely explains part of the weak generalization of the progressive checkpoint.

## 5. Boundary / Localization Result

Strict temporal boundary evaluation on the local held-out test subset:

| Metric | Value |
|---|---:|
| GT anomalous videos | 56 |
| Matched peak pairs | 27 |
| Normal-video false alarms | 30 |
| Mean IoU | 0.1875 |
| Median IoU | 0.0000 |
| IoU >= 0.5 | 8 / 56 |

Boundary bias:

- mean signed start error: `-173.59` frames, early
- mean signed end error: `200.74` frames, late

Interpretation:

- Localization is currently weak.
- The model often starts too early and ends too late.
- Detection quality is stronger than localization quality.

## Final Takeaways

1. The current best result is `Joint, T=32` on the held-out local test subset.
2. `T=32` is the strongest temporal sequence length among the models that were actually tested.
3. The internal validation split is not reliable enough to choose the final checkpoint by itself.
4. The main current weaknesses are:
   - `Robbery` class collapse
   - moderate binary recall
   - weak temporal localization
5. The current Step 2 result should be described honestly as a temporal sequence-length ablation.
