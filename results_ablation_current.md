# Current Ablation Summary

## Step 1: Progressive vs Joint Training
| training_schedule | num_segments | seed | best_video_auc |
|---|---:|---:|---:|
| progressive | 32 | 42 | 0.9371 |
| joint | 32 | 42 | 0.9338 |

Conclusion: progressive training outperformed joint training by 0.0033 AUC on the UCF-Crime validation split.

## Step 2: Temporal Sequence-Length Ablation
| training_schedule | num_segments | seed | best_video_auc |
|---|---:|---:|---:|
| progressive | 8  | 42 | 0.9493 |
| progressive | 16 | 42 | 0.9330 |
| progressive | 32 | 42 | 0.9371 |

Conclusion: T=8 performed best on the current UCF-Crime validation split.

## Important Note
This is a temporal sequence-length ablation over cached I3D features (`num_segments=8/16/32`), not a true frame-level segment-size ablation. A true 8/16/32 frames-per-segment comparison would require re-extracting features from raw videos.
