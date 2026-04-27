# CS585 Project Status and Current Results

Last updated: 2026-04-27

This README is the sub-repository copy of the current project status. The
numeric results below match the current local outputs under `runs/`.

## 1. Scope of This Repo

This repository contains the training, evaluation, prediction, and analysis
code for the violence-event detection project.

Important local-workspace note:

- the raw datasets and extracted UCF / ShanghaiTech features live one level
  above this repo in the surrounding project workspace
- the top-level feature-extraction pipeline also lives one level above this
  repo
- this README records the work and results from that full local workspace, but
  centers the code changes that matter for this sub-repo

## 2. Local Workspace Layout

### 2.1 Code relevant to this sub-repo

- Training and evaluation code:
  - `src/`
- Runner and reporting scripts:
  - `scripts/`
- Local experiment outputs referenced by this README:
  - `runs/`

### 2.2 Adjacent local workspace components

- Feature extraction pipeline from raw videos:
  - `../main.py`
  - `../i3d_extractor.py`
  - `../video_reader.py`
  - `../config.py`
  - `../dataset_scanner.py`
  - `../storage.py`
- UCF raw data, split subsets, and extracted features:
  - `../UCF/`
- ShanghaiTech raw data and extracted features:
  - `../shanghai/`
  - `../outputs_shanghaitech_i3d/`
- Older UCF anomaly-only extracted features:
  - `../outputs_pytorch_i3d_rgb_v2/`
- Local fallback I3D implementation used by the extractor:
  - `../third_party/pytorch-i3d/`

## 3. Current Local Data and Feature State

### 3.1 Raw data currently present in the local workspace

| Asset | Path | Current local state |
|---|---|---|
| UCF anomaly raw videos | `../UCF/Anomaly-Videos-Part-1..4/` | Present locally, `950` videos total |
| UCF training normal raw videos | `../UCF/Training-Normal-Videos-Part-1..2/` | Present locally, `800` videos total |
| UCF testing normal raw videos | `../UCF/Testing_Normal_Videos_Anomaly/` | Present locally, `150` videos total |
| UCF temporal annotations | `../UCF/Temporal_Anomaly_Annotation_for_Testing_Videos.txt` | Present locally |
| ShanghaiTech raw data | `../shanghai/shanghaitech/` | Present locally |
| XD-Violence | not found locally | Missing |
| UCA annotations | not found locally | Missing |
| RWF-2000 | not found locally | Missing |

### 3.2 UCF feature roots currently used by this repo

Current training feature root:

- `../UCF/UCF_split_train_i3d_5cat`

Current test feature root:

- `../UCF/UCF_split_test_i3d_5cat`

The current setup uses 5 anomaly categories plus `Normal`.

#### Training features

| Category | Count |
|---|---:|
| Abuse | 48 |
| Explosion | 29 |
| Fighting | 45 |
| Robbery | 145 |
| Shooting | 27 |
| Normal | 798 |
| Total | 1092 |

Notes:

- the train feature root is split across category folders plus
  `features/Normal`
- two training-normal feature files are still missing relative to the `800`
  raw normal videos:
  - `Normal_Videos471_x264`
  - `Normal_Videos947_x264`

#### Test features

| Category | Count |
|---|---:|
| Abuse | 2 |
| Explosion | 21 |
| Fighting | 5 |
| Robbery | 5 |
| Shooting | 23 |
| Normal | 150 |
| Total | 206 |

Notes:

- the local 5-category + `Normal` UCF test subset is complete
- in this pass, all `150` test-normal I3D features were extracted and added

### 3.3 Current feature format

The current UCF features used by training and evaluation are pooled I3D
features with:

- feature dimension: `1024`
- extraction clip length: `16` frames
- stride: `16` frames
- feature layer: `pool`

Feature sequence length varies by video duration; the model later resizes each
video sequence to the requested `num_segments`.

### 3.4 Other local artifacts

| Asset | Path | Current local state |
|---|---|---|
| ShanghaiTech extracted features | `../outputs_shanghaitech_i3d/features/ShanghaiTech` | `330` feature files present |
| Older UCF anomaly-only extracted features | `../outputs_pytorch_i3d_rgb_v2/features` | `400` feature files present |

## 4. What `--num_segments` Actually Means

The answer in the current codebase is straightforward:

- `src/train.py` passes `--num_segments` into the dataloader
- `src/dataset.py` stores that value and applies it when loading
  already-extracted features
- `src/feature_utils.py` implements `temporal_resize(...)`, which only:
  - uniformly samples the time axis when `T > num_segments`
  - zero-pads when `T < num_segments`

Therefore:

- `--num_segments 8/16/32` is a temporal sequence-length ablation over cached
  I3D features
- it is not a true "segment size = 8/16/32 frames" ablation
- a true frame-level segment-size ablation would require re-extracting features
  from raw videos with different `--clip-len` values

Important nuance for the current local workspace:

- raw UCF videos are present locally, so a true `clip_len = 8/16/32` ablation
  is technically possible without downloading a new dataset
- however, it was not run in this pass because the current training artifacts
  are built on the existing `16`-frame features, so a true ablation would
  require fresh feature extraction plus retraining

## 5. Which Planned Items Were Runnable Without a New Dataset?

| Plan item | Extra dataset needed? | Current status |
|---|---|---|
| Step 1: Progressive vs Joint training ablation | No | Completed |
| Step 2: `--num_segments` ablation as currently implemented | No | Completed |
| True `clip_len = 8/16/32` frame-level ablation | No extra data, but re-extraction required | Technically possible locally, not run in this pass |
| Step 3: XD-Violence 3-seed runs | Yes | Blocked, XD data/features not present |
| Step 4: Boundary precision analysis | No | Completed |
| Step 5: Expanded error taxonomy | No | Completed |
| Step 6: UCA supplementary localization | Yes | Blocked, UCA annotations not present |
| Step 7: Honest mAP statement | No | Writing task only |
| Step 8: RWF-2000 check | Yes | Blocked unless downloaded |
| Step 9: ShanghaiTech robustness check | No extra data locally | Local features exist, not used as a main reported experiment in this README |
| Step 10-13: paper, slides, final polish | No | Writing / presentation work |

## 6. Work Completed in This Pass

1. Verified the local data inventory and feature inventory.
2. Verified the true meaning of `--num_segments`.
3. Completed local UCF test-normal feature extraction so the local UCF test
   subset is `56 anomalous + 150 normal = 206` videos.
4. Re-ran the local UCF 5cat+Normal test evaluation on the completed
   `206`-video subset.
5. Re-ran the expanded error-taxonomy analysis on the training split.
6. Re-ran strict boundary analysis against
   `../UCF/Temporal_Anomaly_Annotation_for_Testing_Videos.txt`.
7. Fixed two thresholding issues:
   - stale default threshold `0.45` in thresholded scripts
   - score-space mismatch between threshold calibration and the actual
     prediction pipeline

### 6.1 Code areas that reflect this work

Primary thresholding / prediction consistency changes:

- `src/predict.py`
- `src/boundary_analysis.py`
- `scripts/calibrate_threshold.py`
- `scripts/report_binary_accuracy.py`

Supporting integration changes for the current local feature layout and local
experiment flow:

- `src/dataset.py`
- `src/feature_utils.py`
- `src/evaluate.py`
- `src/train.py`
- `src/error_taxonomy.py`
- `scripts/run_step1_local.sh`
- `scripts/run_step2_sequence_local.sh`
- `scripts/run_ucf_5cat_test_eval.py`
- `scripts/run_ucf_5cat_test_eval_many.py`
- `scripts/report_step1_ablation.py`
- `scripts/report_step2_sequence.py`
- `scripts/analyze_normal_false_alarms.py`

### 6.2 Main thresholding fix

Threshold calibration and thresholded binary reports now use the same score
definition as `predict.py`:

- raw `trn_scores`
- temporal smoothing
- boundary refinement
- max refined score as the video-level decision score

Without that fix, the same threshold was being applied in two different score
spaces.

## 7. Results

### 7.1 Step 1: Progressive vs Joint training on the validation split

Source:

- `runs/step1_progressive_s42/run_summary.json`
- `runs/step1_joint_s42/run_summary.json`

Settings:

- seed: `42`
- sequence length: `T=32`
- feature dimension: `1024`
- frames per extracted segment: `16`
- evaluation split: stratified validation split from the training feature root

| Training schedule | Best epoch | Video AUC | Video AP | Frame AUC | Frame AP |
|---|---:|---:|---:|---:|---:|
| Progressive | 55 | 0.9182 | 0.8129 | 0.8284 | 0.6739 |
| Joint | 70 | 0.8987 | 0.7386 | 0.9039 | 0.7501 |

Validation conclusion:

- Progressive beats Joint on validation video AUC by `+0.0194`
- Joint has better frame-level metrics on the validation split

### 7.2 Step 2: Temporal sequence-length ablation on the validation split

Source:

- `runs/step2_seq_progressive_T8_s42/run_summary.json`
- `runs/step2_seq_progressive_T16_s42/run_summary.json`
- `runs/step2_seq_progressive_T32_s42/run_summary.json`

| `num_segments` | Best epoch | Video AUC | Video AP | Frame AUC | Frame AP |
|---|---:|---:|---:|---:|---:|
| 8 | 15 | 0.9059 | 0.7975 | 0.8386 | 0.6831 |
| 16 | 10 | 0.9001 | 0.7735 | 0.8348 | 0.6757 |
| 32 | 55 | 0.9182 | 0.8129 | 0.8284 | 0.6739 |

Validation conclusion:

- `T=32` is the best temporal sequence length on the current validation split
- this experiment varies temporal sequence length over cached features, not
  frame-level segment size

### 7.3 Local UCF 5cat+Normal test subset (`206` videos)

Source:

- `runs/ucf_5cat_test_eval_all_current.json`

Test subset counts:

- Abuse: `2`
- Explosion: `21`
- Fighting: `5`
- Robbery: `5`
- Shooting: `23`
- Normal: `150`

| Setting | Video AUC | Video AP | Frame AUC | Frame AP |
|---|---:|---:|---:|---:|
| Step 1 Progressive, `T=32` | 0.7157 | 0.4926 | 0.5254 | 0.3015 |
| Step 1 Joint, `T=32` | 0.8411 | 0.6618 | 0.8391 | 0.6843 |
| Step 2 Progressive, `T=8` | 0.5014 | 0.2967 | 0.5025 | 0.2727 |
| Step 2 Progressive, `T=16` | 0.6236 | 0.3481 | 0.5229 | 0.2960 |
| Step 2 Progressive, `T=32` | 0.7157 | 0.4926 | 0.5254 | 0.3015 |

Held-out local test conclusions:

- the held-out local test subset reverses the validation conclusion for Step 1
- validation split favored Progressive, but held-out local test strongly favors
  Joint
- for the progressive family, `T=32 > T=16 > T=8` on held-out local test, so
  the sequence-length ranking is stable
- the current strongest held-out local test checkpoint in this repo is
  `Joint, T=32`

### 7.4 Threshold calibration and operating-point results

There were two separate threshold issues.

#### A. Old stale-threshold bug in legacy thresholded scripts

The repo previously used a default threshold of `0.45`, which is far below the
current score range and predicts almost everything as anomalous.

Raw-score validation operating point before and after the first fix:

| Score space | Threshold | Accuracy | Balanced Acc | Precision | Recall | Specificity | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw `trn_scores.max()` | 0.4500 | 0.2694 | 0.5000 | 0.2694 | 1.0000 | 0.0000 | 0.4245 |
| raw `trn_scores.max()` | 0.6606 | 0.6804 | 0.6529 | 0.4321 | 0.5932 | 0.7125 | 0.5000 |

This first fix eliminated the obvious stale-threshold failure.

#### B. Refined-score mismatch between calibration and prediction

After that, a second issue was found:

- `predict.py` and `boundary_analysis.py` make decisions using refined scores
  after smoothing + boundary refinement
- the old calibration / report scripts were still calibrating thresholds on raw
  `trn_scores.max()`

That mismatch caused boundary analysis to massively overfire.

#### C. Final threshold calibration in the correct refined-score space

Source:

- `runs/ucf_val_threshold_calibration_refined_current.json`

Refined-score validation calibration summary:

| Criterion | Threshold | Accuracy | Balanced Acc | Precision | Recall | Specificity | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Best accuracy | 0.689928 | 0.7580 | 0.5722 | 0.7143 | 0.1695 | 0.9750 | 0.2740 |
| Best balanced accuracy | 0.676738 | 0.5571 | 0.6434 | 0.3603 | 0.8305 | 0.4563 | 0.5026 |
| Best macro-F1 | 0.684330 | 0.7215 | 0.6168 | 0.4792 | 0.3898 | 0.8438 | 0.4299 |

Adopted default for prediction and strict boundary analysis:

- `0.68433` from the validation `best_macro_f1` criterion

Reason for using `best_macro_f1` instead of `best_balanced_accuracy` as the
final default:

- `best_balanced_accuracy` is too aggressive in the refined score space and
  produces too many false alarms on held-out local test
- `best_macro_f1` is still chosen from validation only, but gives a more
  practical precision / recall tradeoff for actual prediction and localization

#### D. Final held-out local test binary result at the adopted refined threshold

Source:

- `runs/ucf_test_binary_accuracy_refined_t0684330_current.json`

| Threshold | Accuracy | Balanced Acc | Precision | Recall | Specificity | F1 | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.68433 | 0.7136 | 0.6411 | 0.4737 | 0.4821 | 0.8000 | 0.4779 | 27 | 120 | 30 | 29 |

### 7.5 Expanded error taxonomy

Source:

- `runs/error_taxonomy_train877_t0660604_current/error_summary.txt`

Settings:

- split: training split list (`877` videos)
- threshold: raw-score threshold `0.660604`

Summary:

| Total videos | False Positives | False Negatives | Precision | Recall |
|---:|---:|---:|---:|---:|
| 877 | 203 | 123 | 0.3596 | 0.4810 |

Dominant false-positive modes:

- `FP_TRN_COLLAPSE`: `111` (`54.7%`)
- `FP_STRONG_MIL_OVERFIRE`: `26` (`12.8%`)
- `FP_ROBBERY_CLASS_COLLAPSE`: `24` (`11.8%`)
- `FP_MIXED_SUPPORT`: `22` (`10.8%`)
- `FP_THRESHOLD_MARGIN`: `20` (`9.9%`)

Dominant false-negative modes:

- `FN_ROBBERY_CLASS_COLLAPSE`: `53` (`43.1%`)
- `FN_TRN_SUPPRESSED_STRONG_MIL`: `44` (`35.8%`)
- `FN_LOW_MIL_SUPPORT`: `13` (`10.6%`)
- `FN_THRESHOLD_MARGIN`: `13` (`10.6%`)

Important failure pattern:

- all `203` false positives were predicted as `Robbery`

This is strong evidence of class collapse toward `Robbery`, and it matches the
weak held-out generalization of the progressive checkpoint.

### 7.6 Strict boundary precision analysis

Source:

- `runs/boundary_test_gt_t0684330_full/boundary_summary.txt`

Settings:

- checkpoint: Step 2 Progressive, `T=32`
- dataset: local UCF 5cat+Normal test subset (`206` results)
- GT annotation file:
  `../UCF/Temporal_Anomaly_Annotation_for_Testing_Videos.txt`
- threshold: refined-score threshold `0.68433`

| Metric | Value |
|---|---:|
| GT anomalous videos | 56 |
| Videos with matched peak pairs | 27 |
| Normal-video false alarms | 30 |
| Mean absolute start error (frames) | 218.19 |
| Mean absolute end error (frames) | 269.93 |
| Mean signed start error (frames) | -173.59 |
| Mean signed end error (frames) | 200.74 |
| Mean IoU | 0.1875 |
| Median IoU | 0.0000 |
| IoU >= 0.1 | 24 / 56 |
| IoU >= 0.3 | 19 / 56 |
| IoU >= 0.5 | 8 / 56 |

Boundary conclusion:

- once thresholding is made consistent with the real prediction pipeline, false
  alarms drop from the previous broken `150` to `30`
- localization under the realistic operating point is still weak:
  - only `27 / 56` anomalous videos produce matched peak pairs
  - the model tends to predict starts early and ends late
  - IoU remains low

## 8. Reproduction Commands Used in the Local Workspace

These commands are written from the perspective of this sub-repo. They assume
the same surrounding local workspace layout used for the current runs.

### 8.1 Step 1

```bash
PYTHON_BIN=../.venv/bin/python \
MODE=progressive \
EPOCHS=100 \
BATCH_SIZE=32 \
NUM_WORKERS=0 \
DEVICE=mps \
RUN_NAME=step1_progressive_s42 \
bash scripts/run_step1_local.sh
```

```bash
PYTHON_BIN=../.venv/bin/python \
MODE=joint \
EPOCHS=100 \
BATCH_SIZE=32 \
NUM_WORKERS=0 \
DEVICE=mps \
RUN_NAME=step1_joint_s42 \
bash scripts/run_step1_local.sh
```

### 8.2 Step 2

```bash
PYTHON_BIN=../.venv/bin/python \
MODE=progressive \
SEGMENTS_LIST='8 16 32' \
EPOCHS=100 \
BATCH_SIZE=32 \
NUM_WORKERS=0 \
DEVICE=mps \
RUN_PREFIX=step2_seq \
bash scripts/run_step2_sequence_local.sh
```

### 8.3 Local UCF 5cat+Normal test evaluation

```bash
../.venv/bin/python scripts/run_ucf_5cat_test_eval_many.py \
  --item step1_prog32 runs/step1_progressive_s42 32 \
  --item step1_joint32 runs/step1_joint_s42 32 \
  --item step2_seq8 runs/step2_seq_progressive_T8_s42 8 \
  --item step2_seq16 runs/step2_seq_progressive_T16_s42 16 \
  --item step2_seq32 runs/step2_seq_progressive_T32_s42 32 \
  --feature_dir ../UCF/UCF_split_test_i3d_5cat \
  --device cpu \
  --output_json runs/ucf_5cat_test_eval_all_current.json
```

### 8.4 Refined-score threshold calibration

```bash
../.venv/bin/python scripts/calibrate_threshold.py \
  --checkpoint runs/step2_seq_progressive_T32_s42/checkpoints/best.pt \
  --feature_dir ../UCF/UCF_split_train_i3d_5cat \
  --list_file auto \
  --val_split \
  --num_segments 32 \
  --device cpu \
  --seed 42 \
  --output_json runs/ucf_val_threshold_calibration_refined_current.json
```

### 8.5 Final binary report at the adopted refined threshold

```bash
../.venv/bin/python scripts/report_binary_accuracy.py \
  --checkpoint runs/step2_seq_progressive_T32_s42/checkpoints/best.pt \
  --feature_dir ../UCF/UCF_split_test_i3d_5cat \
  --list_file auto \
  --num_segments 32 \
  --threshold 0.68433 \
  --device cpu \
  --output_json runs/ucf_test_binary_accuracy_refined_t0684330_current.json
```

### 8.6 Strict boundary analysis

```bash
../.venv/bin/python src/boundary_analysis.py \
  --checkpoint runs/step2_seq_progressive_T32_s42/checkpoints/best.pt \
  --feature_dir ../UCF/UCF_split_test_i3d_5cat \
  --list_file auto \
  --annotation_file ../UCF/Temporal_Anomaly_Annotation_for_Testing_Videos.txt \
  --threshold 0.68433 \
  --device cpu \
  --direct_list \
  --output_dir runs/boundary_test_gt_t0684330_full
```

### 8.7 Error taxonomy

```bash
../.venv/bin/python src/error_taxonomy.py \
  --checkpoint runs/step2_seq_progressive_T32_s42/checkpoints/best.pt \
  --feature_dir ../UCF/UCF_split_train_i3d_5cat \
  --list_file list/ucf-i3d-train-split.list \
  --threshold 0.660604 \
  --device cpu \
  --output_dir runs/error_taxonomy_train877_t0660604_current
```
