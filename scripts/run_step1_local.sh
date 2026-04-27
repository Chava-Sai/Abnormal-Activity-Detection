#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

DEFAULT_TRAIN_DIR="$PROJECT_ROOT/UCF/UCF_split_train_i3d_5cat"
DEFAULT_TEST_DIR="$PROJECT_ROOT/UCF/UCF_split_test_i3d_5cat"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
MODE="${MODE:-progressive}"
TRAIN_DIR="${TRAIN_DIR:-$DEFAULT_TRAIN_DIR}"
TEST_DIR="${TEST_DIR:-$DEFAULT_TEST_DIR}"
NUM_SEGMENTS="${NUM_SEGMENTS:-32}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-auto}"
RUN_NAME="${RUN_NAME:-${MODE}_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/runs/$RUN_NAME}"

JOINT_FLAG=()
if [[ "$MODE" == "joint" ]]; then
  JOINT_FLAG+=(--joint)
elif [[ "$MODE" != "progressive" ]]; then
  echo "Unsupported MODE=$MODE (use progressive or joint)" >&2
  exit 1
fi

mkdir -p "$OUT_DIR/checkpoints" "$OUT_DIR/logs/tensorboard"

echo "Mode:        $MODE"
echo "Train dir:   $TRAIN_DIR"
echo "Test dir:    $TEST_DIR"
echo "Output dir:  $OUT_DIR"
echo "Segments:    $NUM_SEGMENTS"
echo "Seed:        $SEED"
echo "Device:      $DEVICE"

cmd=(
  "$PYTHON_BIN" "$REPO_ROOT/src/train.py"
  --train_dir "$TRAIN_DIR"
  --test_dir "$TEST_DIR"
  --train_list auto
  --test_list auto
  --val_ratio 0.2
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --lr 1e-4
  --weight_decay 1e-4
  --device "$DEVICE"
  --num_segments "$NUM_SEGMENTS"
  --frames_per_segment 16
  --d_model 512
  --nhead 8
  --trn_layers 2
  --lambda_cls 0.5
  --lambda_bnd 0.3
  --topk_ratio 0.1
  --topk_cls 5
  --eval_freq 5
  --save_freq 10
  --seed "$SEED"
  --checkpoint_dir "$OUT_DIR/checkpoints"
  --log_dir "$OUT_DIR/logs/tensorboard"
)

if [[ "$MODE" == "joint" ]]; then
  cmd+=(--joint)
fi
cmd+=("$@")

"${cmd[@]}"
