#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
MODE="${MODE:-progressive}"
TRAIN_DIR="${TRAIN_DIR:-$PROJECT_ROOT/UCF/UCF_split_train_i3d_5cat}"
TEST_DIR="${TEST_DIR:-$PROJECT_ROOT/UCF/UCF_split_test_i3d_5cat}"
SEGMENTS_LIST="${SEGMENTS_LIST:-8 16 32}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-auto}"
RUN_PREFIX="${RUN_PREFIX:-step2_seq}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/runs}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "Mode:         $MODE"
echo "Train dir:    $TRAIN_DIR"
echo "Test dir:     $TEST_DIR"
echo "Segments:     $SEGMENTS_LIST"
echo "Seed:         $SEED"
echo "Epochs:       $EPOCHS"
echo "Batch size:   $BATCH_SIZE"
echo "Num workers:  $NUM_WORKERS"
echo "Device:       $DEVICE"
echo "Run root:     $RUN_ROOT"

for NUM_SEGMENTS in $SEGMENTS_LIST; do
  RUN_NAME="${RUN_PREFIX}_${MODE}_T${NUM_SEGMENTS}_s${SEED}"
  OUT_DIR="$RUN_ROOT/$RUN_NAME"

  echo
  echo "=== Launching $RUN_NAME ==="

  cmd=(
    env
    "PYTHON_BIN=$PYTHON_BIN"
    "MODE=$MODE"
    "TRAIN_DIR=$TRAIN_DIR"
    "TEST_DIR=$TEST_DIR"
    "NUM_SEGMENTS=$NUM_SEGMENTS"
    "SEED=$SEED"
    "EPOCHS=$EPOCHS"
    "BATCH_SIZE=$BATCH_SIZE"
    "NUM_WORKERS=$NUM_WORKERS"
    "DEVICE=$DEVICE"
    "RUN_NAME=$RUN_NAME"
    "OUT_DIR=$OUT_DIR"
    bash "$SCRIPT_DIR/run_step1_local.sh"
  )

  if [[ -n "$EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    extra_parts=($EXTRA_ARGS)
    cmd+=("${extra_parts[@]}")
  fi

  "${cmd[@]}"
done
