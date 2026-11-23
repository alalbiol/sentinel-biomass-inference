#!/usr/bin/env bash
# Train the ten released models one after another with the original settings (training/train.py defaults),
# then score each on the BioMassters test set. Set the four folders below; see README.md, "Reproduce the training".
set -euo pipefail
cd "$(dirname "$0")/.."
TRAIN_FEATURES=${TRAIN_FEATURES:?converted training chips (.safetensors)}
TRAIN_LABELS=${TRAIN_LABELS:?train_agbm folder}
TEST_FEATURES=${TEST_FEATURES:?converted test chips (.safetensors)}
TEST_LABELS=${TEST_LABELS:?test_agbm folder}
OUT=${OUT:-runs}
for backbone in convnextv2 efficientnet efficientnetv2 mobilenetv4 resnetrs101; do
  for mode in early late; do
    model=${backbone}_${mode}
    [ -f "$OUT/$model/$model.safetensors" ] && { echo "skip $model (done)"; continue; }
    python -m training.train --model "$model" --features-dir "$TRAIN_FEATURES" --labels-dir "$TRAIN_LABELS" --output-dir "$OUT"
    python -m training.evaluate --models "$model" --model-dir "$OUT/$model" \
      --features-dir "$TEST_FEATURES" --labels-dir "$TEST_LABELS" --output "$OUT/$model/biomassters_test.json"
  done
done
