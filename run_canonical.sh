#!/bin/bash
# Runs canonical (first-frame) NeuralAngelo optimization for all benchmark scenes.
# Usage: bash run_canonical.sh
#
# Outputs are saved to logs/canonical/<scene>_canonical/
# W&B logs are reported to project "moangelo".

CONFIG=projects/neuralangelo/configs/custom/canonical_experiments.yaml
GROUP=canonical

SCENES=(
    "actor1_seq1_140"
    "actor5_seq2_176"
    "actor6_seq2_62"
    "actor6_seq2_180"
    "actor7_seq1_456"
    "actor8_seq2_535"
)


for SCENE in "${SCENES[@]}"; do
    NAME="${SCENE}_canonical"
    DATASET="datasets/${SCENE}/"

    if [ ! -d "$DATASET" ]; then
        echo "Dataset not found: $DATASET — skipping $SCENE"
        continue
    fi

    echo "=========================================="
    echo " Canonical optimization: $SCENE"
    echo " Config:   $CONFIG"
    echo " Dataset:  $DATASET"
    echo " Logdir:   logs/${GROUP}/${NAME}"
    echo "=========================================="

    python train.py \
        --single_gpu \
        --expname="${NAME}" \
        --logdir="logs/${GROUP}/${NAME}" \
        --config="${CONFIG}" \
        --show_pbar \
        --wandb \
        --wandb_name moangelo \
        --data.root="${DATASET}"

    if [ $? -ne 0 ]; then
        echo "Training failed for $SCENE."
    fi
done
