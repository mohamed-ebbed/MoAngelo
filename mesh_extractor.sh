#!/bin/bash

BASE_FOLDER="$1"

if [ -z "$BASE_FOLDER" ]; then
    echo "❌ BASE_FOLDER argument is missing!"
    exit 1
fi

MESH_OUTPUT_DIR="${BASE_FOLDER}/meshes"
RESOLUTION=2048
BLOCK_RES=128
EXTRACT_SCRIPT="extract_mesh.py"

mkdir -p "$MESH_OUTPUT_DIR"

for EXP_FOLDER in "$BASE_FOLDER"/*; do
    if [ -d "$EXP_FOLDER" ]; then
        CONFIG="${EXP_FOLDER}/config.yaml"
        EXP_NAME=$(basename "$EXP_FOLDER")
        TIME_STEP=$(echo "$EXP_NAME" | grep -oP '_\K[0-9]+$')

        if [ -z "$TIME_STEP" ]; then
            echo "❌ Could not extract time step from folder name: $EXP_FOLDER"
            continue
        fi

        CHECKPOINT=$(find "$EXP_FOLDER" -type f \( -name '*_checkpoint.pt' -o -name '*.pth' \) | sort | tail -n 1)

        if [ -z "$CHECKPOINT" ]; then
            echo "⚠️  No checkpoint (.pt or .pth) found in $EXP_FOLDER. Skipping."
            continue
        fi

        OUTPUT_FILE="${MESH_OUTPUT_DIR}/${TIME_STEP}.ply"

        echo "✅ Extracting mesh for time step $TIME_STEP from $CHECKPOINT"
        
        python "$EXTRACT_SCRIPT" \
            --config="$CONFIG" \
            --checkpoint="$CHECKPOINT" \
            --output_file="$OUTPUT_FILE" \
            --resolution="$RESOLUTION" \
            --block_res="$BLOCK_RES" \
            --local_rank=0
    fi
done

ZIP_PATH="${BASE_FOLDER}/meshes.zip"
echo "📦 Zipping all meshes into $ZIP_PATH"
zip -j "$ZIP_PATH" "${MESH_OUTPUT_DIR}"/*.ply

echo "🎉 Done processing: $BASE_FOLDER"
