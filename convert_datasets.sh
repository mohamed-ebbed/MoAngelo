#!/bin/bash
# Converts all benchmark ActorsHQ scenes to the transforms.json format.
# Usage: bash convert_datasets.sh --dataroot /path/to/ActorsHQ
#
# Outputs are saved to datasets/<scene_name>/

ACTORSHQ_ROOT=""
CONVERT_SCRIPT=projects/neuralangelo/scripts/convert_actorshq_to_json.py

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataroot) ACTORSHQ_ROOT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$ACTORSHQ_ROOT" ]; then
    echo "Error: --dataroot is required."
    echo "Usage: bash convert_datasets.sh --dataroot /path/to/ActorsHQ"
    exit 1
fi

# scene_name -> "ActorXX/SequenceY frame_start frame_end"
declare -A SCENE_INFO
SCENE_INFO["actor1_seq1_140"]="Actor01/Sequence1 140 164"
SCENE_INFO["actor5_seq2_176"]="Actor05/Sequence2 176 200"
SCENE_INFO["actor6_seq2_62"]="Actor06/Sequence2  62  86"
SCENE_INFO["actor6_seq2_180"]="Actor06/Sequence2 180 204"
SCENE_INFO["actor7_seq1_456"]="Actor07/Sequence1 456 480"
SCENE_INFO["actor8_seq2_535"]="Actor08/Sequence2 535 559"

SCENES=(
    "actor1_seq1_140"
    "actor5_seq2_176"
    "actor6_seq2_62"
    "actor6_seq2_180"
    "actor7_seq1_456"
    "actor8_seq2_535"
)

for SCENE in "${SCENES[@]}"; do
    read -r SEQ START END <<< "${SCENE_INFO[$SCENE]}"

    DATAROOT="${ACTORSHQ_ROOT}/${SEQ}"
    OUTPUT_DIR="datasets/${SCENE}"

    if [ -f "${OUTPUT_DIR}/transforms.json" ]; then
        echo "Already converted: $SCENE — skipping."
        continue
    fi

    echo "Converting $SCENE  (frames $START – $END)..."
    python "${CONVERT_SCRIPT}" \
        --dataroot   "${DATAROOT}" \
        --output_dir "${OUTPUT_DIR}" \
        --frame_start "${START}" \
        --frame_end   "${END}" \
        --aspect_ratio height_larger

    if [ $? -ne 0 ]; then
        echo "Failed: $SCENE"
    else
        echo "Done: $SCENE -> ${OUTPUT_DIR}"
    fi
done
