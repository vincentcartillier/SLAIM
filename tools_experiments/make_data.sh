#!/bin/bash

source ~/.bashrc

CONFIG_FILE="$1"

SCENE_NAME=$(awk '/NAME_EXPERIMENT:/{print $2}' "$CONFIG_FILE")
SCENE_NAME="${SCENE_NAME:1:-1}"
DATASET=$(echo "$SCENE_NAME" | cut -d'_' -f1)
SCENE_NAME_SHORT=$(echo "$SCENE_NAME" | cut -d'_' -f2)

NEW_CONF="data/${SCENE_NAME}/0/configs.yml"
echo $NEW_CONF

conda activate slaim

python tools_make_data/preprocess_camera_poses.py --config "$CONFIG_FILE"

python tools_make_data/save_input_pc_for_init_scene_scaling.py --config "$NEW_CONF"

python tools_make_data/estimate_scale_and_shift_using_GT_pc.py --config "$NEW_CONF"

python tools_make_data/prepare_ngp_format_dataset.py --config "$NEW_CONF"

python tools_make_data/add_poses_to_transforms_json.py --config "$NEW_CONF"
