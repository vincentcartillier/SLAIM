#!/bin/bash

source ~/.bashrc

OUTPUT_DIR="$1"
CONFIG_FILE="$2"
PARENT_FILE="$3"

SCENE_NAME=$(awk '/NAME_EXPERIMENT:/{print $2}' "$CONFIG_FILE")
SCENE_NAME="${SCENE_NAME:0}"
SCENE_NAME_SHORT="${SCENE_NAME:8}"

USE_VIRT_CAMS_IN_EVAL=$(awk '/USE_VIRT_CAMS:/{print $2}' "$CONFIG_FILE")

RUN_ID="0"

echo "OUTPUT DIR: $OUTPUT_DIR"
echo "CONFIG FILE: $CONFIG_FILE"
echo "PARENT FILE: $PARENT_FILE"
echo "SCENE NAME: $SCENE_NAME"
echo "SCENE NAME SHORT: $SCENE_NAME_SHORT"
echo "    --  $RUN_ID"

config_file_dst="$OUTPUT_DIR/$SCENE_NAME/$RUN_ID/configs.yml"

conda activate slaim
python tools_experiments/run_sequential.py --config "$CONFIG_FILE" --output_dir "$OUTPUT_DIR" --run_id $RUN_ID --parent "$PARENT_FILE"
python tools_make_data/postprocess_camera_poses.py --config $config_file_dst
python tools_make_data/check_evaluator.py --config $config_file_dst

