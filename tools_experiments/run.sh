#!/bin/bash


# Function to display the command help
show_help() {
    echo    "## Usage: $0 [options]"
    echo -e "          $0 <output_dir> <config_file> <parent_configs> \n"
    echo
    echo "Options:"
    echo "  --help     Show this help message and exit"
    echo
}

# Check for the --help flag
if [[ -z "$1" || -z "$2" || -z "$3" || "$1" == "-h" || "$1" == "--help" || "$1" == "--helper" ]]; then
    show_help
    exit 0
fi

source ~/.bashrc

OUTPUT_DIR="$1"
CONFIG_FILE="$2"
PARENT_FILE="$3"

SCENE_NAME=$(awk '/NAME_EXPERIMENT:/{print $2}' "$CONFIG_FILE")
SCENE_NAME="${SCENE_NAME:0}"

RUN_ID="0"

echo "OUTPUT DIR: $OUTPUT_DIR"
echo "CONFIG FILE: $CONFIG_FILE"
echo "PARENT FILE: $PARENT_FILE"
echo "SCENE NAME: $SCENE_NAME"
echo "    --  $RUN_ID"

config_file_dst="$OUTPUT_DIR/$SCENE_NAME/$RUN_ID/configs.yml"

echo $config_file_dst

conda activate slaim
python tools_experiments/run_sequential.py --config "$CONFIG_FILE" --output_dir "$OUTPUT_DIR" --run_id $RUN_ID --parent "$PARENT_FILE"
python tools_make_data/postprocess_camera_poses.py --config $config_file_dst
python tools_make_data/check_evaluator.py --config $config_file_dst

