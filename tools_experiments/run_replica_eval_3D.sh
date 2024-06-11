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
echo "USE VIRTUAL CAMERAS: $USE_VIRT_CAMS_IN_EVAL"
echo "    --  $RUN_ID"

config_file_dst="$OUTPUT_DIR/$SCENE_NAME/$RUN_ID/configs.yml"

conda activate slaim

python tools_make_data/save_3d_mesh.py --config $config_file_dst
python tools_make_data/rotate_final_mesh.py --config $config_file_dst

conda activate slaim_3d_eval

cd dependencies/neural_slam_eval/
input_mesh="../../$OUTPUT_DIR/$SCENE_NAME/$RUN_ID/mesh_final.ply"
output_mesh="../../$OUTPUT_DIR/$SCENE_NAME/$RUN_ID/mesh_final_coslam_culling.ply"
virt_cam_path="data/CoSLAM_data/Replica/$SCENE_NAME_SHORT/virtual_cameras"
eval_config_file="configs/Replica/$SCENE_NAME_SHORT.yaml"
if [[ "$USE_VIRT_CAMS_IN_EVAL" == "True" ]]; then
    python cull_mesh.py --config "$eval_config_file" --input_mesh $input_mesh --remove_occlusion --virtual_cameras --virt_cam_path $virt_cam_path --gt_pose --output_mesh $output_mesh
else
    python cull_mesh.py --config "$eval_config_file" --input_mesh $input_mesh --remove_occlusion --gt_pose --output_mesh $output_mesh
fi

cd ../../
python tools_make_data/clean_3d_mesh_connected_components.py --config $config_file_dst
python tools_make_data/check_reconstruction.py --config $config_file_dst --eval_rec_mode "3d"
python tools_make_data/check_reconstruction.py --config $config_file_dst --eval_rec_mode "2d" --recompute_2d_depths

