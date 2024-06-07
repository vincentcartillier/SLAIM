#!/usr/bin/env python3
#
import os
import sys
import json
import argparse
import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm

sys.path.append("./") # remove when project is compiled
from nerf_slam.config import get_cfg
from nerf_slam.utils import default_argument_parser
from nerf_slam.datasets import build_dataset
from nerf_slam.renderers import build_renderer

FIX_SCALE_TO_ONE = False
target_aabb = 1

def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.freeze()
    return cfg

def main(args):
    cfg = setup(args)

    root_output_dir = cfg.META.OUTPUT_DIR
    name_experiment = cfg.META.NAME_EXPERIMENT
    run_id = cfg.META.RUN_ID
    experiment_dir = os.path.join(
        root_output_dir,
        name_experiment,
        str(run_id)
    )

    if os.path.isfile(cfg.DATASET.POSES_FILENAME):
        pc_filename=os.path.join(
            experiment_dir,
            'GT_point_cloud_preprocess_input.ply'
        )
        poses_scale = cfg.DATASET.POSES_SCALE
    else:
        pc_filename=os.path.join(
            experiment_dir,
            'GT_point_cloud_raw_input.ply'
        )
        poses_scale = 1.0

    # check if cleaned version exists
    test_filename = pc_filename[:-4] + "_cleaned.ply"
    if os.path.isfile(test_filename):
        pc_filename=test_filename

    print(" Loading PC: ", pc_filename)

    point_cloud = o3d.io.read_point_cloud(pc_filename)

    points = np.array(point_cloud.points)
    colors = np.array(point_cloud.colors)

    print("Some point clouds stats")
    print("PC # points: ", points.shape)
    print("x-axis: ", points[:,0].min(), points[:,0].max())
    print("y-axis: ", points[:,1].min(), points[:,1].max())
    print("z-axis: ", points[:,2].min(), points[:,2].max())
    print("\n")

    if FIX_SCALE_TO_ONE:
        scale = 1.0
        # find offset
        c = np.mean(points, axis=0)
        offset = np.array([0.5, 0.5, 0.5]) - c

        points = points + offset
        
        #Rotate axis - same as NGP
        points[:,[0,1,2]] = points[:,[1,2,0]]

        # find addbb_scale
        aabb_scale = np.max([points[:,x].max()-points[:,x].min() for x in range(3)])
        aabb_scale = np.ceil(aabb_scale)

        # find desired_rez
        desired_rez = 1. / 0.01

        print("scale: ", scale)
        print("offset: ", offset)
        print("aabb_scale: ", aabb_scale)
        print("desired_rez (1cm): ", desired_rez)

    else:
        scale = np.max([points[:,x].max()-points[:,x].min() for x in range(3)])
        scale += 0.10 * poses_scale
        print("Max distance within box (m): ", scale / poses_scale)

        desired_rez = (scale / poses_scale) / (0.01 * target_aabb)
        #desired_rez = 16*np.round(desired_rez / 16.)
        desired_rez = np.ceil(desired_rez)

        desired_rez_m = target_aabb * (scale / poses_scale) / float(desired_rez)
        
        desired_rez_4cm = desired_rez * desired_rez_m / 0.04
        desired_rez_4cm = 16*np.round(desired_rez_4cm / 16.)
        desired_rez_4cm_m = target_aabb * (scale / poses_scale)/float(desired_rez_4cm)
        
        desired_rez_2cm = desired_rez * desired_rez_m / 0.02
        desired_rez_2cm = 16*np.round(desired_rez_2cm / 16.)
        desired_rez_2cm_m = target_aabb * (scale / poses_scale)/float(desired_rez_2cm)


        points = points / scale * target_aabb

        c = np.min(points, axis=0)
        offset = np.array([0.05, 0.05, 0.05]) * poses_scale / scale * target_aabb - c
        
        #c = np.mean(points, axis=0)
        #offset = np.array([0.5, 0.5, 0.5]) - c

        #offset = [points[:,x].min() for x in range(3)]
        #offset = np.array(offset)
        #offset *= -1.

        points = points + offset

        #Rotate axis - same as NGP
        points[:,[0,1,2]] = points[:,[1,2,0]]

        print("scale: ", 1 / scale * target_aabb)
        print("offset: ", offset)
        print("aabb_scale: ", target_aabb)
        print("desired_rez (1cm): ", desired_rez)
        print("desired_rez (4cm): ", desired_rez_4cm)
        print("actual resolution in m (4cm): ", desired_rez_4cm_m)
        print("desired_rez (2cm): ", desired_rez_2cm)
        print("actual resolution in m (2cm): ", desired_rez_2cm_m)
        
        cfg.defrost()
        cfg.RENDERER.SCALE = float(1 / scale * target_aabb)
        cfg.RENDERER.OFFSET = offset.tolist()
        cfg.RENDERER.AABB_SCALE = int(target_aabb)
        cfg.DATASET.DESIRED_RESOLUTION = int(desired_rez)
        cfg.freeze()
        with open(os.path.join(experiment_dir, "configs.yml"), "w") as f:
              f.write(cfg.dump())

    print("\n")
    print(points[:,0].min(), points[:,0].max())
    print(points[:,1].min(), points[:,1].max())
    print(points[:,2].min(), points[:,2].max())
    
    if args.no_output: return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    output_filename=os.path.join(
        experiment_dir,
        'GT_point_cloud_scaled_and_offset_asin_NGP.ply'
    )
    
    print(" Saving scaled and offset PC: ", output_filename)
    o3d.io.write_point_cloud(
        output_filename,
        pcd
    )










if __name__=="__main__":
    args = default_argument_parser().parse_args()
    main(args)
