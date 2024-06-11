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
from multiprocessing import Pool

sys.path.append("./") # remove when project is compiled
from nerf_slam.config import get_cfg
from nerf_slam.utils import default_argument_parser
from nerf_slam.datasets import build_dataset
from nerf_slam.renderers import build_renderer

def rec_pc_local(inputs):
    i = inputs["input"]
    dataset = inputs["dataset"]
    K = inputs["K"]
    H = inputs["H"]
    W = inputs["W"]

    K_inv = np.linalg.inv(K)

    sample = dataset[i]
    depth = sample['depth']
    rgb = sample["rgb"]
    rgb = rgb[:,:,:3].astype(np.float32) / 256.
    c2w = sample["c2w"]

    if np.any(np.isinf(c2w)):
        return {'index':i, 'pc': [], 'rgb': []}

    ux = np.arange(W)
    uy = np.arange(H)
    u, v = np.meshgrid(ux, uy)
    N = int(H*W)
    pixel_coords = np.asarray(
        [
            u.flatten(),
            v.flatten(),
            np.ones(N)
        ]
    )

    c2w[:3,0] *= 1.0
    c2w[:3,1] *= -1.0
    c2w[:3,2] *= -1.0

    if c2w.shape[0] < 4:
        c2w = np.pad(c2w, ((0,1), (0,0)))
        c2w[-1,-1] = 1.0

    pixel_depth = depth.flatten()
    pixel_rgb = rgb.reshape([N,3])

    mask = pixel_depth > 0.

    cam_coords = K_inv @ pixel_coords * pixel_depth

    cam_coords_homo = np.concatenate(
        [cam_coords, np.ones(N).reshape(1,-1)],
        axis=0
    )

    world_coords = c2w @ cam_coords_homo

    world_coords = world_coords[:3,:]

    world_coords = world_coords.T

    world_coords = world_coords[mask,:]
    pixel_rgb = pixel_rgb[mask,:]

    return {'index':i, 'pc': world_coords, "rgb": pixel_rgb}



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

    # -- build dataset
    dataset = build_dataset(cfg)

    # -- build renderer
    renderer = build_renderer(cfg)

    H = renderer.H
    W = renderer.W
    fx = renderer.fx
    fy = renderer.fy
    cx = renderer.cx
    cy = renderer.cy

    K = np.array(
        [
            [fx, .0, cx],
            [.0, fy, cy],
            [.0, .0, 1.0]
        ]
    )


    if os.path.isfile(cfg.DATASET.POSES_FILENAME):
        output_filename=os.path.join(
            experiment_dir,
            'GT_point_cloud_preprocess_input.ply'
        )
        poses_scale = cfg.DATASET.POSES_SCALE
    else:
        output_filename=os.path.join(
            experiment_dir,
            'GT_point_cloud_raw_input.ply'
        )
        poses_scale = 1.0



    # -- process

    inputs = [{'input':i,
               'dataset': dataset,
               'H': H,
               'W': W,
               'K': K
              #} for i in range(0,len(dataset),1)]
              } for i in range(0,len(dataset),2)] # for debugging NiceSLAM APT

    pool = Pool(16)

    results = list(
        tqdm(
            pool.imap_unordered(
                rec_pc_local, inputs),
                total=len(inputs)
        )
    )

    point_cloud = o3d.geometry.PointCloud()

    #voxel_size = 0.2 * poses_scale
    voxel_size = 0.01 * poses_scale #1cm for smaller scenes

    for i, r in tqdm(enumerate(results)):
        if len(r["pc"])>0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(r["pc"])
            pcd.colors = o3d.utility.Vector3dVector(r["rgb"])
            point_cloud += pcd

        if i%100==0:
            point_cloud=point_cloud.voxel_down_sample(voxel_size=voxel_size)

    print("prev #points: ", point_cloud.points)
    point_cloud=point_cloud.voxel_down_sample(voxel_size=voxel_size)
    print("new #points: ", point_cloud.points)

    o3d.io.write_point_cloud(
        output_filename,
        point_cloud
    )

    print("saved PC at: ", output_filename)







if __name__=="__main__":
    args = default_argument_parser().parse_args()
    main(args)
