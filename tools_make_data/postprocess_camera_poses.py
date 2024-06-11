#!/usr/bin/env python3
#
import os
import sys
import json
import argparse
import numpy as np
import open3d as o3d
from pathlib import Path

sys.path.append("./") # remove when project is compiled
from nerf_slam.config import get_cfg
from nerf_slam.utils import default_argument_parser

USE_FINAL_NGP_POSES=True
USE_FINAL_NGP_AFTER_BA=True
USE_REC=True
USE_MOTION_ONLY_BA=True

# -- some tool functions
# -- taken from NGP
def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2,1e-2,3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]],[v[2], 0, -v[0]],[-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))



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
    output_dir = os.path.join(
        root_output_dir,
        name_experiment,
        str(run_id)
    )
    poses_dirname = cfg.RUNNER.POSES_DIRNAME
    output_dir_poses = os.path.join(
        output_dir,
        poses_dirname,
    )

    output_filename = os.path.join(
        output_dir,
        'final_poses_postprocessed.json'
    )

    # -- save normalization info into configs
    up = np.array(cfg.DATASET.POSES_UP_VECTOR)
    totp = np.array(cfg.DATASET.POSES_CENTER_POINT)
    avglen = cfg.DATASET.POSES_AVGLEN
    poses_scale = cfg.DATASET.POSES_SCALE

    ngp_offset = np.array(cfg.RENDERER.OFFSET)
    ngp_scale = cfg.RENDERER.SCALE

    # -- load dataset
    filename = os.path.join(
        output_dir_poses,
        'final_poses_ngp.json'
    )
    if (USE_FINAL_NGP_AFTER_BA and (cfg.RUNNER.ADD_FINAL_GIANT_BA)):
        filename = os.path.join(
            output_dir_poses,
            'final_poses_ngp_after_final_giant_BA.json'
        )
    if USE_REC:
        filename = filename[:-5]+"_rec.json"

    if (USE_MOTION_ONLY_BA and (cfg.RUNNER.ADD_MOTION_ONLY_BA_TRAJECTORY_FILLER)):
        assert USE_REC
        filename = filename[:-5]+"_motionBA.json"

    if (not os.path.isfile(filename)) or (not USE_FINAL_NGP_POSES):
        poses_files = os.listdir(output_dir_poses)
        poses_files = [x for x in poses_files if x[:5]=='poses']
        poses_files_num = [int(x.split('_')[1].split('.')[0]) for x in poses_files]
        poses_max_files_num = np.max(poses_files_num)
        filename = os.path.join(
            output_dir_poses,
            f'poses_{poses_max_files_num}.json'
        )


    print(" Final poses filename: ", filename)

    all_poses = json.load(open(filename, 'r'))

    if (not os.path.isfile(cfg.DATASET.POSES_FILENAME)) or\
       ("preloaded" in cfg.DATASET.POSES_FILENAME):
        print(" /!\ No poses adjustment BC no preprocessing was done or found")
        print(" output filename: ", output_filename)
        json.dump(all_poses, open(output_filename, "w"))
        return

    all_poses = [np.asarray(x) for x in all_poses]


    # -- # -- UPDATE POSE
    # -- if ngp_scale != 0.33:
    # --     for i in range(len(all_poses)):
    # --         c2w = all_poses[i]
    # --         c2w[0:3,3] *= (0.33 / ngp_scale)
    # --         all_poses[i] = c2w


    # -- UPDATE POSE
    for i in range(len(all_poses)):
        c2w = all_poses[i]
        c2w[0:3,3] /= poses_scale
        all_poses[i] = c2w


    # -- UPDATE POSE
    for i in range(len(all_poses)):
        c2w = all_poses[i]
        c2w[0:3,3] += totp
        all_poses[i] = c2w


    up = up / np.linalg.norm(up)
    R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
    R = np.pad(R,[0,1])
    R[-1, -1] = 1

    R_inv = np.linalg.inv(R)

    # -- UPDATE POSE
    for i in range(len(all_poses)):
        c2w = all_poses[i]
        if c2w.shape[0] < 4:
            c2w = np.pad(c2w,((0,1), (0,0)))
            c2w[-1,-1] = 1
        c2w =  np.matmul(R_inv, c2w)
        all_poses[i] = c2w


    # -- UPDATE POSE
    for i in range(len(all_poses)):
        c2w = all_poses[i]

        c2w[2,:] *= -1 # flip whole world upside down
        c2w = c2w[[1,0,2,3],:] #swap y and z

        all_poses[i] = c2w


    all_poses = [x.tolist() for x in all_poses]
    json.dump(all_poses, open(output_filename,'w'))


if __name__=="__main__":
    args = default_argument_parser().parse_args()
    main(args)
