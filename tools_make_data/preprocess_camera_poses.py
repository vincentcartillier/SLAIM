#!/usr/bin/env python3
#
import os
import sys
import json
import argparse
import numpy as np
import open3d as o3d
from tqdm import tqdm
from pathlib import Path

sys.path.append("./")
from nerf_slam.config import get_cfg
from nerf_slam.utils import default_argument_parser
from nerf_slam.datasets import build_dataset
from nerf_slam.utils import build_experiment_directory


# -- some tool functions
# -- taken from NGP
def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa+ta*da+ob+tb*db) * 0.5, denom


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
    cfg.DATASET.POSES_FILENAME=""
    cfg.freeze()
    build_experiment_directory(cfg)
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

    # -- load dataset
    dataset = build_dataset(cfg)
    n_images = len(dataset)
    all_poses = dataset.get_all_poses()

    # -- filter out missing poses
    all_valid_poses = [x for x in all_poses if not np.any(np.isinf(x))]
    print("Found ", n_images - len(all_valid_poses), " poses with wrong/no poses")

    # -- format camera poses
    up = np.zeros(3)
    for i in range(n_images):
        c2w = all_poses[i]

        if np.any(np.isinf(c2w)):
            continue

        # -- UPDATE POSE
        c2w = c2w[[1,0,2,3],:] #swap y and z
        c2w[2,:] *= -1 # flip whole world upside down

        all_poses[i] = c2w

        up += c2w[0:3,1]

    up = up / np.linalg.norm(up)
    print("The up vector is: ", up)
    R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
    R = np.pad(R,[0,1])
    R[-1, -1] = 1

    # -- UPDATE POSE
    for i in range(len(all_poses)):
        c2w = all_poses[i]

        if np.any(np.isinf(c2w)):
            continue

        c2w =  np.matmul(R, c2w)
        all_poses[i] = c2w


    # find a central point they are all looking at
    print("computing center of attention...")
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for i in tqdm(range(0,n_images,50)):
        c2w = all_poses[i]

        if np.any(np.isinf(c2w)):
            continue

        mf = c2w[0:3,:]
        for j in range(n_images):
            c2w = all_poses[j]
            mg = c2w[0:3,:]
            p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
            if w > 0.00001:
                totp += p*w
                totw += w
    if totw > 0.0:
    	totp /= totw

    print('The center point of attention is: ', totp) # the cameras are looking at totp

    # -- UPDATE POSE
    for i in range(len(all_poses)):
        c2w = all_poses[i]

        if np.any(np.isinf(c2w)):
            continue

        c2w[0:3,3] -= totp
        all_poses[i] = c2w

    avglen = 0.
    for c2w in all_poses:

        if np.any(np.isinf(c2w)):
            continue

        avglen += np.linalg.norm(c2w[0:3,3])
    avglen /= n_images
    avglen = float(avglen)

    print("The avglen coef is: ", avglen)
    print("The normalization coef is: ", 4.0/avglen)

    # -- UPDATE POSE
    for i in range(len(all_poses)):
        c2w = all_poses[i]

        if np.any(np.isinf(c2w)):
            continue

        c2w[0:3,3] *= 4.0/avglen
        all_poses[i] = c2w


    # -- save poses
    all_poses_to_save = []
    for c2w in all_poses:
        all_poses_to_save.append(c2w.tolist())
    output_poses_filename = os.path.join(output_dir, 'poses_processed_NGP.json')
    json.dump(all_poses_to_save, open(output_poses_filename, 'w'))

    o3d_poses = o3d.geometry.TriangleMesh()
    for c2w in all_poses[::10]:

        if np.any(np.isinf(c2w)):
            continue

        cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
        cam.transform(c2w)
        o3d_poses += cam

    output_poses_o3d_filename=os.path.join(output_dir,'poses_processed_NGP.ply')
    o3d.io.write_triangle_mesh(output_poses_o3d_filename, o3d_poses)

    # -- save normalization info into configs
    cfg.defrost()
    cfg.DATASET.POSES_UP_VECTOR = up.tolist()
    cfg.DATASET.POSES_CENTER_POINT = totp.tolist()
    cfg.DATASET.POSES_AVGLEN = avglen
    cfg.DATASET.POSES_SCALE = 4.0/avglen
    cfg.DATASET.POSES_FILENAME=output_poses_filename
    cfg.freeze()
    with open(os.path.join(output_dir, "configs.yml"), "w") as f:
          f.write(cfg.dump())

if __name__=="__main__":
    args = default_argument_parser().parse_args()
    main(args)
