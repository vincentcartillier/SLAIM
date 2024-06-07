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
from imageio import imwrite
from multiprocessing import Pool

sys.path.append("./") # remove when project is compiled
from nerf_slam.config import get_cfg
from nerf_slam.utils import default_argument_parser
from nerf_slam.datasets import build_dataset
from nerf_slam.renderers import build_renderer

use_multiprocessing=True

def frame_extractor(inputs):
    from imageio import imwrite

    i = inputs["input"]
    dataset = inputs["dataset"]
    depth_scaler = inputs["depth_scaler"]
    output_dir_depth = inputs["output_dir_depth"]
    output_dir_rgb = inputs["output_dir_rgb"]
    c2w = inputs["c2w"]

    sample = dataset[i]
    rgb = np.copy(sample['rgb'])
    depth = np.copy(sample['depth'])

    file_path = os.path.join(
        output_dir_rgb,
        f"rgb_{i}.png"
    )
    imwrite(file_path, rgb)

    depth *= depth_scaler

    if depth.max() >= 2**16:
        print("/!\ depth encoding in uint16 overflow")

    depth = depth.astype(np.uint16)
    depth_path = os.path.join(
        output_dir_depth,
        f"depth_{i}.png"
    )
    imwrite(depth_path, depth)

    return  {
            "transform_matrix":c2w.tolist(),
            "file_path":f"rgb/rgb_{i}.png",
            "depth_path":f"depth/depth_{i}.png",
            "index": i
            }




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

    # -- output directories
    output_dir_rgb = os.path.join(
        output_dir,
        "preprocessed_dataset",
        "rgb"
    )
    Path(output_dir_rgb).mkdir(parents=True, exist_ok=True)
    output_dir_depth = os.path.join(
        output_dir,
        "preprocessed_dataset",
        "depth"
    )
    Path(output_dir_depth).mkdir(parents=True, exist_ok=True)
    output_filename = os.path.join(
        output_dir,
        "preprocessed_dataset",
        "transforms.json"
    )

    # -- build dataset
    dataset = build_dataset(cfg)

    # -- build renderer
    renderer = build_renderer(cfg)

    scale = renderer.scale
    offset = renderer.offset
    depth_scale = cfg.DATASET.PNG_DEPTH_SCALE
    if os.path.isfile(cfg.DATASET.POSES_FILENAME):
        poses_scale = cfg.DATASET.POSES_SCALE
    else:
        poses_scale = 1.0
    
    print("Poses scale: ", poses_scale)

    depth_scaler = depth_scale / poses_scale

    fx=renderer.fx
    fy=renderer.fy
    cx=renderer.cx
    cy=renderer.cy
    H=renderer.H
    W=renderer.W
    k1=0.0
    k2=0.0
    p1=0.0
    p2=0.0

    data = {
        "enable_depth_loading": True,
        "scale":scale,
        "offset":offset.tolist(),
        "integer_depth_scale":poses_scale/depth_scale,
        "aabb_scale":renderer.aabb_scale,
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "h": H,
        "w": W,
        "k1": k1,
        "k2": k2,
        "p1": p1,
        "p2": p2,
        "importance_sampling": True,
        "frames":[],
    }

    sample_0 = dataset[0]
    c2w = np.copy(sample_0['c2w'])

    if not use_multiprocessing:
        for i in tqdm(range(len(dataset))):
            sample = dataset[i]
            rgb = np.copy(sample['rgb'])
            depth = np.copy(sample['depth'])

            file_path = os.path.join(
                output_dir_rgb,
                f"rgb_{i}.png"
            )
            imwrite(file_path, rgb)

            depth *= depth_scaler

            if depth.max() >= 2**16:
                print("/!\ depth encoding in uint16 overflow")

            depth = depth.astype(np.uint16)
            depth_path = os.path.join(
                output_dir_depth,
                f"depth_{i}.png"
            )
            imwrite(depth_path, depth)

            data["frames"].append(
                {
                    "transform_matrix":c2w.tolist(),
                    "file_path":f"rgb/rgb_{i}.png",
                    "depth_path":f"depth/depth_{i}.png"
                }
            )



    else:
        inputs = [{'input':i,
                   'dataset': dataset,
                   'depth_scaler': depth_scaler,
                   'output_dir_depth': output_dir_depth,
                   'output_dir_rgb': output_dir_rgb,
                   "c2w": c2w,
                  } for i in range(len(dataset))]

        pool = Pool(16)

        results = list(
            tqdm(
                pool.imap_unordered(
                    frame_extractor, inputs),
                    total=len(inputs)
            )
        )

        results = sorted(results, key=lambda d: d["index"])

        for r in results:
            data["frames"].append(r)


    json.dump(data, open(output_filename, "w"))
    print("Saving file in: ", output_filename)
    
    cfg.defrost()
    cfg.DATASET.NGP_PREPROCESSED_DATA_FILENAME = output_filename 
    cfg.freeze()
    with open(os.path.join(output_dir, "configs.yml"), "w") as f:
          f.write(cfg.dump())


if __name__=="__main__":
    args = default_argument_parser().parse_args()
    main(args)
