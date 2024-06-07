#!/usr/bin/env python3
#
import os
import sys
import json
import argparse
import numpy as np

sys.path.append("./") # remove when project is compiled
from nerf_slam.config import get_cfg
from nerf_slam.utils import default_argument_parser
from nerf_slam.datasets import build_dataset


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

    # -- build dataset
    dataset = build_dataset(cfg)
    
    transforms_filename = cfg.DATASET.NGP_PREPROCESSED_DATA_FILENAME
    print("Input file: ", transforms_filename)

    assert os.path.isfile(transforms_filename)

    transforms = json.load(open(transforms_filename, "r"))
    
    for i in range(len(dataset)):
        c2w = dataset.poses[i]
        if np.any(np.isinf(c2w)) or np.any(np.isnan(c2w)):
            c2w = prev_c2w.copy()
        transforms["frames"][i]["transform_matrix"] = c2w.tolist()
        prev_c2w = c2w.copy()
    
    output_filename = transforms_filename[:-5]+"_with_poses.json"

    print("Saving file in: ", output_filename)
    json.dump(transforms, open(output_filename, "w"))

    cfg.defrost()
    cfg.DATASET.NGP_PREPROCESSED_DATA_FILENAME = output_filename 
    cfg.freeze()
    with open(os.path.join(output_dir, "configs.yml"), "w") as f:
          f.write(cfg.dump())



if __name__=="__main__":
    args = default_argument_parser().parse_args()
    main(args)
