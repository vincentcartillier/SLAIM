import os
import sys
import json
import numpy as np
from pathlib import Path

sys.path.append("./")
from nerf_slam.config import get_cfg
from nerf_slam.datasets import build_dataset
from nerf_slam.utils import default_argument_parser
from nerf_slam.evaluators import build_evaluator


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    #NOTE: do not load NGP format poses
    cfg.DATASET.POSES_FILENAME=""
    cfg.freeze()
    return cfg


def main(args):
    cfg = setup(args)

    root = cfg.META.OUTPUT_DIR
    experiment_name = cfg.META.NAME_EXPERIMENT
    run_id = cfg.META.RUN_ID
    run_id = str(run_id)

    experiment_dir = os.path.join(
        root,
        experiment_name,
        run_id,
    )

    # -- build dataset
    dataset = build_dataset(cfg)

    # -- setup model
    evaluator = build_evaluator(cfg)

    # -- load GT
    gt = dataset.get_all_poses()

    # -- filter out frames with invalid poses (inf or NaN)
    mask_valid = []
    for x in gt:
        if np.any(np.isinf(x)) or np.any(np.isnan(x)):
            mask_valid.append(0)
        else:
            mask_valid.append(1)

    gt = [gt[i] for i in range(len(gt)) if mask_valid[i]==1]

    # -- load predictions
    pred_filename = os.path.join(
        experiment_dir,
        'final_poses_postprocessed.json'
    )
    pred = json.load(open(pred_filename, 'r'))
    pred = [pred[i] for i in range(len(pred)) if mask_valid[i]==1]

    # -- define output dir and filename
    eval_dir = 'eval'

    output_dir = os.path.join(
        experiment_dir,
        eval_dir
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    viz_filename = os.path.join(
        output_dir,
        'ate_viz.png'
    )

    if args.max_frame_id:
        m = int(args.max_frame_id)
        pred = pred[:m]
        gt = gt[:m]

    print("# frames (pred, GT): ", len(pred), len(gt))

    results = evaluator.eval(pred,
                             gt,
                             viz_filename=viz_filename)

    for k,r in results.items():
        print(k, r)

    rmse = results['absolute_translational_error.rmse']
    mean = results['absolute_translational_error.mean']
    median = results['absolute_translational_error.median']
    std = results['absolute_translational_error.std']
    min_val = results['absolute_translational_error.min']
    max_val = results['absolute_translational_error.max']

    rmse = round(rmse * 100.0, 2)
    mean = round(mean * 100.0, 2)
    median = round(median * 100.0, 2)
    std = round(std * 100.0, 2)
    min_val = round(min_val * 100.0, 2)
    max_val = round(max_val * 100.0, 2)

    print(f"{rmse}, {mean}, {median}, {std}, {min_val}, {max_val}")

    output_filename = os.path.join(
        output_dir,
        'results.json'
    )
    json.dump(results, open(output_filename,"w"))


if __name__=="__main__":
    args = default_argument_parser().parse_args()
    main(args)
