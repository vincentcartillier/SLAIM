import os
import sys
import json
from pathlib import Path

sys.path.append("./")
from nerf_slam.config import get_cfg
from nerf_slam.utils import default_argument_parser
from nerf_slam.evaluators import build_evaluator


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.EVALUATOR.NAME="Evaluator3DReconstruction"
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

    # -- setup evaluator
    evaluator = build_evaluator(cfg)

    # -- define output dir and filename
    eval_dir = 'eval'

    output_dir = os.path.join(
        experiment_dir,
        eval_dir
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    use_virt_cams_in_eval=cfg.EVALUATOR.USE_VIRT_CAMS

    # -- load pred and GT meshes
    if os.path.isfile(args.input_filename):
        pred=args.input_filename
    else:
        pred = os.path.join(
            experiment_dir,
            'mesh_final_coslam_culling_connected_comp.ply'
        )

    if not os.path.isfile(pred):
        raise ValueError()

    print("## Using the following pred mesh: ", pred)

    input_folder = cfg.DATASET.INPUT_FOLDER
    scene_name = input_folder.rstrip('/').split('/')[-1]
    if 'Replica' in input_folder:
        if use_virt_cams_in_eval:
            gt = os.path.join(f'Datasets/CoSLAM_data/Replica/{scene_name}/gt_mesh_cull_virt_cams.ply')
        else:
            gt = os.path.join(f'Datasets/Replica/{scene_name}/{scene_name}_mesh_culled_frust_occ.ply')
    elif 'neural_rgbd_data' in input_folder:
        if use_virt_cams_in_eval:
            gt = os.path.join(f'Datasets/CoSLAM_data/neural_rgbd_data/{scene_name}/gt_mesh_cull_virt_cams.ply')
        else:
            gt = os.path.join(f'Datasets/neural_rgbd_data_meshes/{scene_name}/gt_mesh_culled.ply')
    else:
        raise ValueError()

    print("## Using the following GT mesh: ", gt)

    mode = args.eval_rec_mode

    print(f"## Running evaluation in {mode} mode!")
    print("   ---> Tip: use '--eval_rec_mode 2d'  to eval in 2D mode.")

    if args.debug:
        debug_dir=experiment_dir
    else:
        debug_dir=None

    results = evaluator.eval(
        pred,
        gt,
        mode=mode,
        use_virt_cams=use_virt_cams_in_eval,
        debug_dir=debug_dir,
        recompute_2d_depths=args.recompute_2d_depths
    )

    if args.output_filename:
        output_filename=args.output_filename
    else:
        output_filename = os.path.join(
            output_dir,
            f'results_rec_{mode}.json'
        )
    json.dump(results, open(output_filename,"w"))

    print("Saving results here: ", output_filename)



if __name__=="__main__":
    args = default_argument_parser().parse_args()
    main(args)
