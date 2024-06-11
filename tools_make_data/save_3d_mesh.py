import os
import sys
sys.path.append("./") # remove when project is compiled
import json
import torch
import numpy as np
from tqdm import tqdm
from imageio import imwrite
from pathlib import Path

from nerf_slam.config import get_cfg
from nerf_slam.models import build_model
from nerf_slam.mappers import build_mapper
from nerf_slam.trackers import build_tracker
from nerf_slam.renderers import build_renderer
from nerf_slam.datasets import build_dataset
from nerf_slam.utils import default_argument_parser
from nerf_slam.utils import build_experiment_directory
from nerf_slam.runners import build_runner
from nerf_slam.evaluators import build_evaluator

# -- linking Instant-NGP here
# /!\ should link directly in this repo as a submodule
sys.path.append("/srv/essa-lab/flash3/vcartillier3/instant-ngp/build/")
import pyngp as ngp # noqa

sys.path.append("/srv/essa-lab/flash3/vcartillier3/nerf-slam/nerf_slam/utils/")
from ngp_conversion_utils import convert_marching_cubes_bound_to_NGP
from get_convex_hull import get_convex_hull_mask

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

    model_dir = os.path.join(
        experiment_dir,
        'models'
    )

    if args.output_filename:
        output_filename=args.output_filename
    else:
        output_filename = os.path.join(
            experiment_dir,
            'mesh_raw.obj'
        )

    # -- find last saved model
    model_filename = os.path.join(
        model_dir,
        f'model_final.msgpack'
    )
    if cfg.RUNNER.ADD_FINAL_GIANT_BA:
        model_filename = os.path.join(
            model_dir,
            f'model_final_after_final_giant_BA.msgpack'
        )

    if not os.path.isfile(model_filename):
        m = os.listdir(model_dir)
        m = [x for x in m if x.endswith("msgpack")]
        m = [x.split(".")[0] for x in m]
        m = [x.split("_")[1] for x in m]
        m = [int(x) for x in m]
        last_model_num = np.max(m)

        model_filename = os.path.join(
            model_dir,
            f"model_{last_model_num}.msgpack"
        )

    # -- create model
    mode = ngp.TestbedMode.Nerf
    instant_ngp = ngp.Testbed(mode)
    instant_ngp.do_multi_pos_encoding=cfg.MODEL.DO_MULTI_POS_ENCODING

    preprocessed_data_filename = cfg.DATASET.NGP_PREPROCESSED_DATA_FILENAME
    instant_ngp.keep_data_on_cpu=cfg.DATASET.KEEP_DATA_ON_CPU
    if not "_with_poses" in preprocessed_data_filename:
        preprocessed_data_filename = preprocessed_data_filename[:-5]+"_with_poses.json"

    if os.path.isfile(preprocessed_data_filename):
        print(" LOADING DATA from preprocessed files")
        instant_ngp.load_training_data(preprocessed_data_filename)
        if cfg.DATASET.KEEP_DATA_ON_CPU:
            # -- load every other frame to GPU
            print("sending images to GPU")
            print(instant_ngp.nerf.training.n_images_for_training,instant_ngp.nerf.training.n_images_for_training//10)
            mapping_ids = []
            for i in tqdm(range(0,instant_ngp.nerf.training.n_images_for_training,10)):
                instant_ngp.send_image_to_gpu(i)
                mapping_ids.append(i)
            instant_ngp.nerf.training.idx_images_for_mapping=mapping_ids
        else:
            instant_ngp.nerf.training.idx_images_for_mapping=list(range(instant_ngp.nerf.training.n_images_for_training))
    else:
        print("coudn't find dataset file: ", preprocessed_data_filename)
        raise ValueError

    instant_ngp.nerf.training.dataset.aabb_scale = cfg.RENDERER.AABB_SCALE # need that to compute the growth factor resolution in grid NGP (b)
    instant_ngp.nerf.training.dataset.desired_resolution=cfg.DATASET.DESIRED_RESOLUTION

    instant_ngp.load_snapshot(model_filename)

    instant_ngp.density_grid_culling_using_keyframes()

    render_aabb = instant_ngp.render_aabb
    print("m_render_aabb.min: ", render_aabb.min)
    print("m_render_aabb.max: ", render_aabb.max)

    if args.mesh_thresh:
        meshing_thresh = float(args.mesh_thresh)
    else:
        meshing_thresh = cfg.MODEL.MESH_THRESH

    if args.mesh_res:
        res = int(args.mesh_res)
        res = [res,res,res]
    elif cfg.MODEL.MESH_RESOLUTION_METERS > 0.:
        if (cfg.DATASET.POSES_SCALE<=0):
            poses_scale = 1.0
        else:
            poses_scale = cfg.DATASET.POSES_SCALE
        res = np.round(1 / (cfg.MODEL.MESH_RESOLUTION_METERS*cfg.RENDERER.SCALE*poses_scale))
        res = int(res)
        res = [res,res,res]
    else:
        res =  cfg.MODEL.MESH_RESOLUTION
        res = [res,res,res]

    print(" Meshing at resolution: ", res, type(res[0]))
    print(" Meshing threshold: ", meshing_thresh, type(meshing_thresh))

    #Try using the marching cubes bb to crop the meshing area.
    # - not very effective as the bound is axis-aligned
    # - the MCB is aabb and the bound in NGP is also aabb ..
    if cfg.MODEL.USE_MCB_IN_MESHING:
        bound = convert_marching_cubes_bound_to_NGP(cfg)
    else:
        bound = ngp.BoundingBox()

    #TODO: implement a mesh bound (asin ESLAM/CoSLAM) to filter out points
    # outside of the convex hull
    #/!\ needs debugging
    if cfg.MODEL.USE_CONVEX_HULL_FILTER_IN_MARCHING_CUBES:
        use_convex_hull_mask=True
        renderer = build_renderer(cfg)
        convex_hull_mask = get_convex_hull_mask(instant_ngp, renderer, cfg, res)
        instant_ngp.convex_hull_mask=convex_hull_mask.astype(np.uint8)
    else:
        use_convex_hull_mask=False

    # Free memory for meshing
    for i in range(instant_ngp.nerf.training.n_images_for_training):
        instant_ngp.remove_image_from_gpu(i)

    instant_ngp.use_density_grid_in_meshing=False
    ##DEBUG
    ##DEBUG
    instant_ngp.use_anti_aliasing_in_meshing=False
    instant_ngp.n_elements_per_vertex_during_meshing_with_anti_aliasing=16
    ##DEBUG
    ##DEBUG

    #print("bound (aabb): ", bound.min, bound.max)
    #print("bound is empty?: ", bound.is_empty())
    #print(" (m_aabb): ", instant_ngp.aabb.min, instant_ngp.aabb.max)
    #print("render_aabb (m_render_aabb): ", instant_ngp.render_aabb.min, instant_ngp.render_aabb.max)
    #print("render_aabb_to_local (m_render_aabb_to_local): ", instant_ngp.get_render_aabb_to_local())
    #stop
    
    instant_ngp.compute_and_save_marching_cubes_mesh(
        output_filename,
        res,
        aabb=bound,
        thresh=meshing_thresh,
        use_convex_hull_mask=use_convex_hull_mask,
    )
    print(" Saving Mesh here: ", output_filename)


if __name__=="__main__":
    args = default_argument_parser().parse_args()
    main(args)
