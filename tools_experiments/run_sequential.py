import os
import sys
sys.path.append("./") # remove when project is compiled
import torch
from tqdm import tqdm
from imageio import imwrite
import numpy as np
import math
import cv2
import json


from nerf_slam.config import get_cfg
from nerf_slam.mappers import build_mapper
from nerf_slam.trackers import build_tracker
from nerf_slam.renderers import build_renderer
from nerf_slam.datasets import build_dataset
from nerf_slam.utils import default_argument_parser
from nerf_slam.utils import build_experiment_directory
from nerf_slam.utils import compute_dist_norm, get_integrals_coefs, get_dist_sum
from nerf_slam.runners import build_runner
from nerf_slam.bundle_adjustments import build_ba

# -- linking Instant-NGP here
# /!\ should link directly in this repo as a submodule
sys.path.append("dependencies/instant-ngp/build/")
import pyngp as ngp # noqa
sys.path.append("dependencies/instant-ngp/scripts/")
from common import *


def build_instant_ngp(cfg):
    mode = ngp.TestbedMode.Nerf
    testbed = ngp.Testbed(mode)
    testbed.nerf.sharpen = float(0)
    testbed.exposure = 0.0
    testbed.nerf.training.dataset.aabb_scale = cfg.RENDERER.AABB_SCALE # need that to compute the growth factor resolution in grid NGP (b)
    testbed.nerf.training.dataset.desired_resolution=cfg.DATASET.DESIRED_RESOLUTION

    # -- create model
    testbed.do_multi_pos_encoding=cfg.MODEL.DO_MULTI_POS_ENCODING
    network = cfg.MODEL.INSTANT_NGP_CONFIG_FILE
    testbed.reload_network_from_file(network)

    # -- settings
    testbed.nerf.render_with_camera_distortion = False
    testbed.nerf.training.depth_supervision_lambda = cfg.MODEL.DEPTH_SUPERVISION_LAMBDA
    testbed.nerf.training.depth_supervision_lambda_tracking=cfg.MODEL.DEPTH_SUPERVISION_LAMBDA_TRACKING
    testbed.nerf.training.rgb_supervision_lambda = cfg.MODEL.RGB_SUPERVISION_LAMBDA
    testbed.nerf.training.rgb_supervision_lambda_tracking=cfg.MODEL.RGB_SUPERVISION_LAMBDA_TRACKING
    testbed.nerf.training.sample_image_proportional_to_error=False
    testbed.nerf.training.n_steps_between_error_map_updates=cfg.MAPPER.NUM_STEPS_BETWEEN_ERROR_MAP_UPDATE
    testbed.nerf.training.sample_focal_plane_proportional_to_error=False
    testbed.nerf.training.optimize_extrinsics = False
    testbed.nerf.training.optimize_exposure = False
    testbed.nerf.training.optimize_extra_dims = False
    testbed.nerf.training.optimize_distortion = False
    testbed.nerf.training.optimize_focal_length = False
    testbed.nerf.training.include_sharpness_in_error=False
    testbed.nerf.cone_angle_constant=cfg.MODEL.CONE_ANGLE_CONSTANT
    if cfg.MODEL.DENSITY_ACTIVATION=="Exponential10x":
        testbed.nerf.density_activation = ngp.NerfActivation.Exponential10x
    else:
        testbed.nerf.density_activation = ngp.NerfActivation.Exponential

    if (not os.path.isfile(cfg.DATASET.POSES_FILENAME)) or (cfg.DATASET.POSES_SCALE<=0):
        poses_scale = 1.0
        print(" /!\ No POSES SCALE setup. This means that likely the NGP preprocessed datafile hasn't been found'")
        print("        --> setting poses_scale= 1")
    else:
        poses_scale = cfg.DATASET.POSES_SCALE

    # -- Use SDF: (volSDF definition)
    testbed.use_volsdf_in_nerf=cfg.MODEL.USE_VOL_SDF
    testbed.use_stylesdf_in_nerf=cfg.MODEL.USE_STYLE_SDF
    testbed.use_coslam_sdf_in_nerf=cfg.MODEL.USE_COSLAM_SDF
    testbed.nerf.training.truncation_distance=cfg.MODEL.SDF_TRUNCATION_DISTANCE*cfg.RENDERER.SCALE*poses_scale
    testbed.nerf.training.volsdf_beta=cfg.MODEL.SDF_BETA
    testbed.add_sdf_loss=cfg.MODEL.USE_SDF_LOSS
    testbed.add_sdf_loss_tracking=cfg.MODEL.USE_SDF_LOSS_TRACKING
    testbed.nerf.training.sdf_supervision_lambda=cfg.MODEL.SDF_SUPERVISION_LAMBDA
    testbed.nerf.training.sdf_supervision_lambda_tracking=cfg.MODEL.SDF_SUPERVISION_LAMBDA_TRACKING
    testbed.add_sdf_free_space_loss=cfg.MODEL.USE_SDF_FREE_SPACE_LOSS
    testbed.add_sdf_free_space_loss_tracking=cfg.MODEL.USE_SDF_FREE_SPACE_LOSS_TRACKING
    testbed.nerf.training.sdf_free_space_supervision_lambda=cfg.MODEL.SDF_FREE_SPACE_SUPERVISION_LAMBDA
    testbed.nerf.training.sdf_free_space_supervision_lambda_tracking=cfg.MODEL.SDF_FREE_SPACE_SUPERVISION_LAMBDA_TRACKING
    if cfg.MODEL.USE_VOL_SDF or cfg.MODEL.USE_STYLE_SDF or cfg.MODEL.USE_SDF or cfg.MODEL.USE_COSLAM_SDF:
        #TODO: assert only one sdf mode is activated
        testbed.nerf.density_activation = ngp.NerfActivation.none

    # -- Free Space supervision
    testbed.add_free_space_loss=cfg.MODEL.USE_FREE_SPACE_LOSS
    testbed.add_free_space_loss_tracking=cfg.MODEL.USE_FREE_SPACE_LOSS_TRACKING
    testbed.nerf.training.free_space_supervision_lambda=cfg.MODEL.FREE_SPACE_SUPERVISION_LAMBDA
    testbed.nerf.training.free_space_supervision_lambda_tracking=cfg.MODEL.FREE_SPACE_SUPERVISION_LAMBDA_TRACKING
    testbed.nerf.training.free_space_supervision_distance=cfg.MODEL.FREE_SPACE_SUPERVISION_DISTANCE*cfg.RENDERER.SCALE*poses_scale
    if cfg.MODEL.USE_FREE_SPACE_LOSS:
        print("## -- Free Space supervision distance: ", testbed.nerf.training.free_space_supervision_distance)

    # -- Use custom ray marching
    testbed.use_custom_ray_marching=cfg.MODEL.USE_CUSTOM_RAY_MARCHING
    testbed.nerf.training.n_samples_for_regular_sampling=cfg.MODEL.N_SAMPLES_FOR_REGULAR_POINT_SAMPLING
    dt=(cfg.RENDERER.FAR - cfg.RENDERER.NEAR)*cfg.RENDERER.SCALE*poses_scale / cfg.MODEL.N_SAMPLES_FOR_REGULAR_POINT_SAMPLING
    testbed.nerf.training.dt_for_regular_sampling=dt
    if cfg.MODEL.USE_CUSTOM_RAY_MARCHING:
        print("## -- Regular point sampling #pts: ", testbed.nerf.training.n_samples_for_regular_sampling)
        print("## -- Regular point sampling dt: ", testbed.nerf.training.dt_for_regular_sampling)

    testbed.nerf.training.use_pose_scheduler_in_mapping=cfg.MODEL.USE_POSE_SCHEDULER_IN_MAPPING
    testbed.nerf.training.pose_scheduler_coef=cfg.MODEL.POSE_SCHEDULER_COEF
    testbed.nerf.training.pose_scheduler_norm=cfg.MODEL.POSE_SCHEDULER_NORM
    testbed.nerf.training.use_gradient_clipping_for_extrinsics=cfg.MODEL.USE_GRADIENT_CLIPPING_EXTRINSICS

    # -- Depth guided sampling
    testbed.use_depth_guided_sampling=cfg.MODEL.USE_DEPTH_GUIDED_SAMPLING
    testbed.nerf.training.truncation_distance_for_depth_guided_sampling=cfg.MODEL.TRUNCATION_DISTANCE_FOR_DEPTH_GUIDED_SAMPLING*cfg.RENDERER.SCALE*poses_scale
    testbed.nerf.training.dt_for_depth_guided_sampling=\
            cfg.MODEL.TRUNCATION_DISTANCE_FOR_DEPTH_GUIDED_SAMPLING*cfg.RENDERER.SCALE*poses_scale*2/cfg.MODEL.NUM_SAMPLES_FOR_DEPTH_GUIDED_SAMPLING
    if cfg.MODEL.USE_DEPTH_GUIDED_SAMPLING:
        print("## -- truncation distance depth guided sampling: ", testbed.nerf.training.truncation_distance_for_depth_guided_sampling)
        print("## -- dt in depth guided sampling: ", testbed.nerf.training.dt_for_depth_guided_sampling)

    # -- DS Nerf Loss regulirizer
    testbed.add_DS_nerf_loss=cfg.MODEL.USE_DS_NERF_LOSS
    testbed.add_DS_nerf_loss_tracking=cfg.MODEL.USE_DS_NERF_LOSS_TRACKING
    testbed.nerf.training.DS_nerf_supervision_lambda=cfg.MODEL.DS_NERF_LOSS_LAMBDA
    testbed.nerf.training.DS_nerf_supervision_lambda_tracking=cfg.MODEL.DS_NERF_LOSS_LAMBDA_TRACKING
    testbed.nerf.training.DS_nerf_supervision_depth_sigma=cfg.MODEL.DS_NERF_LOSS_DEPTH_SIGMA*cfg.RENDERER.SCALE*poses_scale

    testbed.use_DS_nerf_loss_with_sech2_dist=cfg.MODEL.DS_NERF_USE_SECH2_DIST
    testbed.nerf.training.DS_nerf_supervision_sech2_scale=cfg.MODEL.DS_NERF_SECH2_SCALE
    if cfg.MODEL.DS_NERF_USE_SECH2_DIST:
        norm = compute_dist_norm(cfg.MODEL.DS_NERF_SECH2_SCALE, testbed.nerf.training.DS_nerf_supervision_depth_sigma)
        intA, intB = get_integrals_coefs(cfg.MODEL.DS_NERF_SECH2_SCALE, testbed.nerf.training.DS_nerf_supervision_depth_sigma)
        dist_sum = get_dist_sum(cfg.MODEL.DS_NERF_SECH2_SCALE, testbed.nerf.training.DS_nerf_supervision_depth_sigma)
        print("## -- Sech2 (norm, intA, intB, sum): ", norm, intA, intB, dist_sum)
        testbed.nerf.training.DS_nerf_supervision_sech2_norm=norm
        testbed.nerf.training.DS_nerf_supervision_sech2_int_A=intA
        testbed.nerf.training.DS_nerf_supervision_sech2_int_B=intB
        testbed.nerf.training.DS_nerf_supervision_lambda=testbed.nerf.training.DS_nerf_supervision_lambda/dist_sum
        print('## new DS loss lbda: ', testbed.nerf.training.DS_nerf_supervision_lambda)

    if cfg.MODEL.USE_DS_NERF_LOSS:
        print("## -- DS Loss depth sigma: ", testbed.nerf.training.DS_nerf_supervision_depth_sigma)

    testbed.keep_data_on_cpu=cfg.DATASET.KEEP_DATA_ON_CPU

    testbed.nerf.training.depth_loss_type = ngp.LossType.L1
    testbed.add_sdf_loss_tracking = False

    testbed.snap_to_pixel_centers = True #for faster rendering

    testbed.nerf.training.near_distance = cfg.RENDERER.NEAR
    testbed.render_near_distance = cfg.RENDERER.NEAR

    testbed.nerf.training.density_grid_decay = cfg.MODEL.DENSITY_GRID_DECAY

    testbed.use_density_in_nerf_sampling=cfg.MODEL.USE_DENSITY_IN_POINT_SAMPLING

    testbed.debug_mode=False

    testbed.use_depth_median_filter=cfg.MODEL.USE_MEDIAN_DEPTH

    if cfg.MODEL.NUM_STEPS_RESOLUTION_METERS > 0.:
        stepping_res = cfg.MODEL.NUM_STEPS_RESOLUTION_METERS #m
        nerf_steps = np.round(np.sqrt(3) / (stepping_res*cfg.RENDERER.SCALE*poses_scale))
        testbed.max_num_steps_per_ray=int(nerf_steps)
    else:
        testbed.max_num_steps_per_ray=cfg.MODEL.MAX_NUM_STEPS_PER_RAY
    print("## NERF_STEPS(): ", testbed.max_num_steps_per_ray)

    return testbed



def init_instant_ngp(cfg, instant_ngp, dataset, renderer):

    preprocessed_data_filename = cfg.DATASET.NGP_PREPROCESSED_DATA_FILENAME
    if os.path.isfile(preprocessed_data_filename):
        print(" LOADING DATA from preprocessed files")

        frame_sampling_rate = cfg.DATASET.FRAME_SAMPLING_RATE
        frame_sampling_offset = cfg.DATASET.FRAME_SAMPLING_OFFSET
        if frame_sampling_rate > 1:
            assert str(frame_sampling_rate)+"_rate" in preprocessed_data_filename
        elif frame_sampling_offset > 0:
            assert str(frame_sampling_offset)+"_offset" in preprocessed_data_filename
        else:
            assert ("transforms.json" in preprocessed_data_filename) or\
                    ("transforms_with_poses" in preprocessed_data_filename) or\
                    ("transforms_aabb" in preprocessed_data_filename)


        # -- assert scale and aabb_scale are matching
        test_data = json.load(open(preprocessed_data_filename, "r"))
        assert test_data["scale"] == cfg.RENDERER.SCALE
        assert test_data["aabb_scale"] == cfg.RENDERER.AABB_SCALE
        del test_data

        instant_ngp.load_training_data(preprocessed_data_filename)

        fx=renderer.fx
        fy=renderer.fy
        cx=renderer.cx
        cy=renderer.cy
        H=renderer.H
        W=renderer.W

        rez = [W, H]

        fov_axis = instant_ngp.fov_axis
        zoom = instant_ngp.zoom

        instant_ngp.relative_focal_length = np.array([fx/(rez[fov_axis]*zoom), fy/(rez[fov_axis]*zoom)])
        instant_ngp.screen_center = [1-cx/W, 1-cy/H]

        #TODO: this is hacky but poses are not loaded properly with automatic
        # loader so loading them manually. (for first frame)
        #sample_0 = dataset[0]
        #c2w = np.copy(sample_0['c2w'])
        #
        #instant_ngp.nerf.training.set_camera_extrinsics(
        #    0,
        #    c2w[:3,:].copy(), #c2w 3x4
        #    True, #cvt_to_ngp
        #)

    else:
        print(" LOADING DATA manually")
        instant_ngp.create_empty_nerf_dataset(
            len(dataset),
            renderer.aabb_scale,
            False,
        )

        instant_ngp.nerf.training.dataset.scale = renderer.scale
        instant_ngp.nerf.training.dataset.offset = renderer.offset

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
        scale=renderer.scale

        rez = [W, H]

        fov_axis = instant_ngp.fov_axis
        zoom = instant_ngp.zoom

        instant_ngp.relative_focal_length = np.array([fx/(rez[fov_axis]*zoom), fy/(rez[fov_axis]*zoom)])
        instant_ngp.screen_center = [1-cx/W, 1-cy/H]

        sample_0 = dataset[0]
        c2w = np.copy(sample_0['c2w'])

        for i in range(len(dataset)):
            sample = dataset[i]
            rgb = np.copy(sample['rgb'])
            depth = np.copy(sample['depth'])

            rgb = rgb.astype(np.float32)
            rgb /= 255.
            rgb[...,0:3] = srgb_to_linear(rgb[...,0:3])
            rgb[...,0:3] *= rgb[...,3:4]

            instant_ngp.nerf.training.set_image(
                i, #image_idx
                rgb,
                depth,
                scale,
            )

            instant_ngp.nerf.training.set_camera_intrinsics(
                i, fx, fy, cx, cy, k1, k2, p1, p2
            )

            instant_ngp.nerf.training.set_camera_extrinsics(
                i,
                c2w[:3,:].copy(), #c2w 3x4
                True, #cvt_to_ngp
            )

        instant_ngp.nerf.training.n_images_for_training = len(dataset)


def setup(args):
    cfg = get_cfg()
    print(" ###### Loading configs from: ", args.config)
    cfg.merge_from_file(args.config)
    if len(args.parent) > 0:
        cfg.PARENT = args.parent
    if len(cfg.PARENT) > 0:
        parent_config = cfg.PARENT
        if os.path.isfile(parent_config):
            print(" ### Loading PARENT configs from: ", parent_config)
            cfg.merge_from_file(parent_config)
        else:
            parent_config = os.path.join("/".join(args.config.split("/")[:-1]),cfg.PARENT)
            if os.path.isfile(parent_config):
                print(" ### Loading PARENT configs from: ", parent_config)
                cfg.merge_from_file(parent_config)
            else:
                print("/!\ no PARENT configs loaded !! ")
    else:
        print("/!\ no PARENT configs loaded !! ")
    if len(args.output_dir) > 0:
        cfg.META.OUTPUT_DIR = args.output_dir
    print(" ### OUTPUT_DIR: ", cfg.META.OUTPUT_DIR)
    if len(args.run_id) > 0:
        cfg.META.RUN_ID = args.run_id
    print(" ### RUN_ID: ", cfg.META.RUN_ID)
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

    if cfg.MODEL.MESH_THRESH<0:
        if cfg.MODEL.DENSITY_ACTIVATION=="Exponential10x":
            meshing_thresh = 0.25
        else:
            meshing_thresh = 2.5
    else:
        meshing_thresh=cfg.MODEL.MESH_THRESH

    # -- build dataset
    dataset = build_dataset(cfg)

    # -- build instant-NGP
    instant_ngp = build_instant_ngp(cfg)

    # -- build renderer
    renderer = build_renderer(cfg)

    # -- load data to NGP
    init_instant_ngp(cfg, instant_ngp, dataset, renderer)

    # -- build mapper
    mapper = build_mapper(cfg)
    mapper.add_renderer(renderer)
    mapper.add_instant_ngp(instant_ngp)

    # -- build tracker
    tracker = build_tracker(cfg)
    tracker.add_renderer(renderer)
    tracker.add_instant_ngp(instant_ngp)

    # -- build BA
    ba = build_ba(cfg.BUNDLE_ADJUSTMENTS.LOCAL_BA)
    ba.add_renderer(renderer)
    ba.add_instant_ngp(instant_ngp)

    # -- build runner
    runner = build_runner(cfg)
    runner.add_renderer(renderer)
    runner.add_instant_ngp(instant_ngp)
    runner.add_mapper(mapper)
    runner.add_tracker(tracker)
    runner.add_ba(ba)
    runner.run(dataset)

    res = cfg.MODEL.MESH_RESOLUTION
    instant_ngp.nerf.training.idx_images_for_mapping=list(range(0,len(dataset),5))
    instant_ngp.density_grid_culling_using_keyframes()
    mesh_output_filename=os.path.join(output_dir, 'mesh.obj')
    runner.instant_ngp.compute_and_save_marching_cubes_mesh(
        mesh_output_filename,
        [res, res, res],
        thresh=meshing_thresh
    )



if __name__=="__main__":
    args = default_argument_parser().parse_args()
    main(args)
