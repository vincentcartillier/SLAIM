import os
import cv2
import json
import time
import math
import torch
import numpy as np
from pathlib import Path
from imageio import imwrite
from tensorboardX import SummaryWriter
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from typing import Any, Dict, List, Optional, Tuple

from .build import RUNNER_REGISTRY

# -- linking Instant-NGP here
# /!\ should link directly in this repo as a submodule
import sys
sys.path.append("dependencies/instant-ngp/build/")
import pyngp as ngp # noqa
sys.path.append("dependencies/instant-ngp/scripts/")
from common import *



__all__ = ["RunnerInstantNGPSequentialCoarse2Fine"]

@RUNNER_REGISTRY.register()
class RunnerInstantNGPSequentialCoarse2Fine():

    def _parse_cfg(self, cfg):
        self.skip_mapping_first_image = cfg.RUNNER.SKIP_MAPPING_FIRST_IMAGE
        self.num_mapping_iterations_first_frame=cfg.RUNNER.NUM_MAPPING_ITERATIONS_FIRST_IMAGE
        self.target_batch_size_first_frame=cfg.RUNNER.TARGET_BATCH_SIZE_FIRST_IMAGE

        self.root_output_dir = cfg.META.OUTPUT_DIR
        self.name_experiment = cfg.META.NAME_EXPERIMENT
        self.run_id = cfg.META.RUN_ID

        self.poses_dirname = cfg.RUNNER.POSES_DIRNAME
        self.meshes_dirname = cfg.RUNNER.MESHES_DIRNAME
        self.mapping_dirname = cfg.RUNNER.MAPPING_VISUALS_DIRNAME
        self.tracking_dirname = cfg.RUNNER.TRACKING_VISUALS_DIRNAME
        self.model_ckpt_dirname = cfg.RUNNER.MODEL_CHECKPOINT_DIRNAME
        self.logs_dirname = cfg.RUNNER.LOGS_DIRNAME
        self.keyframe_rate = cfg.RUNNER.KEYFRAME_RATE
        self.motion_filter_thresh = cfg.RUNNER.KEYFRAME_MOTION_FILTER_THRESH
        self.const_speed_assumption = cfg.RUNNER.CONSTANT_SPEED_ASSUMPTION
        if (not os.path.isfile(cfg.DATASET.POSES_FILENAME)) or (cfg.DATASET.POSES_SCALE==0):
            self.poses_scale = 1.0
        else:
            self.poses_scale = cfg.DATASET.POSES_SCALE
        self.ba_during_mapping = cfg.MAPPER.DO_BA_WHILE_MAPPING
        self.ba_during_local_mapping = cfg.MAPPER.DO_BA_WHILE_LOCAL_MAPPING

        self.viz = cfg.RUNNER.VIZ
        self.viz_freq = cfg.RUNNER.SAVE_VIZ_EVERY

        self.verbose = cfg.RUNNER.VERBOSE
        self.do_local_ba = cfg.RUNNER.DO_ADDITIONAL_LOCAL_BA
        self.do_logs = cfg.RUNNER.SAVE_TB_LOGS
        self.add_final_giant_BA = cfg.RUNNER.ADD_FINAL_GIANT_BA

        self.do_local_finetuning = cfg.RUNNER.DO_ADDITIONAL_LOCAL_FINETUNING
        self.local_finetuning_rate=cfg.RUNNER.LOCAL_FINETUNING_RATE

        self.do_decoder_only_pretraining=cfg.RUNNER.DO_DECODER_ONLY_PRETRAINING

        self.mapping_rate=cfg.RUNNER.MAPPING_RATE
        self.init_phase=cfg.RUNNER.USE_INITIALIZATION
        self.init_phase_iterations=cfg.RUNNER.INITIALIZATION_ITERATIONS

        self.save_models=cfg.RUNNER.SAVE_MODELS
        self.save_models_rate=cfg.RUNNER.SAVE_MODELS_RATE

        self.use_gt_camera_poses_for_mapping=cfg.RUNNER.USE_GT_CAMERA_POSES_FOR_MAPPING
        self.kf_motion_based_selection=cfg.RUNNER.KF_MOTION_BASED_SELECTION
        self.kf_slection_rate=cfg.RUNNER.KF_SELECTION_RATE
        self.kf_slection_rate_max=cfg.RUNNER.KF_SELECTION_RATE_MAX

        self.save_meshes=cfg.RUNNER.SAVE_MESHES
        self.save_meshes_rate=cfg.RUNNER.SAVE_MESHES_RATE

        self.keep_data_on_cpu=cfg.DATASET.KEEP_DATA_ON_CPU

        if cfg.MODEL.DENSITY_ACTIVATION=="Exponential10x":
            self.meshing_thresh = 0.25
        else:
            self.meshing_thresh = 2.5

        self.do_local_mapping=cfg.MAPPER.DO_LOCAL_MAPPING
        self.local_mapping_iterations=cfg.MAPPER.LOCAL_MAPPING_ITERATIONS

        self.use_sharpness_in_keyframe_selection=cfg.RUNNER.USE_SHARPNESS_IN_KF_SELECTION
        self.sharpness_thresh=cfg.RUNNER.SHARPNESS_THRESH
        
        self.add_motion_only_BA_trajectory_filler=cfg.RUNNER.ADD_MOTION_ONLY_BA_TRAJECTORY_FILLER
        self.motion_only_ba_iterations=cfg.RUNNER.MOTION_ONLY_BA_ITERATIONS

    def __init__(self,cfg):
        self._parse_cfg(cfg)

        self.is_init = False

        self.motion_filter = MotionFilter(thresh=self.motion_filter_thresh)

        # -- set output paths
        self.output_dir = os.path.join(
            self.root_output_dir,
            self.name_experiment,
            str(self.run_id)
        )
        self.output_dir_poses = os.path.join(
            self.output_dir,
            self.poses_dirname,
        )
        Path(self.output_dir_poses).mkdir(parents=True, exist_ok=True)
        self.output_dir_mapping_viz = os.path.join(
            self.output_dir,
            self.mapping_dirname,
        )
        Path(self.output_dir_mapping_viz).mkdir(parents=True, exist_ok=True)
        self.output_dir_tracking_viz = os.path.join(
            self.output_dir,
            self.tracking_dirname,
        )
        Path(self.output_dir_tracking_viz).mkdir(parents=True, exist_ok=True)
        self.output_dir_models = os.path.join(
            self.output_dir,
            self.model_ckpt_dirname,
        )
        Path(self.output_dir_models).mkdir(parents=True, exist_ok=True)
        self.output_dir_logs = os.path.join(
            self.output_dir,
            self.logs_dirname,
        )
        Path(self.output_dir_logs).mkdir(parents=True, exist_ok=True)
        self.output_dir_meshes = os.path.join(
            self.output_dir,
            self.meshes_dirname,
        )
        Path(self.output_dir_meshes).mkdir(parents=True, exist_ok=True)


        # -- create logger
        if self.do_logs:
            self.writer = SummaryWriter(logdir=self.output_dir_logs)
        else:
            self.writer = None

    def add_instant_ngp(self, instant_ngp):
        self.instant_ngp = instant_ngp

    def add_mapper(self, mapper):
        self.mapper = mapper

    def add_renderer(self, renderer):
        self.renderer = renderer

    def add_tracker(self, tracker):
        self.tracker = tracker

    def add_ba(self, ba):
        self.ba = ba

    def render_depth(self, c2w):
        screenshot_spp = 2
        self.instant_ngp.set_nerf_camera_matrix(np.matrix(c2w)[:3,:])
        self.instant_ngp.render_mode = ngp.Depth
        image = self.instant_ngp.render(
            self.renderer.W,
            self.renderer.H,
            screenshot_spp,
            True
        )
        depth = image[:,:,0]
        self.instant_ngp.render_mode = ngp.Shade
        return depth

    def render_rgb(self, c2w):

        def linear_to_srgb(img):
            limit = 0.0031308
            return np.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)

        screenshot_spp = 2
        self.instant_ngp.set_nerf_camera_matrix(np.matrix(c2w)[:3,:])

        img = self.instant_ngp.render(
            self.renderer.W,
            self.renderer.H,
            screenshot_spp,
            True
        )

        if img.shape[2] == 4:
            img = np.copy(img)
            # Unmultiply alpha
            img[...,0:3] = np.divide(img[...,0:3], img[...,3:4], out=np.zeros_like(img[...,0:3]), where=img[...,3:4] != 0)
            img[...,0:3] = linear_to_srgb(img[...,0:3])
        else:
            img = linear_to_srgb(img)

        img = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        img = img[:,:,:3]

        return img



    def save_frame(self,rgb, depth,c2w,filename):
        prev_bg = self.instant_ngp.background_color
        prev_transmittance = self.instant_ngp.nerf.render_min_transmittance

        self.instant_ngp.background_color = [0.0, 0.0, 0.0, 1.0]
        self.instant_ngp.nerf.render_min_transmittance = 1e-4
        self.instant_ngp.shall_train = False
        self.instant_ngp.nerf.render_with_camera_distortion = False
        self.instant_ngp.render_ground_truth = False

        rec_depth = self.render_depth(c2w)
        rec_color = self.render_rgb(c2w)
        self.save_reconstruction_snapshot(
            rgb,
            depth,
            rec_color,
            rec_depth,
            filename
        )

        self.instant_ngp.background_color = prev_bg
        self.instant_ngp.nerf.render_min_transmittance=prev_transmittance
        self.instant_ngp.nerf.render_with_camera_distortion = False


    def save_reconstruction_snapshot(self,rgb, depth, rec_color, rec_depth, filename):
        rgb_reco = rec_color
        depth_reco = rec_depth
        depth_reco -= depth_reco.min()
        depth_reco /= depth_reco.max()
        depth_reco *= 255
        depth_reco = depth_reco.astype(np.uint8)
        depth_reco = cv2.cvtColor(depth_reco, cv2.COLOR_GRAY2RGB)

        canvas = cv2.hconcat([rgb_reco, depth_reco])

        depth_OG = depth
        depth_OG -= depth_OG.min()
        depth_OG /= depth_OG.max()
        depth_OG *= 255
        depth_OG = depth_OG.astype(np.uint8)
        depth_OG = cv2.cvtColor(depth_OG, cv2.COLOR_GRAY2RGB)
        color_OG = rgb[:,:,:3]

        canvas_OG = cv2.hconcat([color_OG, depth_OG])

        canvas = cv2.vconcat([canvas_OG, canvas])

        imwrite(filename, canvas)


    def save_poses(self,
                   poses: List,
                   filename: str):
        poses_list = poses
        poses_list = [x.tolist() for x in poses_list]
        json.dump(poses_list, open(filename,'w'))

    
    def save_keyframes(self, keyframes, filename):
        kfs = [x["index"] for x in keyframes]
        json.dump(kfs, open(filename,'w'))


    def eval_rotations(self, dataset, idx_images_for_training_extrinsics,eval_rate):
        angles = []
        for x in idx_images_for_training_extrinsics[::eval_rate]:
            sample = dataset[x]
            gt_c2w = sample['c2w']
            est_c2w = self.instant_ngp.nerf.training.get_camera_extrinsics(x)
            gt_R = np.array(gt_c2w[:3,:3])
            est_R = np.array(est_c2w[:3,:3])
            R = gt_R.T @ est_R
            R = Rotation.from_matrix(R)
            r = R.as_rotvec(degrees=True)
            tracking_angle_error = np.linalg.norm(r)
            angles.append(tracking_angle_error)
        return np.mean(angles)

    def measure_image_sharpness(self, image):
        tmp = image[:,:,:3].copy()
        gray = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        return fm

    
    def pose_trajectory_filler(self, poses, poses_delta, keyframes):
        #import pdb; pdb.set_trace()
        N = len(poses_delta)
        M = len(keyframes)
        keyframe_ids = [x["index"] for x in keyframes]
        keyframe_index = 0
        keyframe_id_next = keyframe_ids[keyframe_index]
        new_poses = {}
        for i in range(N):
            if i == keyframe_id_next:
                new_poses[i] = poses[i]
                keyframe_index += 1
                if keyframe_index < M:
                    keyframe_id_next = keyframe_ids[keyframe_index]
                else:
                    keyframe_id_next = -1
            else:
                kf_frame_id = keyframe_ids[keyframe_index-1] #always have cam_id=0 as a KF
                c2w_key = poses[kf_frame_id]
                if c2w_key.shape[0] < 4:
                    c2w_key = np.pad(
                        c2w_key,
                        pad_width=[(0,1), (0,0)]
                    )
                    c2w_key[3,3] = 1.0
                delta = poses_delta[i] 
                new_poses[i] = delta @ c2w_key
        new_poses = [new_poses[i][:3,:] for i in range(N)]
        return new_poses

    
    def motion_only_BA(self, poses, keyframes):
        N = len(poses)
        M = len(keyframes)
        
        keyframe_ids = [x["index"] for x in keyframes]
        keyframe_index = 1
        keyframe_id_next = keyframe_ids[keyframe_index]
        
        mapping_ids=[0]
        tracking_ids=[]
        for i in tqdm(range(1,N)):
            if i==keyframe_id_next:
                mapping_ids.append(i)
                loss = self.mapper.optimize_map(
                    [],
                    None,
                    None,
                    None,
                    None,
                    global_iter=-1,
                    writer=None,
                    mapping_iterations=self.motion_only_ba_iterations,
                    BA=True,
                    train_grid=False,
                    train_decoder=False,
                    keyframe_selection_method=None,
                    idx_images_for_mapping = mapping_ids,
                    idx_images_for_tracking = tracking_ids,
                )
                mapping_ids = [i]
                tracking_ids = []
                keyframe_index += 1
                if keyframe_index < M:
                    keyframe_id_next = keyframe_ids[keyframe_index]
                else:
                    keyframe_id_next = -1
            else:
                mapping_ids.append(i)
                tracking_ids.append(i)

        if len(tracking_ids) > 0:
            loss = self.mapper.optimize_map(
                [],
                None,
                None,
                None,
                None,
                global_iter=-1,
                writer=None,
                mapping_iterations=self.motion_only_ba_iterations,
                BA=True,
                train_grid=False,
                train_decoder=False,
                keyframe_selection_method=None,
                idx_images_for_mapping = mapping_ids,
                idx_images_for_tracking = tracking_ids,
            )
 
        camera_poses_ngp = []
        for i in range(N):
            c2w = self.instant_ngp.nerf.training.get_camera_extrinsics(i)
            camera_poses_ngp.append(c2w)
        return camera_poses_ngp
        
 



    def run(self, dataset):
        r"""
        TODO
        """
        intrinsics = [self.renderer.fx, self.renderer.fy, self.renderer.cx, self.renderer.cy]

        num_images = len(dataset)

        keyframes = []
        camera_poses = []
        camera_poses_delta = []

        # -- map first image
        sample = dataset[0]

        rgb = sample['rgb']
        depth = sample['depth']
        c2w = sample['c2w']
        cam_id = sample['index']

        if self.keep_data_on_cpu:
            self.instant_ngp.send_image_to_gpu(0)

        if not self.skip_mapping_first_image:
            first_mapping_loss = self.mapper.optimize_map(
                keyframes,
                rgb,
                depth,
                c2w,
                cam_id,
                global_iter=0,
                writer=self.writer,
                mapping_iterations=self.num_mapping_iterations_first_frame,
                target_batch_size=self.target_batch_size_first_frame
            )

            depth_reco = self.render_depth(c2w)
            mask = depth>0
            depth_reco_metric = np.mean(np.abs(depth_reco-depth)[mask]) / self.poses_scale

            filename = os.path.join(
                self.output_dir_mapping_viz,
                f'map_first_frame_0.jpg'
            )
            self.save_frame(rgb,depth.copy(),c2w,filename)
            if self.verbose:
                print("First Mapping loss: ", first_mapping_loss,
                      " | depth reco err (m): ",depth_reco_metric,
                      " | num rays sampled: ", self.instant_ngp.rays_per_batch)
        else:
            first_mapping_loss = -1.0

        if self.save_models:
            filename_state = os.path.join(
                self.output_dir_models,
                f'model_0.msgpack'
            )
            self.instant_ngp.save_snapshot(filename_state, False)


        if self.save_meshes:
           res =  256
           mesh_output_filename = os.path.join(
               self.output_dir_meshes,
               f'mesh_0.obj'
           )
           self.instant_ngp.compute_and_save_marching_cubes_mesh(
               mesh_output_filename,
               [res, res, res],
               thresh=self.meshing_thresh
           )


        if self.writer is not None:
            self.writer.add_scalar("frame-mapping-loss", first_mapping_loss, 0)

        if not self.kf_motion_based_selection:
            keyframes.append(
                {
                    'index':cam_id,
                    'rgb':rgb,
                    'depth':depth,
                    'c2w':c2w,
                    'loss': first_mapping_loss,
                }
            )
        elif self.motion_filter.track(0, intrinsics, c2w, None):
            keyframes.append(
                {
                    'index':cam_id,
                    'rgb':rgb,
                    'depth':depth,
                    'c2w':c2w,
                    'loss': first_mapping_loss,
                }
            )
        else:
            print("We must keep the first frame as a KF")
            raise ValueError

        camera_poses.append(c2w)
        camera_poses_delta.append(np.eye(4))

        prev_c2w = c2w

        if not self.verbose:
            track_bar = tqdm(desc="SLAM", total=num_images, unit="step")
            track_bar.reset()
            tqdm_last_update = 0
            old_training_step = 0

        loss = -1
        n_since_kf = 0
        for i in range(1, num_images):
            
            # -- -- -- --
            # -- -- -- --
            # -- -- -- --
            # -- init
            if self.verbose:
                print(" \n iter: ", i)

            if self.keep_data_on_cpu:
                self.instant_ngp.send_image_to_gpu(i)

            sample = dataset[i]
            cam_id = sample['index']
            rgb = sample['rgb']
            depth = sample['depth']
            gt_c2w = sample['c2w']

            cur_est_c2w = prev_c2w

            if self.const_speed_assumption and (i > 1):
                pre_c2w = cur_est_c2w.copy()
                if pre_c2w.shape[0] < 4:
                    pre_c2w = np.pad(pre_c2w, pad_width=[(0,1), (0,0)])
                    pre_c2w[3,3] = 1.0
                a = camera_poses[i-2]
                if a.shape[0] < 4:
                    a = np.pad(a, pad_width=[(0,1), (0,0)])
                    a[3,3] = 1.0
                a = np.linalg.inv(a)
                delta = pre_c2w @ a
                cur_est_c2w = delta @ pre_c2w

            if self.verbose:
                if not np.any(np.isinf(gt_c2w)):
                    tracking_pose_error = np.linalg.norm( cur_est_c2w[:3,3] - gt_c2w[:3,3]) / self.poses_scale
                else: tracking_pose_error = -1
                print("            ---> before Tracking mean abs error : ", tracking_pose_error)

            # -- -- -- --
            # -- -- -- --
            # -- -- -- --
            # -- tracking
            cur_est_c2w, tracking_loss = self.tracker.optimize_camera(
                cur_est_c2w,
                cam_id,
                global_iter=i,
                writer=self.writer
            )

            # -- save relative pose of non-KF
            c2w_key = keyframes[-1]["c2w"]
            delta_pose = cur_est_c2w @ np.linalg.inv(c2w_key) #TODO make sure c2w_key is 4x4
            camera_poses_delta.append(delta_pose)

            if not np.any(np.isinf(gt_c2w)):
                tracking_pose_error = np.linalg.norm( cur_est_c2w[:3,3] - gt_c2w[:3,3]) / self.poses_scale
            else: tracking_pose_error = -1

            if self.writer is not None:
                self.writer.add_scalar("frame-tracking-loss", tracking_loss, i)
                self.writer.add_scalar("tracking-mean-abs-error", tracking_pose_error, i)

            if self.verbose:
                print("                   ---> Tracking mean abs error : ", tracking_pose_error)
                print("Tracking final loss: ", tracking_loss)

            if self.viz:
                if i%self.viz_freq==0:
                    filename = os.path.join(
                        self.output_dir_tracking_viz,
                        f'track_{i}.jpg'
                    )
                    self.save_frame(
                        np.copy(rgb),
                        np.copy(depth),
                        cur_est_c2w,
                        filename
                    )

            if self.use_gt_camera_poses_for_mapping:
                self.instant_ngp.nerf.training.set_camera_extrinsics(
                    cam_id,
                    gt_c2w[:3,:].copy(), #c2w 3x4
                    True, #cvt_to_ngp
                )
                cur_est_c2w = gt_c2w.copy()


            # -- -- -- --
            # -- -- -- --
            # -- -- -- --
            # -- Mapping + local BA
            if self.writer is not None:
                self.writer.add_scalar("num_keyframes", len(keyframes), i)

            if self.verbose:
                print("                          # frames in keyframes : ", len(keyframes))

            if self.do_decoder_only_pretraining:
                loss = self.mapper.optimize_map(
                    keyframes,
                    rgb,
                    depth,
                    cur_est_c2w,
                    cam_id,
                    global_iter=i,
                    writer=None,
                    BA=False,
                    train_grid=False,
                    mapping_iterations=100,
                )
            

            do_mapping = (i%self.mapping_rate == 0) or (self.init_phase and (i<self.init_phase_iterations))
            do_local_mapping = self.do_local_mapping and not (self.init_phase and (i<self.init_phase_iterations))

            if do_mapping:

                if do_local_mapping:
                    loss = self.mapper.optimize_map(
                        keyframes,
                        rgb,
                        depth,
                        cur_est_c2w,
                        cam_id,
                        global_iter=i,
                        writer=self.writer,
                        BA=self.ba_during_local_mapping,
                        mapping_iterations=self.local_mapping_iterations,
                        keyframe_selection_method="last",
                        train_decoder=False
                    )

                loss = self.mapper.optimize_map(
                    keyframes,
                    rgb,
                    depth,
                    cur_est_c2w,
                    cam_id,
                    global_iter=i,
                    writer=self.writer,
                    BA=self.ba_during_mapping
                )

                cur_est_c2w = self.instant_ngp.nerf.training.get_camera_extrinsics(cam_id)
                if not np.any(np.isinf(gt_c2w)):
                    tracking_pose_error = np.linalg.norm( cur_est_c2w[:3,3] - gt_c2w[:3,3]) / self.poses_scale
                else: tracking_pose_error = -1

                # - get reco metrics PSNR + depth err
                if (self.writer is not None) or self.verbose:
                    depth_reco = self.render_depth(cur_est_c2w)
                    mask = depth>0
                    depth_reco_metric = np.mean(np.abs(depth_reco-depth)[mask]) / self.poses_scale
                    rgb_reco = self.render_rgb(cur_est_c2w)
                    A = np.clip(rgb[...,:3].astype(np.float32) / 255., 0.0, 1.0)
                    R = np.clip(rgb_reco[...,:3].astype(np.float32) / 255., 0.0, 1.0)
                    mse = float(compute_error("MSE", A, R))
                    psnr = mse2psnr(mse)

                if self.writer is not None:
                    self.writer.add_scalar("frame-mapping-loss", loss, i)
                    self.writer.add_scalar("BA-mean-abs-error", tracking_pose_error, i)
                    self.writer.add_scalar("cur_frame_psnr", psnr, i)
                    self.writer.add_scalar("depth_rec_err", depth_reco_metric, i)

                if self.verbose:
                    print("                 ---> Mapping ba mean abs error : ", tracking_pose_error)
                    print("Mapping final loss: ", loss,
                          " | PSNR: ", psnr,
                          " | depth err: ", depth_reco_metric,
                          " | num rays sampled: ", self.instant_ngp.rays_per_batch)


            # -- -- -- --
            # -- -- -- --
            # -- Keyframe selection
            if self.kf_motion_based_selection:
                depth_reco = self.render_depth(cur_est_c2w)
                select_kf = self.motion_filter.track(i, intrinsics, cur_est_c2w, depth_reco)
            elif n_since_kf >= (self.kf_slection_rate-1):
                if self.use_sharpness_in_keyframe_selection and (n_since_kf<self.kf_slection_rate_max):
                    sharpness = dataset.sharpness[i]
                    if sharpness > self.sharpness_thresh:
                        select_kf = True
                    else:
                        select_kf = False
                else:
                    select_kf = True
            else:
                select_kf = False

            if select_kf:
                if cur_est_c2w.shape[0] < 4:
                    cur_est_c2w = np.pad(
                        cur_est_c2w,
                        pad_width=[(0,1), (0,0)]
                    )
                    cur_est_c2w[3,3] = 1.0
                keyframes.append(
                    {
                        'index':cam_id,
                        'rgb':rgb,
                        'depth':depth,
                        'c2w': cur_est_c2w,
                        'loss': loss,
                    }
                )
                if self.verbose:
                    print(" /!\/!\ ADDING KF !! -> ", len(keyframes))
                # -- -- -- --
                # -- -- -- --
                # -- Local BA
                if self.do_local_ba:
                    if len(keyframes) > 8:
                        loss = self.ba.run(
                            keyframes,
                            writer=self.writer
                        )
                        if self.verbose:
                            print("BA final loss: ", loss)
                n_since_kf = 0
            else:
                n_since_kf +=1


            # -- -- -- --
            # -- -- -- --
            # -- -- -- --
            # -- local finetuning
            if self.do_local_finetuning:
                if i%self.local_finetuning_rate==0:
                    first_mapping_loss = self.mapper.optimize_map(
                       [],
                       rgb,
                       depth,
                       cur_est_c2w,
                       cam_id,
                       global_iter=i,
                       writer=self.writer,
                       train_grid=False,
                       mapping_iterations=100,
                    )


            camera_poses.append(cur_est_c2w)

            prev_c2w = cur_est_c2w.copy()

            if self.viz:
                if i%self.viz_freq==0:
                    filename = os.path.join(
                        self.output_dir_mapping_viz,
                        f'map_{i}.jpg'
                    )
                    self.save_frame(rgb,depth,cur_est_c2w,filename)


            if not self.verbose:
                now = time.monotonic()
                if now - tqdm_last_update > 0.1:
                    track_bar.update(i - old_training_step)
                    num_rays_per_batch=self.instant_ngp.ray_counter
                    angle_error=self.eval_rotations(dataset,[i],1)
                    if do_mapping:
                        measured_bs=self.instant_ngp.nerf.training.counters_rgb.measured_batch_size
                        measured_bs_bc=self.instant_ngp.nerf.training.counters_rgb.measured_batch_size_before_compaction
                    else:
                        measured_bs=self.instant_ngp.nerf.training.counters_rgb_tracking.measured_batch_size
                        measured_bs_bc=self.instant_ngp.nerf.training.counters_rgb_tracking.measured_batch_size_before_compaction
                    if np.abs(self.mapper.target_batch_size*16 - measured_bs_bc) < 1024:
                        print(self.mapper.target_batch_size*16, measured_bs_bc)
                    rays_per_batch=self.instant_ngp.nerf.training.counters_rgb.rays_per_batch
                    if num_rays_per_batch>0:
                        samples_per_ray = measured_bs / num_rays_per_batch
                    else:
                        samples_per_ray = -1
                    track_bar.set_postfix(
                        loss=loss,
                        Nrays=num_rays_per_batch,
                        spls_ray=samples_per_ray,
                        ate=round(tracking_pose_error,3),
                        Angle=round(angle_error,2),
                        KF=len(keyframes)
                    )
                    old_training_step = i
                    tqdm_last_update = now


            if self.save_models:
                if i%self.save_models_rate==0:
                    filename_state = os.path.join(
                        self.output_dir_models,
                        f'model_{i}.msgpack'
                    )
                    self.instant_ngp.save_snapshot(filename_state, False)

            if self.save_meshes:
                if i%self.save_meshes_rate==0:
                    res =  256
                    mesh_output_filename = os.path.join(
                        self.output_dir_meshes,
                        f'mesh_{i}.obj'
                    )
                    self.instant_ngp.compute_and_save_marching_cubes_mesh(
                        mesh_output_filename,
                        [res, res, res],
                        thresh=self.meshing_thresh
                    )

            if i%100==0:
                poses_filename = os.path.join(
                    self.output_dir_poses,
                    f'poses_{i}.json'
                )
                self.save_poses(camera_poses, poses_filename)
                
                keyframes_filename = os.path.join(
                    self.output_dir_poses,
                    f'keyframes_{i}.json'
                )
                self.save_keyframes(keyframes, keyframes_filename)


            if self.keep_data_on_cpu and not select_kf:
                self.instant_ngp.remove_image_from_gpu(i)

        # save delta poses (for DEBUGING)
        filename = os.path.join(
            self.output_dir_poses,
            f'final_delta_poses.json'
        )
        self.save_poses(camera_poses_delta, filename)
                
        keyframes_filename = os.path.join(
            self.output_dir_poses,
            f'keyframes_final.json'
        )
        self.save_keyframes(keyframes, keyframes_filename)

        camera_poses_ngp = []
        for i in range(num_images):
            c2w = self.instant_ngp.nerf.training.get_camera_extrinsics(i)
            camera_poses_ngp.append(c2w)
        filename = os.path.join(
            self.output_dir_poses,
            f'final_poses_ngp.json'
        )
        self.save_poses(camera_poses_ngp, filename)

        filename_state = os.path.join(
            self.output_dir_models,
            f'model_final.msgpack'
        )
        self.instant_ngp.save_snapshot(filename_state, False)

        camera_poses_ngp_rec = self.pose_trajectory_filler(camera_poses_ngp, camera_poses_delta, keyframes)
        filename = os.path.join(
            self.output_dir_poses,
            f'final_poses_ngp_rec.json'
        )
        self.save_poses(camera_poses_ngp_rec, filename)
        
        if self.add_motion_only_BA_trajectory_filler:
            camera_poses_ngp_rec_motion_BA = self.motion_only_BA(
                camera_poses_ngp_rec,
                keyframes
            )
            filename = os.path.join(
                self.output_dir_poses,
                f'final_poses_ngp_rec_motionBA.json'
            )
            self.save_poses(camera_poses_ngp_rec_motion_BA, filename)
 

        # -- Final giant BA here
        if self.add_final_giant_BA:
            loss = self.ba.run(
                keyframes,
                ba_params=[
                    {"iterations":10000, 'mgl': 0.5, 'gpl':1},
                    {"iterations":10000, 'mgl': 1.0, 'gpl':0},
                ],
                writer=self.writer
            )
            if self.verbose:
                print("BA final loss: ", loss)

            camera_poses_ngp = []
            for i in range(num_images):
                c2w = self.instant_ngp.nerf.training.get_camera_extrinsics(i)
                camera_poses_ngp.append(c2w)
            filename = os.path.join(
                self.output_dir_poses,
                f'final_poses_ngp_after_final_giant_BA.json'
            )
            self.save_poses(camera_poses_ngp, filename)

            filename_state = os.path.join(
                self.output_dir_models,
                f'model_final_after_final_giant_BA.msgpack'
            )
            self.instant_ngp.save_snapshot(filename_state, False)

            camera_poses_ngp_rec = self.pose_trajectory_filler(camera_poses_ngp, camera_poses_delta, keyframes)
            filename = os.path.join(
                self.output_dir_poses,
                f'final_poses_ngp_after_final_giant_BA_rec.json'
            )
            self.save_poses(camera_poses_ngp_rec, filename)
        
            if self.add_motion_only_BA_trajectory_filler:
                camera_poses_ngp_rec_motion_BA = self.motion_only_BA(
                    camera_poses_ngp_rec,
                    keyframes
                )
                filename = os.path.join(
                    self.output_dir_poses,
                    f'final_poses_ngp_after_final_giant_BA_rec_motionBA.json'
                )
                self.save_poses(camera_poses_ngp_rec_motion_BA, filename)
 




class MotionFilter:
    def __init__(self, thresh=2.5):

        self.thresh = thresh
        self.count = 0

        self.prev_pose = None

    def measure_motion(self, intrinsics, pose, depth):
        # build inputs
        c2w_0 = self.prev_pose.copy()
        if c2w_0.shape[0] < 4:
            c2w_0 = np.pad(
                c2w_0,
                pad_width=[(0,1), (0,0)]
            )
            c2w_0[3,3] = 1.0
        c2w_0[:3,1] *= -1
        c2w_0[:3,2] *= -1

        c2w_1 = pose.copy()
        if c2w_1.shape[0] < 4:
            c2w_1 = np.pad(
                c2w_1,
                pad_width=[(0,1), (0,0)]
            )
            c2w_1[3,3] = 1.0
        c2w_1[:3,1] *= -1
        c2w_1[:3,2] *= -1

        depth_rec_1 = depth
        H, W = depth_rec_1.shape

        fx, fy, cx, cy = intrinsics

        K = np.array(
            [
                [fx, .0, cx],
                [.0, fy, cy],
                [.0, .0, 1.]
            ]
        )

        K_inv = np.linalg.inv(K)

        # unproject
        ux = np.arange(W)
        uy = np.arange(H)
        u, v = np.meshgrid(ux, uy)

        pixel_coords = np.asarray(
            [
                u.flatten(),
                v.flatten(),
                np.ones(H*W)
            ]
        )
        pixel_depth = depth_rec_1.flatten()

        mask_depth = pixel_depth > 0.0

        pixel_coords = pixel_coords[:,mask_depth]
        pixel_depth = pixel_depth[mask_depth]

        N = len(pixel_depth)

        cam_coords = K_inv @ pixel_coords * pixel_depth

        cam_coords_homo = np.concatenate(
            [cam_coords, np.ones(N).reshape(1,-1)],
            axis=0
        )

        world_coords = c2w_1 @ cam_coords_homo

        # - re-projection
        w2c = np.linalg.inv(c2w_0)

        new_cam_coords = w2c @ world_coords

        new_cam_coords = new_cam_coords[:3,:]

        new_pixel_coords = K @ new_cam_coords

        s = new_pixel_coords[2,:] + 1e-8

        new_uv = new_pixel_coords[:2,:] / s

        new_uv = np.round(new_uv)

        new_uv = new_uv.astype(np.int32)

        mask_proj = (new_uv[0,:] < W) *\
                    (new_uv[0,:] >= 0) *\
                    (new_uv[1,:] < H) *\
                    (new_uv[1,:] >= 0)

        mask_proj = mask_proj & (s > 0)

        new_uv = new_uv[:,mask_proj]

        M = new_uv.shape[1]

        pre_uv = pixel_coords[:2,:]
        pre_uv = pre_uv[:,mask_proj]

        flow = pre_uv - new_uv # flow jj->ii

        flow_map = np.zeros((H,W,2))
        flow_map[new_uv[1], new_uv[0], :] = flow.T
        mask = np.zeros((H,W), dtype=np.bool)
        mask[new_uv[1], new_uv[0]] = 1

        mask = mask[3::8,3::8]
        flow_map = flow_map[3::8,3::8,:]
        flow_map /= 8.0

        return flow_map, mask



    def track(self, tstamp, intrinsics, pose, depth_reco):
        r"""
        """

        ### always add first frame to the depth video ###
        if tstamp == 0:
            self.prev_pose = pose
            return True

        ### only add new frame if there is enough motion ###
        else:
            flow, mask = self.measure_motion(
                intrinsics,
                pose,
                depth_reco
            )

            delta = np.linalg.norm(flow, axis=-1)
            delta = delta[mask]

            # check motion magnitue / add new frame to video
            if delta.mean() > self.thresh:
                self.count = 0
                self.prev_pose = pose
                return True
            else:
                self.count += 1
                return False




