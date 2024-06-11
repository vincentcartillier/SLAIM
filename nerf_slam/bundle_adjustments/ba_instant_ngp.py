import os
import json
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from scipy.spatial.transform import Rotation

from .build import BUNDLE_ADJUSTER_REGISTRY

from ..losses import build_loss


__all__ = ["BundleAdjustementNGP"]

@BUNDLE_ADJUSTER_REGISTRY.register()
class BundleAdjustementNGP(object):
    """
    Instant-NGP BA.

    """
    def _parse_cfg(self, cfg):
        self.ba_iterations = cfg.NUM_ITERATIONS
        
        self.margin_sample_away_from_edge_W = cfg.MARGIN_SAMPLE_WIDTH
        self.margin_sample_away_from_edge_H = cfg.MARGIN_SAMPLE_HEIGHT
        
        self.max_grid_level_rand_training = cfg.MAX_GRID_LEVEL_RAND_TRAINING
        self.extrinsic_learning_rate = cfg.EXTRINSIC_LEARNING_RATE
        self.sample_image_proportional_to_error = cfg.SAMPLE_IMAGE_PROPORTIONAL_TO_ERROR

        self.num_rays_to_sample = cfg.NUM_RAYS_TO_SAMPLE

        self.target_batch_size = cfg.TARGET_BATCH_SIZE
        self.n_steps_between_cam_updates=cfg.N_STEPS_BETWEEN_CAM_UPDATES
        
        self.ba_mode=cfg.BA_MODE
        
        self.min_rays_per_image=cfg.MIN_RAYS_PER_IMAGE
        self.use_ray_counters_per_image=cfg.USE_RAY_COUNTERS_PER_IMAGE

        self.tracking_hyperparameters_filename=cfg.HYPERPARAMETERS_FILE
        if os.path.isfile(self.tracking_hyperparameters_filename):
            self.ba_hyperparams = json.load(open(self.tracking_hyperparameters_filename, 'r'))
            self.ba_iterations = np.sum([x['iterations'] for x in self.ba_hyperparams])
        else:
            self.ba_hyperparams = None
            print(" /!\ Did not find BA hyperparameters. ")
            raise ValueError


    def __init__(self, cfg):
        self._parse_cfg(cfg)
        self.global_iter = 0

    def add_renderer(self, renderer):
        self.renderer = renderer

    def add_instant_ngp(self, instant_ngp):
        self.instant_ngp = instant_ngp

    def init_instant_ngp_dataset(self, keyframes, motion_only):

        idx_images_for_training_slam = []
        for k in keyframes:
            idx_images_for_training_slam.append(k['index'])
        
        idx_images_for_training_slam_pose = [x for x in idx_images_for_training_slam if x!=0]

        self.instant_ngp.nerf.training.idx_images_for_mapping=idx_images_for_training_slam
        self.instant_ngp.nerf.training.idx_images_for_training_extrinsics=idx_images_for_training_slam_pose
        self.instant_ngp.nerf.training.sample_image_proportional_to_error=self.sample_image_proportional_to_error
        self.instant_ngp.nerf.training.optimize_extrinsics = True
        self.instant_ngp.nerf.training.optimize_exposure = False
        self.instant_ngp.nerf.training.optimize_extra_dims = False
        self.instant_ngp.nerf.training.optimize_distortion = False
        self.instant_ngp.nerf.training.optimize_focal_length = False
        self.instant_ngp.nerf.training.include_sharpness_in_error=False
        self.instant_ngp.nerf.training.n_steps_between_cam_updates = self.n_steps_between_cam_updates
        self.instant_ngp.nerf.training.n_steps_since_cam_update = 0
        self.instant_ngp.nerf.training.n_steps_since_error_map_update = 0
        if motion_only:
            self.instant_ngp.shall_train_encoding = False
            self.instant_ngp.shall_train_network = False
        else:
            self.instant_ngp.shall_train_encoding = True
            self.instant_ngp.shall_train_network = True
        self.instant_ngp.shall_train = True
        self.instant_ngp.max_level_rand_training = self.max_grid_level_rand_training
        self.instant_ngp.max_grid_level_factor = 2.0
        self.instant_ngp.ba_mode = self.ba_mode
        self.instant_ngp.nerf.training.use_depth_var_in_tracking_loss = False
        self.instant_ngp.reset_prep_nerf_mapping = True
        self.instant_ngp.nerf.training.error_map.is_cdf_valid = False
        self.instant_ngp.nerf.training.extrinsic_learning_rate = self.extrinsic_learning_rate
        
        self.instant_ngp.nerf.training.min_num_rays_per_image_for_pose_update_in_ba=self.min_rays_per_image
        self.instant_ngp.nerf.training.use_ray_counter_per_image_in_ba=self.use_ray_counters_per_image
        self.instant_ngp.nerf.training.reset_ray_counters_and_gradients_for_ba=self.use_ray_counters_per_image
        
        if self.num_rays_to_sample > 0:
            self.instant_ngp.nerf.training.target_num_rays_for_mapping = self.num_rays_to_sample
            self.instant_ngp.nerf.training.target_num_rays_for_ba = self.num_rays_to_sample
            self.instant_ngp.nerf.training.set_fix_num_rays_to_sample = True
        else:
            self.instant_ngp.nerf.training.set_fix_num_rays_to_sample = False


    def run(self,
            keyframes,
            writer=None,
            iterations=None,
            motion_only=False,
            ba_params=None
           ):
        r"""
        """
        # -- setup NGP
        self.init_instant_ngp_dataset(
            keyframes,
            motion_only
        )

        batch_size=self.target_batch_size #target BS -> will sample as many rays as possible
        
        if ba_params is None:
            if self.ba_hyperparams is None:
                ba_params = [
                    {"iterations":1000, 'mgl': 1.0, 'gpl':0}
                ]
            else:
                ba_params = self.ba_hyperparams

        # -- start BA
        ba_iter = 0
        all_iterations = np.sum([x['iterations'] for x in ba_params])
        for bap in ba_params:
            
            mgl = bap["mgl"]
            gpl = bap["gpl"]
            num_iterations = bap["iterations"]

            self.instant_ngp.tracking_max_grid_level = mgl
            self.instant_ngp.tracking_gaussian_pyramid_level = gpl
            
            if 'n_steps_between_cam_updates' in bap:
                n_steps_between_cam_updates = bap['n_steps_between_cam_updates']
                self.instant_ngp.nerf.training.n_steps_between_cam_updates=n_steps_between_cam_updates
            else:
                self.instant_ngp.nerf.training.n_steps_between_cam_updates=self.n_steps_between_cam_updates

            for _ in range(num_iterations):
                if (gpl==0) or (mgl==1.):
                    self.instant_ngp.map(batch_size)
                    measured_bs=self.instant_ngp.nerf.training.counters_rgb.measured_batch_size
                else:
                    self.instant_ngp.ba(batch_size)
                    measured_bs=self.instant_ngp.nerf.training.counters_rgb_ba.measured_batch_size
                loss = self.instant_ngp.loss
                loss = loss * batch_size / measured_bs

                if writer is not None:
                    iter = self.global_iter*all_iterations + ba_iter
                    writer.add_scalar("ba-loss", loss, iter)

                ba_iter+=1
       
        self.global_iter += 1
                
        return loss



