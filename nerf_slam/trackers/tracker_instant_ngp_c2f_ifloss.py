import os
import json
import numpy as np

from .build import TRACKER_REGISTRY

# -- linking Instant-NGP here
import sys
sys.path.append("dependencies/instant-ngp/build/")
import pyngp as ngp # noqa

__all__ = ["TrackerInstantNGPCoarse2FineIfLoss"]


@TRACKER_REGISTRY.register()
class TrackerInstantNGPCoarse2FineIfLoss(object):

    def _parse_cfg(self, cfg):
        self.device = cfg.MODEL.DEVICE
        self.tracking_iterations = cfg.TRACKER.TRACKING_ITERATIONS
        self.tracking_mode = cfg.TRACKER.INSTANT_NGP_TRACKING_MODE
        self.use_depth_var_in_tracking_loss=cfg.TRACKER.USE_DEPTH_VAR_IN_DEPTH_LOSS
        self.use_if_loss=cfg.TRACKER.USE_IF_LOSS

        self.lr = cfg.TRACKER.LR
        self.separate_pos_and_rot_lr = cfg.TRACKER.SEPARATE_POS_AND_ROT_LR
        self.pos_lr = cfg.TRACKER.POS_LR
        self.rot_lr = cfg.TRACKER.ROT_LR

        self.margin_sample_away_from_edge_W = cfg.TRACKER.MARGIN_SAMPLE_WIDTH
        self.margin_sample_away_from_edge_H = cfg.TRACKER.MARGIN_SAMPLE_HEIGHT

        self.num_rays_to_sample = cfg.TRACKER.NUM_RAYS_TO_SAMPLE

        self.target_batch_size = cfg.TRACKER.TARGET_BATCH_SIZE

        self.tracking_hyperparameters_filename=cfg.TRACKER.TRACKING_HYPERPARAMETERS_FILE
        if os.path.isfile(self.tracking_hyperparameters_filename):
            self.tracking_hyperparams = json.load(open(self.tracking_hyperparameters_filename, 'r'))
            self.tracking_iterations = np.sum([x['iterations'] for x in self.tracking_hyperparams])
        else:
            self.tracking_hyperparams = None
            print(" Cannot find tracking hyperparams")
            raise ValueError

        self.debug_tracking = cfg.TRACKER.DEBUG
        if self.debug_tracking:
            self.cpt_images = 0
            self.debug_output_dir = os.path.join(
                cfg.META.OUTPUT_DIR,
                cfg.META.NAME_EXPERIMENT,
                str(cfg.META.RUN_ID),
                cfg.RUNNER.LOGS_DIRNAME,
                "tracking_debug"
            )
            from pathlib import Path
            Path(self.debug_output_dir).mkdir(parents=True, exist_ok=True)


    def __init__(self, cfg):
        self._parse_cfg(cfg)


    def add_instant_ngp(self, instant_ngp):
        self.instant_ngp = instant_ngp

    def add_renderer(self, renderer):
        self.renderer = renderer

    def init_instant_ngp(self, cam_id):
        self.instant_ngp.nerf.training.sample_image_proportional_to_error=False
        self.instant_ngp.nerf.training.optimize_extrinsics = False
        self.instant_ngp.nerf.training.optimize_exposure = False
        self.instant_ngp.nerf.training.optimize_extra_dims = False
        self.instant_ngp.nerf.training.optimize_distortion = False
        self.instant_ngp.nerf.training.optimize_focal_length = False
        self.instant_ngp.nerf.training.include_sharpness_in_error=False
        self.instant_ngp.nerf.training.idx_images_for_training_extrinsics=[]
        self.instant_ngp.nerf.training.idx_images_for_mapping = []
        self.instant_ngp.nerf.training.indice_image_for_tracking_pose = cam_id
        self.instant_ngp.nerf.training.n_steps_between_cam_updates = 1
        self.instant_ngp.nerf.training.n_steps_since_cam_update = 0
        self.instant_ngp.nerf.training.n_steps_since_error_map_update = 0
        self.instant_ngp.nerf.training.sample_away_from_border_margin_h_tracking=self.margin_sample_away_from_edge_H
        self.instant_ngp.nerf.training.sample_away_from_border_margin_w_tracking=self.margin_sample_away_from_edge_W
        self.instant_ngp.shall_train_encoding = False
        self.instant_ngp.shall_train_network = False
        self.instant_ngp.shall_train = True
        self.instant_ngp.max_level_rand_training = False
        self.instant_ngp.nerf.training.use_depth_var_in_tracking_loss = self.use_depth_var_in_tracking_loss
        self.instant_ngp.tracking_mode = self.tracking_mode
        if self.num_rays_to_sample > 0:
            self.instant_ngp.nerf.training.target_num_rays_for_tracking = self.num_rays_to_sample
            self.instant_ngp.nerf.training.set_fix_num_rays_to_sample = True
        else:
            self.instant_ngp.nerf.training.set_fix_num_rays_to_sample = False

        if self.separate_pos_and_rot_lr:
            self.instant_ngp.nerf.training.extrinsic_learning_rate_pos=self.pos_lr
            self.instant_ngp.nerf.training.extrinsic_learning_rate_rot=self.rot_lr
        else:
            self.instant_ngp.nerf.training.extrinsic_learning_rate_pos = self.lr
            self.instant_ngp.nerf.training.extrinsic_learning_rate_rot = self.lr


    def optimize_camera(self,
                        cur_c2w,
                        cam_id,
                        global_iter = 0,
                        writer=None):

        self.instant_ngp.nerf.training.set_camera_extrinsics(
            cam_id,
            cur_c2w[:3,:].copy(), #c2w 3x4
            True, #cvt_to_ngp
        )

        self.init_instant_ngp(cam_id)

        batch_size=self.target_batch_size # - not really relevant here since we're setting the#rays to sample - this is only a target BS

        if self.debug_tracking:
            # init debug info
            debug_logs = {
                'loss': [],
                'ite': [],
                'batch_size': [],
                'measured_bs': [],
                'measured_bs_bc': [],
                'cur_loss': [], #diff of loss if use n_steps_before_gradients
                'cur_c2w': [],
                'mgl': [],
                'gpl': [],
                'pos_lr': [],
                'rot_lr': [],
                'n_step_bf_update': [],
                'cur_step_bf_update': [],
                'ray_counter': [],
                'rays_per_batch': [],
            }

        tracking_iter = 0
        final_c2w = self.instant_ngp.nerf.training.get_camera_extrinsics(cam_id)
        for e in self.tracking_hyperparams:
            if 'mlg' in e:
                mgl = e['mgl']
                self.instant_ngp.tracking_max_grid_level = mgl
            else:
                mgl = -1
            if 'gpl' in e:
                gpl = e['gpl']
                self.instant_ngp.tracking_gaussian_pyramid_level = gpl
            else:
                gpl = -1

            iterations = e['iterations']
            lr_factor = e["lr_factor"]
            if "lr_factor_rot" in e:
                lr_factor_rot = e["lr_factor_rot"]
            else:
                lr_factor_rot = lr_factor
            n_steps_between_cam_updates_tracking = e['n_steps_between_cam_updates']

            self.instant_ngp.nerf.training.n_steps_since_cam_update = 0
            self.instant_ngp.nerf.training.n_steps_between_cam_updates = n_steps_between_cam_updates_tracking

            if self.separate_pos_and_rot_lr:
                self.instant_ngp.nerf.training.extrinsic_learning_rate_pos=self.pos_lr*lr_factor
                self.instant_ngp.nerf.training.extrinsic_learning_rate_rot=self.rot_lr*lr_factor_rot
            else:
                self.instant_ngp.nerf.training.extrinsic_learning_rate_pos=self.lr*lr_factor
                self.instant_ngp.nerf.training.extrinsic_learning_rate_rot=self.lr*lr_factor_rot

            if (gpl==0) or (mgl==1.):
                self.instant_ngp.tracking_mode = 0
            else:
                self.instant_ngp.tracking_mode = 1

            min_loss = 1000000.
            cur_loss = 0.
            init_loss = None
            cur_c2w = self.instant_ngp.nerf.training.get_camera_extrinsics(cam_id)
            for i in range(iterations):

                #DEBUG
                #self.instant_ngp.debug_mode=True
                #DEBUG

                self.instant_ngp.track(batch_size)
                loss = self.instant_ngp.loss_tracking
                measured_bs=self.instant_ngp.nerf.training.counters_rgb_tracking.measured_batch_size
                measured_bs_bc=self.instant_ngp.nerf.training.counters_rgb_tracking.measured_batch_size_before_compaction
                #loss = loss * batch_size / measured_bs
                cur_loss += loss

                ray_counter=np.array(self.instant_ngp.ray_counter)
                rays_per_batch=np.array(self.instant_ngp.rays_per_batch)

                #DEBUG
                # -- numsteps_compacted=np.array(self.instant_ngp.numsteps_compacted)
                # -- coords_compacted=np.array(self.instant_ngp.coords_compacted_filled)
                # -- coords_gradient=np.array(self.instant_ngp.coords_gradient)
                # -- #print(ray_counter, rays_per_batch)
                # -- #print(numsteps_compacted.shape)
                # -- #print(coords_compacted.shape)
                # -- #print(coords_gradient.shape)
                # -- N = np.sum(numsteps_compacted[::2])
                # -- #print(N, measured_bs, N*7)
                # -- q = coords_gradient[:7][:3]
                # -- w = coords_gradient[N*7:(N+1)*7][:3]
                # -- #print(q/np.linalg.norm(q), np.linalg.norm(q))
                # -- #print(w/np.linalg.norm(w), np.linalg.norm(w))
                # -- grad = coords_gradient[:N*7]
                # -- grad = grad.reshape(N,7)
                # -- print(" ")
                # -- print(" ")
                # -- print(" ")
                # -- print(grad[-10:,:])
                # -- #grad = grad[:,:3]
                # -- #grad = np.mean(grad, axis=0)
                # -- ##print("grad pose vec: ", grad/np.linalg.norm(grad), np.linalg.norm(grad))
                # -- #for x in range(N, batch_size):
                # -- #    a = x%N
                # -- #    assert np.all(coords_compacted[a*7:(a+1)*7]==coords_compacted[x*7:(x+1)*7])
                # -- pos_grad = np.array(self.instant_ngp.pos_gradient)
                # -- rot_grad = np.array(self.instant_ngp.rot_gradient)
                # -- print(pos_grad, np.linalg.norm(pos_grad), " | ", rot_grad, np.linalg.norm(rot_grad))
                #DEBUG

                # -- -- if ray_counter!=rays_per_batch:
                # -- --     print("  /!\ /!\ samples rays diff:  ", ray_counter, " != ",rays_per_batch)

                # -- -- if measured_bs_bc > batch_size*16:
                # -- --     print("  /!\ /!\ BS not big enough in Tracking (BC):  ",measured_bs_bc, " > ", batch_size)

                # -- -- if measured_bs > batch_size:
                # -- --     print("  /!\ /!\ BS not big enough in Tracking:  ", measured_bs, " > ", batch_size)
                if writer is not None:
                    iter = global_iter*self.tracking_iterations + tracking_iter
                    writer.add_scalar("tracking-loss", loss, iter)

                if self.debug_tracking:
                    # final pose, final loss, final chosen loss
                    debug_logs['loss'].append(float(loss))
                    debug_logs['ite'].append(int(tracking_iter))
                    debug_logs['measured_bs'].append(float(measured_bs))
                    debug_logs['measured_bs_bc'].append(float(measured_bs_bc))
                    debug_logs['batch_size'].append(int(batch_size))
                    debug_logs['cur_loss'].append(float(cur_loss))
                    tmp_cur_c2w = self.instant_ngp.nerf.training.get_camera_extrinsics(cam_id)
                    tmp_cur_c2w = tmp_cur_c2w.tolist()
                    debug_logs['cur_c2w'].append(tmp_cur_c2w)
                    debug_logs['mgl'].append(float(mgl))
                    debug_logs['gpl'].append(float(gpl))
                    debug_logs['pos_lr'].append(float(self.instant_ngp.nerf.training.extrinsic_learning_rate_pos))
                    debug_logs['rot_lr'].append(float(self.instant_ngp.nerf.training.extrinsic_learning_rate_rot))
                    debug_logs['n_step_bf_update'].append(int(self.instant_ngp.nerf.training.n_steps_between_cam_updates))
                    debug_logs['cur_step_bf_update'].append(int(self.instant_ngp.nerf.training.n_steps_since_cam_update))
                    debug_logs['ray_counter'].append(int(ray_counter))
                    debug_logs['rays_per_batch'].append(int(rays_per_batch))

                if measured_bs_bc > batch_size*16:
                    batch_size=int(128*np.ceil(measured_bs_bc / 128.))
                elif measured_bs > batch_size:
                    batch_size=int(128*np.ceil(measured_bs / 128.))
                else:
                    batch_size=self.target_batch_size

                if (i+1)%n_steps_between_cam_updates_tracking==0:
                    cur_loss /= n_steps_between_cam_updates_tracking
                    if init_loss is None: init_loss = cur_loss
                    if cur_loss < min_loss:
                        min_loss = cur_loss
                        final_c2w = cur_c2w.copy()
                        final_ite = tracking_iter
                    cur_loss = 0.
                    cur_c2w = self.instant_ngp.nerf.training.get_camera_extrinsics(cam_id)

                tracking_iter+=1


        if self.use_if_loss:
            self.instant_ngp.nerf.training.set_camera_extrinsics(
                cam_id,
                final_c2w.copy(), #c2w 3x4
                True, #cvt_to_ngp
            )
        else: # grab the last pose after optimization
            final_c2w = self.instant_ngp.nerf.training.get_camera_extrinsics(cam_id)

        if self.debug_tracking:
            # -- ssave debug info
            debug_logs["final_pose"] = final_c2w.tolist()
            debug_logs["final_loss"] = float(loss)
            debug_logs["final_chosen_ite"] = int(final_ite)
            debug_logs["final_chosen_loss"] = float(min_loss)

            output_filename=os.path.join(self.debug_output_dir,
                                         f"{self.cpt_images}.json")
            json.dump(debug_logs, open(output_filename, 'w'))
            self.cpt_images += 1


        return final_c2w, loss




