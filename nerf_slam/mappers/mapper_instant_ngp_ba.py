import os
import cv2
import json
import numpy as np

from .build import MAPPER_REGISTRY


__all__ = ["MapperInstantNGP_C2FBA"]


@MAPPER_REGISTRY.register()
class MapperInstantNGP_C2FBA(object):
    """
    Instant-NGP Mapper thread.

    """
    def _parse_cfg(self, cfg):
        self.mapping_iterations = cfg.MAPPER.MAPPING_ITERATIONS
        self.max_frames_per_iter = cfg.MAPPER.MAX_FRAMES_PER_ITER
        self.keyframe_selection_method = cfg.MAPPER.KEYFRAME_SELECTION_METHOD

        self.margin_sample_away_from_edge_W = cfg.MAPPER.MARGIN_SAMPLE_WIDTH
        self.margin_sample_away_from_edge_H = cfg.MAPPER.MARGIN_SAMPLE_HEIGHT

        self.max_grid_level_rand_training = cfg.MAPPER.MAX_GRID_LEVEL_RAND_TRAINING
        self.max_grid_level_factor = cfg.MAPPER.MAX_GRID_LEVEL_FACTOR
        self.extrinsic_learning_rate = cfg.MAPPER.EXTRINSIC_LEARNING_RATE
        self.extrinsic_learning_rate_ba_pos=cfg.MAPPER.EXTRINSIC_LEARNING_RATE_POS
        self.extrinsic_learning_rate_ba_rot=cfg.MAPPER.EXTRINSIC_LEARNING_RATE_ROT

        self.sample_image_proportional_to_error = cfg.MAPPER.SAMPLE_IMAGE_PROPORTIONAL_TO_ERROR

        self.num_rays_to_sample = cfg.MAPPER.NUM_RAYS_TO_SAMPLE

        self.aabb_scale = cfg.RENDERER.AABB_SCALE

        self.no_num_rays_imposed_on_first_frame=cfg.MAPPER.NO_NUM_RAYS_IMPOSED_ON_FIRST_FRAME

        self.target_batch_size = cfg.MAPPER.TARGET_BATCH_SIZE
        self.n_steps_between_cam_updates=cfg.MAPPER.N_STEPS_BETWEEN_CAM_UPDATES

        self.ba_mode=cfg.MAPPER.BA_MODE

        self.tracking_hyperparameters_filename=cfg.MAPPER.TRACKING_HYPERPARAMETERS_FILE
        if os.path.isfile(self.tracking_hyperparameters_filename):
            self.ba_hyperparams = json.load(open(self.tracking_hyperparameters_filename, 'r'))
            self.ba_iterations = np.sum([x['iterations'] for x in self.ba_hyperparams])
        else:
            self.ba_hyperparams = None
            print(" /!\ Couldn't find mappier hyper params'")
            raise ValueError

        self.lbd_depth = cfg.MODEL.DEPTH_SUPERVISION_LAMBDA
        self.lbd_rgb = cfg.MODEL.RGB_SUPERVISION_LAMBDA

        self.debug_mapping = cfg.MAPPER.DEBUG
        if self.debug_mapping:
            self.cpt_images = 0
            self.debug_output_dir = os.path.join(
                cfg.META.OUTPUT_DIR,
                cfg.META.NAME_EXPERIMENT,
                str(cfg.META.RUN_ID),
                cfg.RUNNER.LOGS_DIRNAME,
                "mapping_debug"
            )
            from pathlib import Path
            Path(self.debug_output_dir).mkdir(parents=True, exist_ok=True)


    def __init__(self, cfg):
        self._parse_cfg(cfg)

    def add_renderer(self, renderer):
        self.renderer = renderer

    def add_instant_ngp(self, instant_ngp):
        self.instant_ngp = instant_ngp

    def random_select(self, l, k):
        """
        Random select k values from 0..l.

        """
        return list(np.random.permutation(np.array(range(l)))[:min(l, k)])


    def init_instant_ngp_dataset(self,
                                 keyframes,
                                 optimize_frame_indices,
                                 cur_cam_id,
                                 BA=False,
                                 train_grid=True,
                                 train_decoder=True,
                                 keyframe_selection_method=None,
                                 idx_images_for_mapping=None,
                                 idx_images_for_tracking=None
                                ):

        if idx_images_for_mapping is not None:
            idx_images_for_training_slam = idx_images_for_mapping
        else:
            idx_images_for_training_slam = []
            for e in optimize_frame_indices:
                if e=='current':
                    idx_images_for_training_slam.append(cur_cam_id)
                elif e=='last_keyframe':
                    idx_images_for_training_slam.append(keyframes[-1]['index'])
                else:
                    idx_images_for_training_slam.append(keyframes[e]['index'])

        self.instant_ngp.nerf.training.idx_images_for_mapping=idx_images_for_training_slam
        self.instant_ngp.nerf.training.sample_image_proportional_to_error=self.sample_image_proportional_to_error
        self.instant_ngp.nerf.training.optimize_extrinsics = False
        self.instant_ngp.nerf.training.optimize_exposure = False
        self.instant_ngp.nerf.training.optimize_extra_dims = False
        self.instant_ngp.nerf.training.optimize_distortion = False
        self.instant_ngp.nerf.training.optimize_focal_length = False
        self.instant_ngp.nerf.training.include_sharpness_in_error=False
        self.instant_ngp.nerf.training.idx_images_for_training_extrinsics=[]
        self.instant_ngp.nerf.training.n_steps_between_cam_updates = self.n_steps_between_cam_updates
        self.instant_ngp.nerf.training.n_steps_since_cam_update = 0
        self.instant_ngp.nerf.training.n_steps_since_error_map_update = 0
        self.instant_ngp.nerf.training.sample_away_from_border_margin_h_mapping = self.margin_sample_away_from_edge_H
        self.instant_ngp.nerf.training.sample_away_from_border_margin_w_mapping = self.margin_sample_away_from_edge_W
        self.instant_ngp.shall_train_encoding = train_grid
        self.instant_ngp.shall_train_network = train_decoder
        self.instant_ngp.shall_train = True
        self.instant_ngp.max_level_rand_training = self.max_grid_level_rand_training
        self.instant_ngp.max_grid_level_factor = self.max_grid_level_factor
        self.instant_ngp.ba_mode = self.ba_mode
        self.instant_ngp.nerf.training.use_depth_var_in_tracking_loss = False
        self.instant_ngp.reset_prep_nerf_mapping = True
        self.instant_ngp.nerf.training.error_map.is_cdf_valid = False
        if self.num_rays_to_sample > 0:
            self.instant_ngp.nerf.training.target_num_rays_for_mapping = self.num_rays_to_sample
            self.instant_ngp.nerf.training.target_num_rays_for_ba = self.num_rays_to_sample
            self.instant_ngp.nerf.training.set_fix_num_rays_to_sample = True
        else:
            self.instant_ngp.nerf.training.set_fix_num_rays_to_sample = False

        if BA:
            if idx_images_for_tracking is not None:
                idx_images_for_training_slam_pose = idx_images_for_tracking
            else:
                if keyframe_selection_method=="last":
                    idx_images_for_training_slam_pose = [cur_cam_id]
                else:
                    idx_images_for_training_slam_pose = [x for x in idx_images_for_training_slam if x!=0]
            self.instant_ngp.nerf.training.optimize_extrinsics=True
            self.instant_ngp.nerf.training.extrinsic_learning_rate = self.extrinsic_learning_rate
            self.instant_ngp.nerf.training.extrinsic_learning_rate_ba_pos=self.extrinsic_learning_rate_ba_pos
            self.instant_ngp.nerf.training.extrinsic_learning_rate_ba_rot=self.extrinsic_learning_rate_ba_rot
            self.instant_ngp.nerf.training.idx_images_for_training_extrinsics=idx_images_for_training_slam_pose



    def optimize_map(self,
                     keyframes,
                     rgb,
                     depth,
                     cur_c2w,
                     cam_id,
                     global_iter=0,
                     writer=None,
                     mapping_iterations=None,
                     target_batch_size=None,
                     BA=False,
                     train_grid=True,
                     train_decoder=True,
                     keyframe_selection_method=None,
                     idx_images_for_mapping=None,
                     idx_images_for_tracking=None
                    ):
        """
        Mapping iterations. Sample pixels from selected keyframes,
        then optimize scene representation and camera poses(if local BA enables).

        Args:
            num_joint_iters (int): number of mapping iterations.
            lr_factor (float): the factor to times on current lr.
            idx (int): the index of current frame
            cur_gt_color (tensor): gt_color image of the current camera.
            cur_gt_depth (tensor): gt_depth image of the current camera.
            keyframe_dict (list): list of keyframes info dictionary.
            keyframe_list (list): list ofkeyframe index.
            cur_c2w (tensor): the estimated camera to world matrix of current frame.

        Returns:
            cur_c2w/None (tensor/None): return the updated cur_c2w, return None if no BA
        """

        if keyframe_selection_method is None:
            keyframe_selection_method=self.keyframe_selection_method

        # -- keyframe selection
        optimize_frame_indices = None
        if ((idx_images_for_mapping is None) or (idx_images_for_tracking is None)):
            if len(keyframes) == 0:
                optimize_frame_indices = []
            else:
                if keyframe_selection_method == 'global':
                    num = self.max_frames_per_iter-2
                    optimize_frame_indices = self.random_select(len(keyframes)-1, num)
                elif keyframe_selection_method == 'overlap':
                    num = self.max_frames_per_iter-2
                    optimize_frame_indices = self.keyframe_selection_overlap(
                        depth,
                        cur_c2w.copy(),
                        keyframes[:-1],
                        num,
                        samples=10000
                    )
                elif keyframe_selection_method == 'all':
                    optimize_frame_indices = list(range(len(keyframes)-1))
                elif keyframe_selection_method == 'all_keyframes_only':
                    optimize_frame_indices = list(range(len(keyframes)))
                elif keyframe_selection_method == 'mix':
                    num = (self.max_frames_per_iter - 2) // 2
                    optimize_frame_indices = self.keyframe_selection_overlap(
                        depth,
                        cur_c2w.copy(),
                        keyframes[:-1],
                        num,
                        samples=10000
                    )
                    num_b = self.max_frames_per_iter-2 - num
                    aa = [x for x in range(len(keyframes)-1) if x not in optimize_frame_indices]
                    if (num_b > 0) and (len(aa)>0):
                        if (num_b >=len(aa)):
                            optimize_frame_indices_b = aa
                        else:
                            optimize_frame_indices_b = np.random.choice(aa,num_b,replace=False)
                            optimize_frame_indices_b = optimize_frame_indices_b.tolist()
                    else:
                        optimize_frame_indices_b = []
                    optimize_frame_indices += optimize_frame_indices_b
                elif keyframe_selection_method== 'last':
                    num = self.max_frames_per_iter-1
                    optimize_frame_indices = list(range(len(keyframes)))
                    optimize_frame_indices = optimize_frame_indices[-num:-1]
                else:
                    raise ValueError

            if not (keyframe_selection_method == 'all_keyframes_only'):
                # add last KF
                if len(keyframes) > 0:
                    optimize_frame_indices.append('last_keyframe')
                # add current frame
                optimize_frame_indices.append('current')
            else:
                if len(keyframes) == 0:
                    optimize_frame_indices.append('current')

        # -- setup NGP
        self.init_instant_ngp_dataset(
            keyframes,
            optimize_frame_indices,
            cam_id,
            BA,
            train_grid,
            train_decoder,
            keyframe_selection_method,
            idx_images_for_mapping,
            idx_images_for_tracking
        )

        if target_batch_size is None:
            target_batch_size=self.target_batch_size #target BS -> will sample as many rays as possible
        else:
            target_batch_size=target_batch_size
        batch_size=target_batch_size

        if mapping_iterations is not None:
            ba_params = [
                {"iterations":mapping_iterations, 'mgl': 1.0, 'gpl': 0}
            ]
            if self.no_num_rays_imposed_on_first_frame:
                self.instant_ngp.nerf.training.set_fix_num_rays_to_sample = False
        else:
            if self.ba_hyperparams is None:
                mapping_iterations = self.mapping_iterations
                ba_params = [
                    {"iterations":mapping_iterations, 'mgl': 1.0, 'gpl': 0}
                ]
            else:
                ba_params = self.ba_hyperparams

        if self.debug_mapping:
            # init debug info
            debug_logs = {
                'loss': [],
                'ite': [],
                'batch_size': [],
                'measured_bs': [],
                'measured_bs_bc': [],
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


        # -- start mapping
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

            if "lr_factor" in bap:
                lr_factor = bap["lr_factor"]
            else:
                lr_factor = 1.0

            if "lr_factor_rot" in bap:
                lr_factor_rot = bap["lr_factor_rot"]
            else:
                lr_factor_rot = lr_factor

            self.instant_ngp.nerf.training.extrinsic_learning_rate_ba_pos=self.extrinsic_learning_rate_ba_pos*lr_factor
            self.instant_ngp.nerf.training.extrinsic_learning_rate_ba_rot=self.extrinsic_learning_rate_ba_rot*lr_factor_rot

            if "lbd_rgb_factor" in bap:
                lbd_rgb_factor = bap["lbd_rgb_factor"]
            else:
                lbd_rgb_factor = 1.0

            if "lbd_depth_factor" in bap:
                lbd_depth_factor = bap["lbd_depth_factor"]
            else:
                lbd_depth_factor = 1.0

            self.instant_ngp.nerf.training.rgb_supervision_lambda=self.lbd_rgb*lbd_rgb_factor
            self.instant_ngp.nerf.training.depth_supervision_lambda=self.lbd_depth*lbd_depth_factor

            self.instant_ngp.nerf.training.n_steps_since_cam_update = 0

            #DEBUG
            #self.instant_ngp.debug_mode=True
            #DEBUG

            for _ in range(num_iterations):
                if (gpl==0) or (mgl==1.):
                    self.instant_ngp.map(batch_size)
                    measured_bs=self.instant_ngp.nerf.training.counters_rgb.measured_batch_size
                    measured_bs_bc=self.instant_ngp.nerf.training.counters_rgb.measured_batch_size_before_compaction
                else:
                    self.instant_ngp.ba(batch_size)
                    measured_bs=self.instant_ngp.nerf.training.counters_rgb_ba.measured_batch_size
                    measured_bs_bc=self.instant_ngp.nerf.training.counters_rgb_ba.measured_batch_size_before_compaction

                loss = self.instant_ngp.loss
                #loss = loss * batch_size / measured_bs

                #DEBUG
                ray_counter=np.array(self.instant_ngp.ray_counter)
                rays_per_batch=np.array(self.instant_ngp.rays_per_batch)
                #sample_z_vals = np.array(self.instant_ngp.sample_z_vals)
                #numsteps=np.array(self.instant_ngp.numsteps)
                #measured_bs_bc=self.instant_ngp.nerf.training.counters_rgb.measured_batch_size_before_compaction
                #measured_bs=self.instant_ngp.nerf.training.counters_rgb.measured_batch_size
                #numsteps_compacted=np.array(self.instant_ngp.numsteps_compacted)
                #a = np.sum(numsteps_compacted[0::2])
                #b = np.sum(numsteps[0::2])
                #if a!=measured_bs or (measured_bs > batch_size):
                #if ray_counter!=rays_per_batch or (measured_bs > batch_size):
                #    print("ray counter / rays_per_batch: ", ray_counter,rays_per_batch)
                #    print("targetBS (x16): ", batch_size, batch_size*16)
                #    #print("N inference samples / measured_BS: ",b, numsteps[-1]+numsteps[-2], measured_bs_bc)
                #    #print("N CP samples / measured_BS_CP: ", a, measured_bs)
                #    print("measured bs / BC", measured_bs, measured_bs_bc)
                #DEBUG

                # -- if measured_bs_bc > batch_size*16:
                # --     print("  /!\ /!\ BS not big enough in Mapping (BC):  ",measured_bs_bc, " > ", batch_size)

                # -- if measured_bs > batch_size:
                # --     print("  /!\ /!\ BS not big enough in Mapping:  ",
                # --           measured_bs, " > ", batch_size)



                if self.debug_mapping:
                    # final pose, final loss, final chosen loss
                    debug_logs['loss'].append(float(loss))
                    debug_logs['ite'].append(int(ba_iter))
                    debug_logs['measured_bs'].append(float(measured_bs))
                    debug_logs['measured_bs_bc'].append(float(measured_bs_bc))
                    debug_logs['batch_size'].append(int(batch_size))
                    tmp_cur_c2w = self.instant_ngp.nerf.training.get_camera_extrinsics(cam_id)
                    tmp_cur_c2w = tmp_cur_c2w.tolist()
                    debug_logs['cur_c2w'].append(tmp_cur_c2w)
                    debug_logs['mgl'].append(float(mgl))
                    debug_logs['gpl'].append(float(gpl))
                    debug_logs['pos_lr'].append(float(self.instant_ngp.nerf.training.extrinsic_learning_rate_ba_pos))
                    debug_logs['rot_lr'].append(float(self.instant_ngp.nerf.training.extrinsic_learning_rate_ba_rot))
                    debug_logs['n_step_bf_update'].append(int(self.instant_ngp.nerf.training.n_steps_between_cam_updates))
                    debug_logs['cur_step_bf_update'].append(int(self.instant_ngp.nerf.training.n_steps_since_cam_update))
                    debug_logs['ray_counter'].append(int(ray_counter))
                    debug_logs['rays_per_batch'].append(int(rays_per_batch))



                if measured_bs_bc > batch_size*16:
                    batch_size=int(128*np.ceil(measured_bs_bc / 128.))
                elif measured_bs > batch_size:
                    batch_size=int(128*np.ceil(measured_bs / 128.))
                else:
                    batch_size=target_batch_size

                if writer is not None:
                    iter = global_iter*all_iterations + ba_iter
                    writer.add_scalar("mapping-loss", loss, iter)

                ba_iter+=1

        if self.debug_mapping:
            # -- ssave debug info
            output_filename=os.path.join(self.debug_output_dir,f"{self.cpt_images}.json")
            json.dump(debug_logs, open(output_filename, 'w'))
            self.cpt_images += 1

        return loss




    def keyframe_selection_overlap(self,
                                   cur_depth,
                                   cur_c2w,
                                   keyframes,
                                   k,
                                   samples=10000):
        """
        FIXME /!\

        Select overlapping keyframes to the current camera observation.

        """

        if k >= len(keyframes):
            return list(range(len(keyframes)))

        if cur_c2w.shape[0] < 4:
            cur_c2w = np.pad(
                cur_c2w,
                pad_width=[(0,1), (0,0)]
            )
            cur_c2w[3,3] = 1.0

        cur_c2w[:3,1] *= -1
        cur_c2w[:3,2] *= -1

        H = self.renderer.H
        W = self.renderer.W
        fx = self.renderer.fx
        fy = self.renderer.fy
        cx = self.renderer.cx
        cy = self.renderer.cy

        K = np.array(
            [
                [fx, .0, cx],
                [.0, fy, cy],
                [.0, .0, 1.0]
            ]
        )

        K_inv = np.linalg.inv(K)

        # -- unproject depth
        H, W = cur_depth.shape
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
        pixel_depth = cur_depth.flatten()

        mask_depth = pixel_depth > 0.0

        pixel_coords = pixel_coords[:,mask_depth]
        pixel_depth = pixel_depth[mask_depth]

        N = len(pixel_depth)

        if samples != "all":
            indices = np.random.randint(0, N, samples)

            pixel_coords = pixel_coords[:,indices]
            pixel_depth = pixel_depth[indices]

            N = samples


        cam_coords = K_inv @ pixel_coords * pixel_depth

        cam_coords_homo = np.concatenate(
            [cam_coords, np.ones(N).reshape(1,-1)],
            axis=0
        )

        world_coords = cur_c2w @ cam_coords_homo

        list_keyframe = []
        for keyframeid, keyframe in enumerate(keyframes):
            c2w = keyframe['c2w'].copy()
            tmp_H, tmp_W = keyframe['depth'].shape

            if c2w.shape[0] < 4:
                c2w = np.pad(
                    c2w,
                    pad_width=[(0,1), (0,0)]
                )
                c2w[3,3] = 1.0

            c2w[:3,1] *= -1
            c2w[:3,2] *= -1

            w2c = np.linalg.inv(c2w)

            new_cam_coords = w2c @ world_coords

            new_cam_coords = new_cam_coords[:3,:]

            new_pixel_coords  = K @ new_cam_coords

            s = new_pixel_coords[2,:] + 1e-8

            new_uv = new_pixel_coords[:2,:] / s

            new_uv = new_uv.astype(np.float32)

            mask = (new_uv[0,:] < tmp_W) *\
                   (new_uv[0,:] >= 0) *\
                   (new_uv[1,:] < tmp_H) *\
                   (new_uv[1,:] >= 0)

            mask = mask & (s > 0)

            percent_inside = mask.sum() / new_uv.shape[1]

            list_keyframe.append(
                {
                    'id': keyframeid,
                    'percent_inside': percent_inside
                }
            )

        list_keyframe = sorted(
            list_keyframe,
            key=lambda i: i['percent_inside'],
            reverse=True
        )

        selected_keyframe_list = [dic['id'] for dic in list_keyframe]
        selected_keyframe_list = selected_keyframe_list[:k]
        return selected_keyframe_list







