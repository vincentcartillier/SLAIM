import os
import cv2
import json
import glob
import torch
import numpy as np
import torch.nn.functional as F

from .build import DATASET_REGISTRY

from .base import BaseDataset
from .base import readEXR_onlydepth

__all__ = ["ReplicaDatasetInstantNGP"]

@DATASET_REGISTRY.register()
class ReplicaDatasetInstantNGP(BaseDataset):

    def _parse_cfg(self, cfg):
        self.near = cfg.RENDERER.NEAR
        self.far = cfg.RENDERER.FAR
        self.poses_filename = cfg.DATASET.POSES_FILENAME
        if os.path.isfile(self.poses_filename):
            self.poses_scale = cfg.DATASET.POSES_SCALE
        else:
            print("WARNING: No preprocessed poses found!",
                 "If you are running iNGP, you might want to run preprocess_camera_poses.py first")
            self.poses_scale = None

    def __init__(self, cfg):
        super(ReplicaDatasetInstantNGP, self).__init__(cfg)

        self._parse_cfg(cfg)

        self.color_paths = sorted(
            glob.glob(
                os.path.join(
                    self.input_folder,
                    'results/frame*.jpg'
                )
            )
        )
        self.depth_paths = sorted(
            glob.glob(
                os.path.join(
                    self.input_folder,
                    'results/depth*.png'
                )
            )
        )
        self.n_img = len(self.color_paths)
        self.load_poses()

    def load_poses(self, ):

        if self.poses_filename and os.path.isfile(self.poses_filename):
            processed_poses = json.load(open(self.poses_filename, 'r'))
            poses = []
            for c2w in processed_poses:
                c2w = np.array(c2w)
                c2w = c2w.astype(np.float32)
                poses.append(c2w)
            self.poses = poses

        else:
            filename = os.path.join(
                self.input_folder,
                'traj.txt'
            )
            self.poses = []
            with open(filename, "r") as f:
                lines = f.readlines()
            for i in range(self.n_img):
                line = lines[i]
                c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
                c2w[:3, 1] *= -1
                c2w[:3, 2] *= -1
                #c2w = torch.from_numpy(c2w).float()
                self.poses.append(c2w.astype(np.float))



    def get_all_poses(self,):
        return self.poses

    def get_all(self,):
        data = []
        for i in range(len(self.images)):
            d = self.__getitem__(i)
            data.append(
                {
                    'index': d['index'],
                    'rgb': d['rgb'],
                    'depth': d['depth'],
                    'gt_c2w': d['c2w'],
                }
            )
        return data

    def __getitem__(self, index):
        # -- load RGB image
        color_path = self.color_paths[index]
        color_data = cv2.imread(color_path)

        # -- load depth
        depth_path = self.depth_paths[index]
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        elif '.exr' in depth_path:
            depth_data = readEXR_onlydepth(depth_path)

        # -- undistord image id needed
        if self.distortion is not None:
            K = np.eye(3)
            K[0, 0] = self.fx
            K[1, 1] = self.fy
            K[0, 2] = self.cx
            K[1, 2] = self.cy
            # undistortion is only applied on color image, not depth!
            color_data = cv2.undistort(color_data, K, self.distortion)

        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = color_data / 255.

        depth_data = depth_data.astype(np.float32) / self.png_depth_scale

        depth_data[depth_data < self.near] = 0
        depth_data[depth_data > self.far] = 0

        H, W = depth_data.shape
        color_data = cv2.resize(color_data, (W, H))

        if self.crop_size is not None:
            # -- convert to torch for cropping
            color_data = torch.from_numpy(color_data)
            depth_data = torch.from_numpy(depth_data)

            # follow the pre-processing step in lietorch, actually is resize
            color_data = color_data.permute(2, 0, 1)
            color_data = F.interpolate(
                color_data[None],
                self.crop_size,
                mode='bilinear',
                align_corners=True)[0]

            depth_data = F.interpolate(
                depth_data[None, None],
                self.crop_size,
                mode='nearest')[0, 0]

            color_data = color_data.permute(1, 2, 0).contiguous()

            color_data = color_data.numpy()
            depth_data = depth_data.numpy()

        if self.crop_edge is not None:
            # crop image edge, there are invalid value on the edge of the color image
            edge = self.crop_edge
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]

        # -- get pose
        pose = self.poses[index]

        color_data = color_data.astype(np.float32)
        depth_data = depth_data.astype(np.float32)
        depth_data[depth_data<0.0] = 0.0
        pose = pose.astype(np.float32)

        # -- convert data back to Instant NGP format
        color_data *= 255.0
        color_data = color_data.astype(np.uint8)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_RGB2RGBA)

        if self.poses_scale is not None:
            depth_data *= self.poses_scale

        return {
            'index': index,
            'rgb': color_data,
            'depth': depth_data,
            'c2w': pose,
            'timestamp': -1,
            'c2w_timestamp': -1,
            'rgb_filename': color_path,
            'depth_filename': depth_path
        }


    def __len__(self):
        return self.n_img

