import os
import cv2
import json
import glob
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from .build import DATASET_REGISTRY

from .base import BaseDataset
from .base import readEXR_onlydepth

__all__ = ["ScannetDatasetInstantNGP"]

@DATASET_REGISTRY.register()
class ScannetDatasetInstantNGP(BaseDataset):

    def _parse_cfg(self, cfg):
        self.near = cfg.RENDERER.NEAR
        self.far = cfg.RENDERER.FAR
        self.poses_filename = cfg.DATASET.POSES_FILENAME
        self.sharpness_filename = cfg.DATASET.SHARPNESS_FILENAME
        if os.path.isfile(self.poses_filename):
            self.poses_scale = cfg.DATASET.POSES_SCALE
        else:
            print("WARNING: No preprocessed poses found!",
                 "If you are running iNGP, you might want to run preprocess_camera_poses.py first")
            self.poses_scale = None

        self.frame_sampling_rate = cfg.DATASET.FRAME_SAMPLING_RATE
        self.frame_sampling_offset = cfg.DATASET.FRAME_SAMPLING_OFFSET


    def __init__(self, cfg):
        super(ScannetDatasetInstantNGP, self).__init__(cfg)

        self._parse_cfg(cfg)

        self.input_folder = os.path.join(self.input_folder, 'frames')

        self.color_paths = sorted(
            glob.glob(
                os.path.join(
                    self.input_folder,
                    'color',
                    '*.jpg'
                )
            ),
            key=lambda x: int(os.path.basename(x)[:-4])
        )
        self.color_paths = self.color_paths[self.frame_sampling_offset:]
        if self.frame_sampling_rate > 1:
            self.color_paths = self.color_paths[::self.frame_sampling_rate]

        self.depth_paths = sorted(
            glob.glob(
                os.path.join(
                    self.input_folder,
                    'depth',
                    '*.png'
                )
            ),
            key=lambda x: int(os.path.basename(x)[:-4])
        )
        self.depth_paths = self.depth_paths[self.frame_sampling_offset:]
        if self.frame_sampling_rate > 1:
            self.depth_paths = self.depth_paths[::self.frame_sampling_rate]

        self.n_img = len(self.color_paths)
        self.load_poses()

        self.poses = self.poses[self.frame_sampling_offset:]
        if self.frame_sampling_rate > 1:
            self.poses = self.poses[::self.frame_sampling_rate]

        #load image sharpness if any
        if os.path.isfile(self.sharpness_filename):
            print(" LOADING Sharpness from: ", self.sharpness_filename)
            sharpness = json.load(open(self.sharpness_filename, 'r'))
            self.sharpness = sharpness

            self.sharpness = self.sharpness[self.frame_sampling_offset:]
            if self.frame_sampling_rate > 1:
                self.sharpness = self.sharpness[::self.frame_sampling_rate]
        else:
            self.sharpness=[]

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
            self.poses = []
            pose_paths = sorted(
                glob.glob(
                    os.path.join(
                        self.input_folder,
                        'pose',
                        '*.txt'
                    )
                ),
                key=lambda x: int(os.path.basename(x)[:-4])
            )
            for pose_path in tqdm(pose_paths):
                with open(pose_path, "r", encoding='UTF-8') as f:
                    lines = f.readlines()
                ls = []
                for line in lines:
                    l = list(map(float, line.split(' ')))
                    ls.append(l)
                c2w = np.array(ls).reshape(4, 4)
                c2w[:3, 1] *= -1
                c2w[:3, 2] *= -1
                #c2w = torch.from_numpy(c2w).float()
                self.poses.append(c2w.astype(np.float32))

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
                    'sharpness': d['sharpness']
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

        depth_data[depth_data < self.near] = -1
        depth_data[depth_data > self.far] = -1

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

        # -- load sharpess if any:
        if len(self.sharpness) > 0:
            sharpness = self.sharpness[index]
        else:
            sharpness = 10000.

        return {
            'index': index,
            'rgb': color_data,
            'depth': depth_data,
            'c2w': pose,
            'timestamp': -1,
            'c2w_timestamp': -1,
            'rgb_filename': color_path,
            'depth_filename': depth_path,
            'sharpness': sharpness
        }


    def __len__(self):
        return self.n_img

