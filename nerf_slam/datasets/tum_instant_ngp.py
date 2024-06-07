import os
import cv2
import json
import errno
import torch
import numpy as np
import torch.nn.functional as F

from .build import DATASET_REGISTRY

from .base import BaseDataset
from .base import readEXR_onlydepth

from .associate_tum import associate, read_file_list

__all__ = ["TUMDatasetInstantNGP"]

@DATASET_REGISTRY.register()
class TUMDatasetInstantNGP(BaseDataset):
    
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
        
        # not used for TUM
        self.frame_sampling_rate = cfg.DATASET.FRAME_SAMPLING_RATE
        self.frame_sampling_offset = cfg.DATASET.FRAME_SAMPLING_OFFSET


    def __init__(self, cfg):
        super(TUMDatasetInstantNGP, self).__init__(cfg)

        self._parse_cfg(cfg)
        
        self.load_data()

        #load image sharpness if any
        if os.path.isfile(self.sharpness_filename):
            print(" LOADING Sharpness from: ", self.sharpness_filename)
            self.sharpness = json.load(open(self.sharpness_filename, 'r'))
        else:
            self.sharpness=[]
        
    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        data = np.loadtxt(filepath, 
                          delimiter=' ',
                          dtype=np.unicode_, 
                          skiprows=skiprows)
        return data

    
    def load_data(self):
        
        datapath = self.input_folder

        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
            pose_list = os.path.join(datapath, 'pose.txt')
        else:
           raise FileNotFoundError(
               errno.ENOENT, 
               os.strerror(errno.ENOENT), 
               os.path.join(datapath, 'pose.txt')
           )

        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')

        image_data = read_file_list(image_list)
        depth_data = read_file_list(depth_list)
        
        # -- match RGB and Depth
        matches = associate(
            image_data, 
            depth_data,
            0.0,
            0.02,
        )    
        
        images, depths, timestamps = [], [], []

        for m in matches:
            timestamps.append(
                0.5*(m[0]+m[1])
            )
            images.append(
                os.path.join(
                    datapath,
                    image_data[m[0]][0]
                )
            )
            depths.append(
                os.path.join(
                    datapath,
                    depth_data[m[1]][0],
                )
            )

        self.images = images
        self.depths = depths
        self.timestamps = timestamps

        # -- match with GT poses
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)
        pose_tstamps = pose_data[:, 0].astype(np.float64)
        
        pose_list = {t:i for i,t in enumerate(pose_tstamps)}
        rgbd_list = {t:i for i,t in enumerate(timestamps)}

        matches_poses = associate(
            rgbd_list, 
            pose_list,
            0.0,
            0.02,
        )
        
        # assert all rgbd input has an associate GT pose
        set_rgbd_tstamps = set(rgbd_list.keys())
        set_matches_tstamps = set([m[0] for m in matches_poses])
        
        #assert set_rgbd_tstamps.issubset(set_matches_tstamps)
        if not set_rgbd_tstamps.issubset(set_matches_tstamps):
            print(
                " /!\ some frames don't have associated GT poses (cf. TUM associate.py) /!\ ",
                " \n Trying to increase the max timestamp diff. Otherwise dropping the frame."
            )

            matches_poses = associate(
                rgbd_list, 
                pose_list,
                0.0,
                0.1,
            )
            set_matches_tstamps = set([m[0] for m in matches_poses])
            
            if not set_rgbd_tstamps.issubset(set_matches_tstamps):
                print(" --> Need to drop frames.")
                raise NotImplementedError
        
        poses = []
        poses_timestamps = []
        inv_pose = None
        for i, m in enumerate(matches_poses):

            assert timestamps[i] == m[0]
            
            index = pose_list[m[1]]

            poses_timestamps.append(m[1])

            pose = pose_vecs[index]
            
            c2w = self.pose_matrix_from_quaternion(pose)

            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose@c2w
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1

            poses.append(c2w)
        
        # replace poses with the preprocessed ones if exists 
        if self.poses_filename and os.path.isfile(self.poses_filename):
            processed_poses = json.load(open(self.poses_filename, 'r'))
            poses = []
            for c2w in processed_poses:
                c2w = np.array(c2w)
                c2w = c2w.astype(np.float32)
                poses.append(c2w)
            assert len(poses) == len(poses_timestamps)
        
        self.poses = poses
        self.poses_timestamps = poses_timestamps
            

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose
    
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
        color_path = self.images[index]
        color_data = cv2.imread(color_path)
        
        # -- load depth
        depth_path = self.depths[index]
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

        timestamp = self.timestamps[index]

        pose_timestamp = self.poses_timestamps[index]

        return {
            'index': index,
            'rgb': color_data,
            'depth': depth_data,
            'c2w': pose,
            'timestamp': timestamp,
            'c2w_timestamp': pose_timestamp,
            'rgb_filename': color_path,
            'depth_filename': depth_path,
            'sharpness': sharpness
        }

    
    def __len__(self):
        return len(self.images)

