import os
import torch
import numpy as np

from .ate_utils import evaluate_ate

from scipy.spatial.transform import Rotation

from .build import EVALUATOR_REGISTRY

__all__ = ["EvaluatorATEInstantNGP"]

@EVALUATOR_REGISTRY.register()
class EvaluatorATEInstantNGP():
    def _parse_cfg(self, cfg):
        self.scale = cfg.RENDERER.SCALE

    def __init__(self, cfg):
        self._parse_cfg(cfg)


    def get_tensor_from_camera(self, T):
        """
        Convert transformation matrix to quaternion and translation.
        /!\ here we are returning [t, quat]
        This is different from the tracking method ([quat,t])
        """
        T = T.detach().cpu().numpy()
        R, t = T[:3, :3], T[:3, 3]
        rot = Rotation.from_matrix(R)
        quat = rot.as_quat()
        tensor = np.concatenate([t, quat], 0)
        tensor = torch.from_numpy(tensor).float()
        return tensor


    def convert_poses(self, c2w_list):
        poses = []
        mask = []
        for i in range(len(c2w_list)):
            c2w = c2w_list[i]
            c2w = np.asarray(c2w)
            c2w = torch.from_numpy(c2w)

            poses.append(self.get_tensor_from_camera(c2w))
            mask.append(1)
        poses = torch.stack(poses)
        mask = torch.Tensor(mask).bool()
        return poses, mask




    def eval(self,
             pred,
             gt,
             viz_filename=None):
        """
        TODO: Need to define pred and gt formats
        /!\ need to be able to evaluate with less preds
        /!\ Ignore first pose as pred==gt
        """
        # -- preprocess poses
        poses_gt, mask = self.convert_poses(gt)
        poses_pred, _ = self.convert_poses(pred)

        # -- eval
        poses_gt = poses_gt.cpu().numpy()
        poses_pred = poses_pred.cpu().numpy()

        N = poses_pred.shape[0]
        poses_gt = dict([(i, poses_gt[i]) for i in range(N)])
        poses_pred = dict([(i, poses_pred[i]) for i in range(N)])

        if viz_filename is None:
            plot=""
        else:
            plot=viz_filename

        results = evaluate_ate(poses_gt, poses_pred, plot)

        return results


