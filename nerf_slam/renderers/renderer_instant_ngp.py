import torch
from .build import RENDERER_REGISTRY
import torch.nn.functional as F
import numpy as np

__all__ = ["RendererInstantNGP"]


@RENDERER_REGISTRY.register()
class RendererInstantNGP(object):

    def _parse_cfg(self, cfg):
        self.H = cfg.RENDERER.IMAGE_H
        self.W = cfg.RENDERER.IMAGE_W
        self.fx = cfg.RENDERER.FX
        self.fy = cfg.RENDERER.FY
        self.cx = cfg.RENDERER.CX
        self.cy = cfg.RENDERER.CY

        if cfg.DATASET.CROP_SIZE.ENABLED:
            self.crop_size = cfg.DATASET.CROP_SIZE.VALUE
        else:
            self.crop_size = None

        if cfg.DATASET.CROP_EDGE.ENABLED:
            self.crop_edge = cfg.DATASET.CROP_EDGE.VALUE
        else:
            self.crop_edge = None

        # update intrinsics if processing inputs
        self.adjust_intrinsics_if_preprocessing()
        
        self.scale = cfg.RENDERER.SCALE
        self.offset = np.array(cfg.RENDERER.OFFSET)
        self.aabb_scale = cfg.RENDERER.AABB_SCALE


    def __init__(self, cfg):
        self._parse_cfg(cfg)

    def adjust_intrinsics_if_preprocessing(self):
        """
        Update the camera intrinsics according to pre-processing config,
        such as resize or edge crop.
        """
        # resize the input images to crop_size (variable name used in lietorch)
        if self.crop_size is not None:
            crop_size = self.crop_size
            sx = crop_size[1] / self.W
            sy = crop_size[0] / self.H
            self.fx = sx*self.fx
            self.fy = sy*self.fy
            self.cx = sx*self.cx
            self.cy = sy*self.cy
            self.W = crop_size[1]
            self.H = crop_size[0]

        # croping will change H, W, cx, cy, so need to change here
        if self.crop_edge is not None:
            self.H -= self.crop_edge*2
            self.W -= self.crop_edge*2
            self.cx -= self.crop_edge
            self.cy -= self.crop_edge


