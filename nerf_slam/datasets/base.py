import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    
    def _parse_cfg_base(self,cfg):
        self.name = cfg.DATASET.NAME
        self.device = cfg.MODEL.DEVICE
        self.scale = cfg.RENDERER.SCALE
        
        self.png_depth_scale = cfg.DATASET.PNG_DEPTH_SCALE 

        self.H = cfg.RENDERER.IMAGE_H
        self.W = cfg.RENDERER.IMAGE_W
        self.fx = cfg.RENDERER.FX
        self.fy = cfg.RENDERER.FY
        self.cx = cfg.RENDERER.CX
        self.cy = cfg.RENDERER.CY
        
        self.input_folder = cfg.DATASET.INPUT_FOLDER

        if cfg.RENDERER.DISTORTION.ENABLED:
            self.distortion = np.array(cfg.RENDERER.DISTORTION.PARAMS)
        else:
            self.distortion = None

        if cfg.DATASET.CROP_SIZE.ENABLED:
            self.crop_size = cfg.DATASET.CROP_SIZE.VALUE
        else:
            self.crop_size = None
        
        if cfg.DATASET.CROP_EDGE.ENABLED:
            self.crop_edge = cfg.DATASET.CROP_EDGE.VALUE
        else:
            self.crop_edge = None



    def __init__(self, cfg):
        super(BaseDataset, self).__init__()

        self._parse_cfg_base(cfg)




def readEXR_onlydepth(filename):
    """
    Read depth data from EXR image file.

    Args:
        filename (str): File path.

    Returns:
        Y (numpy.array): Depth buffer in float32 format.
    """
    # move the import here since only CoFusion needs these package
    # sometimes installation of openexr is hard, you can run all other datasets
    # even without openexr
    import Imath
    import OpenEXR as exr

    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    dw = header['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    for c in header['channels']:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    Y = None if 'Y' not in header['channels'] else channelData['Y']

    return Y



