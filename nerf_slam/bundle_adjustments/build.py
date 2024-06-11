import torch

from nerf_slam.utils.registry import Registry

BUNDLE_ADJUSTER_REGISTRY = Registry("BUNDLE_ADJUSTER")

def build_ba(cfg_ba):
    """
    Build a Bundle Adjustement system for SLAM.
    """
    ba_name = cfg_ba.NAME
    ba = BUNDLE_ADJUSTER_REGISTRY.get(ba_name)(cfg_ba)
    return ba
