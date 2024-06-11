import torch

from nerf_slam.utils.registry import Registry

TRACKER_REGISTRY = Registry("TRACKER")

def build_tracker(cfg):
    """
    Build a Tracker for SLAM.
    """
    tracker_name = cfg.TRACKER.NAME
    tracker = TRACKER_REGISTRY.get(tracker_name)(cfg)
    return tracker
