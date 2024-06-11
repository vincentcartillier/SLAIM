import torch

from nerf_slam.utils.registry import Registry

EVALUATOR_REGISTRY = Registry("EVALUATOR")

def build_evaluator(cfg):
    """
    Build a evaluator for NERF-SLAM
    """
    evaluator_name = cfg.EVALUATOR.NAME
    evaluator = EVALUATOR_REGISTRY.get(evaluator_name)(cfg)
    return evaluator
