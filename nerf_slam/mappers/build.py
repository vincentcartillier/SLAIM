import torch

from nerf_slam.utils.registry import Registry

MAPPER_REGISTRY = Registry("MAPPER")

def build_mapper(cfg):
    """
    Build a Mapper for Nerf.
    """
    mapper_name = cfg.MAPPER.NAME
    mapper = MAPPER_REGISTRY.get(mapper_name)(cfg)
    return mapper
