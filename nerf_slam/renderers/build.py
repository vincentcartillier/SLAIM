import torch

from nerf_slam.utils.registry import Registry

RENDERER_REGISTRY = Registry("RENDERER")

def build_renderer(cfg):
    """
    Build Renderer for Nerf.
    """
    renderer_name = cfg.RENDERER.NAME
    renderer = RENDERER_REGISTRY.get(renderer_name)(cfg)
    return renderer
