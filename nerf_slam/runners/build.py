from nerf_slam.utils.registry import Registry

RUNNER_REGISTRY = Registry("RUNNER")

def build_runner(cfg):
    """
    Build a Runner for SLAM
    """
    runner_name = cfg.RUNNER.NAME
    runner = RUNNER_REGISTRY.get(runner_name)(cfg)
    return runner
