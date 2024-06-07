from nerf_slam.utils.registry import Registry

DATASET_REGISTRY = Registry("DATASET")

def build_dataset(cfg):
    """
    Build dataset for Nerf.
    """
    dataset_name = cfg.DATASET.NAME
    dataset = DATASET_REGISTRY.get(dataset_name)(cfg)
    return dataset
