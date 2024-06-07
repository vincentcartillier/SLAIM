from .build import DATASET_REGISTRY, build_dataset

from .tum_instant_ngp import TUMDatasetInstantNGP
from .replica_instant_ngp import ReplicaDatasetInstantNGP
from .scannet_instant_ngp import ScannetDatasetInstantNGP

__all__ = list(globals().keys())
