from .arkit import ARKitScenes
from .dynamic_replica import DynamicReplica
from .point_odyssey import PointOdyssey
from .scannet import ScanNet
from .tartan import TartanAir
from .waymo import Waymo


DATASET_CLASSES = {
    "tartanair": TartanAir,
    "arkitscenes": ARKitScenes,
    "dynamic_replica": DynamicReplica,
    "point_odyssey": PointOdyssey,
    "scannet": ScanNet,
    "waymo": Waymo,
}


def dataset_factory(dataset_config, **kwargs):
    """Create the training mixture described by the YAML data section."""
    weighted_datasets = []
    for name, config in dataset_config.items():
        dataset = DATASET_CLASSES[name](datapath=config.root, **kwargs)
        print(
            f"{dataset.name} has {len(dataset)} samples; "
            f"sampling {config.samples_per_epoch} each epoch"
        )
        weighted_datasets.append(config.samples_per_epoch @ dataset)

    mixture = weighted_datasets[0]
    for dataset in weighted_datasets[1:]:
        mixture = mixture + dataset
    return mixture
