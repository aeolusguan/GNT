from .arkit import ARKitScenes
from .dynamic_replica import DynamicReplica
from .tartan import TartanAir


def dataset_factory(dataset_config, **kwargs):
    """Create the fixed three-dataset training mixture."""
    tartan_config = dataset_config.tartanair
    arkit_config = dataset_config.arkitscenes
    dynamic_config = dataset_config.dynamic_replica
    tartan = TartanAir(datapath=tartan_config.root, **kwargs)
    arkit = ARKitScenes(datapath=arkit_config.root, **kwargs)
    dynamic = DynamicReplica(datapath=dynamic_config.root, **kwargs)

    print(
        f"TartanAir has {len(tartan)} samples; "
        f"sampling {tartan_config.samples_per_epoch} each epoch"
    )
    print(
        f"ARKitScenes has {len(arkit)} samples; "
        f"sampling {arkit_config.samples_per_epoch} each epoch"
    )
    print(
        f"Dynamic Replica has {len(dynamic)} samples; "
        f"sampling {dynamic_config.samples_per_epoch} each epoch"
    )
    return (
        tartan_config.samples_per_epoch @ tartan
        + arkit_config.samples_per_epoch @ arkit
        + dynamic_config.samples_per_epoch @ dynamic
    )
