# RGBD-Dataset
from .tartan import TartanAir


def dataset_factory(dataset_list, **kwargs):
    """ create a combined dataset """

    from torch.utils.data import ConcatDataset

    dataset_map = { 'tartan': (TartanAir, 10_000) }
    db_all = None
    for key in dataset_list:
        # cache datasets for faster future loading
        db = dataset_map[key][0](**kwargs)
        size = dataset_map[key][1]

        print("Dataset {} has {} images, sampling {} each epoch".format(key, len(db), size))
        db_all = size @ db if db_all is None else db_all + size @ db
    
    return db_all