# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data

import os
import random
import h5py
from glob import glob
import os.path as osp

from .utils import frame_utils
from .utils.augmentor import FlowAugmentor, SparseFlowAugmentor
from .utils.utils import induced_flow, check_cycle_consistency

class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        self.dataset = 'unknown'
        self.subsample_groundtruth = False

        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.mask_list = []
        self.extra_info = []

    def __getitem__(self, index):
        while True:
            try:
                return self.fetch(index)
            except Exception as e:
                index = random.randint(0, len(self) - 1)
    
    def fetch(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]

            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]
        
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            if self.dataset == 'TartanAir':
                flow = np.load(self.flow_list[index])
                valid = np.load(self.mask_list[index])
                # rescale the valid mask to [0, 1]
                valid = 1 - valid / 100

            elif self.dataset == 'MegaDepth':
                depth0 = np.array(h5py.File(self.extra_info[index][0], 'r')['depth'])
                depth1 = np.array(h5py.File(self.extra_info[index][1], 'r')['depth'])
                camera_data = self.megascene[index]
                flow_01, flow_10 = induced_flow(depth0, depth1, camera_data)
                valid = check_cycle_consistency(flow_01, flow_10)
                flow = flow_01
            else:
                flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            if self.dataset == 'Infinigen':
                # Infinigen flow is stored as a 3D numpy array, [Flow, Depth]
                flow = np.load(self.flow_list[index])
                flow = flow[..., :2]
            elif self.dataset == 'vKITTI':
                flow = frame_utils.readFlowvKITTI(self.flow_list[index])
            else:
                flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        if self.subsample_groundtruth:
            # use only every second value in both spatial directions ==> flow will have some dimensions as images
            # used for spring dataset
            flow = flow[::2, ::2]

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        flow[torch.isnan(flow)] = 0
        flow[flow.abs() > 1e9] = 0

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float()
    
    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
    
    def __len__(self):
        return len(self.image_list)
    
class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/Sintel', dstype='clean'):
        super().__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True
        
        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))

class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingChairs/data'):
        super().__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        for i in range(len(flows)):
            self.flow_list += [ flows[i] ]
            self.image_list += [ [images[2*i], images[2*i+1]] ]

class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingThings3D/', dstype='frames_cleanpass', split='TRAIN'):
        super().__init__(aug_params)

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, split, '*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/%s/*/*'%split)))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                        elif direction == 'into_past':
                            self.image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]

class TartanAir(FlowDataset):

    # scale depths to balance rot & trans
    DEPTH_SCALE = 5.0

    def __init__(self, aug_params=None, root='datasets/TartanAir/', split='train'):
        super(TartanAir, self).__init__(aug_params, sparse=True)
        self.n_frames = 2
        self.dataset = 'TartanAir'
        self.root = root
        self.split = split
        self._build_dataset()

    def _build_dataset(self):
        scenes = glob(osp.join(self.root, '*/*/*'))
        for scene in sorted(scenes):
            if self.split == 'train' and ('westerndesert' in scene or 'soulcity' in scene):
                continue
            elif self.split == 'val' and ('westerndesert' not in scene and 'soulcity' not in scene):
                continue
            images = sorted(glob(osp.join(scene, 'image_left/*.png')))
            for idx in range(len(images) - 1):
                frame0 = str(idx).zfill(6)
                frame1 = str(idx + 1).zfill(6)
                self.image_list.append([images[idx], images[idx + 1]])
                self.flow_list.append(osp.join(scene, 'flow', f"{frame0}_{frame1}_flow.npy"))
                self.mask_list.append(osp.join(scene, 'flow', f"{frame0}_{frame1}_mask.npy"))
        if self.split == 'val':
            self.image_list = self.image_list[::100]
            self.flow_list = self.flow_list[::100]
            self.mask_list = self.mask_list[::100]

class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI/2015'):
        super().__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))

class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1K/'):
        super().__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [ [images[i], images[i+1]] ]

            seq_ix += 1

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def fetch_dataloader(args, rank=0, world_size=1):
    """ Create the data loader for the corresponding training set """

    if args.dataset == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': args.scale - 0.1, 'max_scale': args.scale + 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params)
    elif args.dataset == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': args.scale - 0.4, 'max_scale': args.scale + 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset
    elif args.dataset == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': args.scale - 0.2, 'max_scale': args.scale + 0.6, 'do_flip': True}
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')
        train_dataset = sintel_clean + sintel_final
    elif args.dataset == 'tartanair':
        aug_params = {'crop_size': args.image_size, 'min_scale': args.scale - 0.2, 'max_scale': args.scale + 0.4, 'do_flip': True}
        train_dataset = TartanAir(aug_params)
    elif args.dataset == 'TSKH':
        aug_params = {'crop_size': args.image_size, 'min_scale': args.scale - 0.2, 'max_scale': args.scale + 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')
        kitti = KITTI({'crop_size': args.image_size, 'min_scale': args.scale - 0.3, 'max_scale': args.scale + 0.5, 'do_flip': True})
        hd1k = HD1K({'crop_size': args.image_size, 'min_scale': args.scale - 0.5, 'max_scale': args.scale + 0.2, 'do_flip': True})
        train_dataset = 20 * sintel_clean + 20 * sintel_final + 80 * kitti + 30 * hd1k + things

    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    assert (
        args.batch_size > 0 and args.batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({})".format(
        args.batch_size, world_size
    )
    batch_size = args.batch_size // world_size 

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8,
                                   pin_memory=True, drop_last=True, worker_init_fn=seed_worker, sampler=train_sampler)
    
    print('Training with %d image pairs' % len(train_dataset))
    return train_loader, train_sampler