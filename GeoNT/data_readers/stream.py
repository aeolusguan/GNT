import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import csv
import os
import cv2
import math
import random
import json
import pickle
import os.path as osp

from .rgbd_utils import *


class RGBDStream(data.Dataset):
    def __init__(self, datapath, frame_rate=-1, image_size=[384,512], crop_size=[0,0]):
        super().__init__()
        self.datapath = datapath
        self.frame_rate = frame_rate
        self.image_size = image_size
        self.crop_size = crop_size
        self._build_dataset_index()

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        return np.load(depth_file)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        """ return training video """
        image = self.__class__.image_read(self.images[index])
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)

        try:
            tstamp = self.tstamps[index]
        except:
            tstamp = index

        pose = torch.from_numpy(self.poses[index]).float()
        intrinsic = torch.from_numpy(self.intrinsics[index]).float()

        # resize image
        sx = self.image_size[1] / image.shape[2]
        sy = self.image_size[0] / image.shape[1]

        image = F.interpolate(image[None], self.image_size, mode='bilinear', align_corners=False)[0]

        fx, fy, cx, cy = intrinsic.unbind(dim=0)
        fx, cx = sx * fx, sx * cx
        fy, cy = sy * fy, sy * cy
        
        # crop image
        if self.crop_size[0] > 0:
            cy = cy - self.crop_size[0]
            image = image[:,self.crop_size[0]:-self.crop_size[0],:]

        if self.crop_size[1] > 0:
            cx = cx - self.crop_size[1]
            image = image[:,:,self.crop_size[1]:-self.crop_size[1]]

        intrinsic = torch.stack([fx, fy, cx, cy])

        return tstamp, image, pose, intrinsic


class ImageStream(data.Dataset):
    def __init__(self, datapath, intrinsics, rate=1, image_size=[384,512]):
        rgb_list = osp.join(datapath, 'rgb.txt')
        if os.path.isfile(rgb_list):
            rgb_list = np.loadtxt(rgb_list, delimiter=' ', dtype=np.unicode_)
            self.timestamps = rgb_list[:,0].astype(np.float)
            self.images = [os.path.join(datapath, x) for x in rgb_list[:,1]]
            self.images = self.images[::rate]
            self.timestamps = self.timestamps[::rate]

        else:
            import glob
            self.images = sorted(glob.glob(osp.join(datapath, '*.jpg'))) +  sorted(glob.glob(osp.join(datapath, '*.png')))
            self.images = self.images[::rate]

        self.intrinsics = intrinsics
        self.image_size = image_size

    def __len__(self):
        return len(self.images)

    @staticmethod
    def image_read(imfile):
        return cv2.imread(imfile)

    def __getitem__(self, index):
        """ return training video """
        image = self.__class__.image_read(self.images[index])

        try:
            tstamp = self.timestamps[index]
        except:
            tstamp = index

        ht0, wd0 = image.shape[:2]
        ht1, wd1 = self.image_size

        intrinsics = torch.as_tensor(self.intrinsics)
        intrinsics[0] *= wd1 / wd0
        intrinsics[1] *= ht1 / ht0
        intrinsics[2] *= wd1 / wd0
        intrinsics[3] *= ht1 / ht0

        # resize image
        ikwargs = {'mode': 'bilinear', 'align_corners': True}
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        image = F.interpolate(image[None], self.image_size, **ikwargs)[0]

        return tstamp, image, intrinsics