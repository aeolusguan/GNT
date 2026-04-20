# This file includes code from SEA-RAFT (https://github.com/princeton-vl/SEA-RAFT)
# Copyright (c) 2024, Princeton Vision & Learning Lab
# Licensed under the BSD 3-Clause License

from __future__ import division
import torch
import numpy as np
from torch.nn import functional as F


class SpatialAug(object):
    def __init__(self, crop, scale=None, rot=None, trans=None, squeeze=None, schedule_coeff=1, order=1, black=False):
        self.crop = crop
        self.scale = scale
        self.rot = rot
        self.trans = trans
        self.squeeze = squeeze
        self.t = np.zeros(6)
        self.schedule_coeff = schedule_coeff
        self.order = order
        self.black = black

    def to_identity(self):
        self.t[0] = 1; self.t[2] = 0; self.t[4] = 0; self.t[1] = 0; self.t[3] = 1; self.t[5] = 0;

    def left_multiply(self, u0, u1, u2, u3, u4, u5):
        result = np.zeros(6)
        result[0] = self.t[0]*u0 + self.t[1]*u2;
        result[1] = self.t[0]*u1 + self.t[1]*u3;

        result[2] = self.t[2]*u0 + self.t[3]*u2;
        result[3] = self.t[2]*u1 + self.t[3]*u3;

        result[4] = self.t[4]*u0 + self.t[5]*u2 + u4;
        result[5] = self.t[4]*u1 + self.t[5]*u3 + u5;
        self.t = result

    def inverse(self):
        result = np.zeros(6)
        a = self.t[0]; c = self.t[2]; e = self.t[4];
        b = self.t[1]; d = self.t[3]; f = self.t[5];

        denom = a*d - b*c;
    
        result[0] = d / denom;
        result[1] = -b / denom;
        result[2] = -c / denom;
        result[3] = a / denom;
        result[4] = (c*f-d*e) / denom;
        result[5] = (b*e-a*f) / denom;
        
        return result
    
    def grid_transform(self, meshgrid, t, normalize=True, gridsize=None):
        if gridsize is None:
            h, w = meshgrid[0].shape
        else:
            h, w = gridsize
        vgrid = torch.cat([(meshgrid[0] * t[0] + meshgrid[1] * t[2] + t[4])[:,:,np.newaxis],
                           (meshgrid[0] * t[1] + meshgrid[1] * t[3] + t[5])[:,:,np.newaxis]],-1)
        if normalize:
            vgrid[:,:,0] = 2.0*vgrid[:,:,0]/max(w-1,1)-1.0
            vgrid[:,:,1] = 2.0*vgrid[:,:,1]/max(h-1,1)-1.0
        return vgrid
    
    def __call__(self, inputs, target):
        h, w, _ = inputs[0].shape
        th, tw = self.crop
        meshgrid = torch.meshgrid([torch.Tensor(range(th)), torch.Tensor(range(tw))])[::-1]
        cornergrid = torch.meshgrid([torch.Tensor([0,th-1]), torch.Tensor([0,tw-1])])[::-1]

        for i in range(50):
            # im0
            self.to_identity()
            #TODO add mirror
            if np.random.binomial(1,0.5):
                mirror = True
            else:
                mirror = False
            ##TODO
            #mirror = False
            if mirror:
                self.left_multiply(-1, 0, 0, 1, .5 * tw, -.5 * th);
            else:
                self.left_multiply(1, 0, 0, 1, -.5 * tw, -.5 * th);
            scale0 = 1; scale1 = 1; squeeze0 = 1; squeeze1 = 1;
            if not self.rot is None:
                rot0 = np.random.uniform(-self.rot[0],+self.rot[0])
                rot1 = np.random.uniform(-self.rot[1]*self.schedule_coeff, self.rot[1]*self.schedule_coeff) + rot0
                self.left_multiply(np.cos(rot0), np.sin(rot0), -np.sin(rot0), np.cos(rot0), 0, 0)
            if not self.trans is None:
                trans0 = np.random.uniform(-self.trans[0],+self.trans[0], 2)
                trans1 = np.random.uniform(-self.trans[1]*self.schedule_coeff,+self.trans[1]*self.schedule_coeff, 2) + trans0
                self.left_multiply(1, 0, 0, 1, trans0[0] * tw, trans0[1] * th)
            if not self.squeeze is None:
                squeeze0 = np.exp(np.random.uniform(-self.squeeze[0], self.squeeze[0]))
                squeeze1 = np.exp(np.random.uniform(-self.squeeze[1]*self.schedule_coeff, self.squeeze[1]*self.schedule_coeff)) * squeeze0
            if not self.scale is None:
                scale0 = np.exp(np.random.uniform(self.scale[2]-self.scale[0], self.scale[2]+self.scale[0]))
                scale1 = np.exp(np.random.uniform(-self.scale[1]*self.schedule_coeff, self.scale[1]*self.schedule_coeff)) * scale0
            self.left_multiply(1.0/(scale0*squeeze0), 0, 0, 1.0/(scale0/squeeze0), 0, 0)

            self.left_multiply(1, 0, 0, 1, .5 * w, .5 * h);
            transmat0 = self.t.copy()

            # im1
            self.to_identity()
            if mirror:
                self.left_multiply(-1, 0, 0, 1, .5 * tw, -.5 * th);
            else:
                self.left_multiply(1, 0, 0, 1, -.5 * tw, -.5 * th);
            if not self.rot is None:
                self.left_multiply(np.cos(rot1), np.sin(rot1), -np.sin(rot1), np.cos(rot1), 0, 0)
            if not self.trans is None:
                self.left_multiply(1, 0, 0, 1, trans1[0] * tw, trans1[1] * th)
            self.left_multiply(1.0/(scale1*squeeze1), 0, 0, 1.0/(scale1/squeeze1), 0, 0)
            self.left_multiply(1, 0, 0, 1, .5 * w, .5 * h);
            transmat1 = self.t.copy()
            transmat1_inv = self.inverse()

            if self.black:
                # black augmentation, allowing 0 values in the input images
                # https://github.com/lmb-freiburg/flownet2/blob/master/src/caffe/layers/black_augmentation_layer.cu
                break
            else:
                if ((self.grid_transform(cornergrid, transmat0, gridsize=[float(h),float(w)]).abs()>1).sum() +\
                    (self.grid_transform(cornergrid, transmat1, gridsize=[float(h),float(w)]).abs()>1).sum()) == 0:
                    break
        if i==49:
            print('max_iter in augmentation')
            self.to_identity()
            self.left_multiply(1, 0, 0, 1, -.5 * tw, -.5 * th);
            self.left_multiply(1, 0, 0, 1, .5 * w, .5 * h);
            transmat0 = self.t.copy()
            transmat1 = self.t.copy()

        # do the real work
        vgrid = self.grid_transform(meshgrid, transmat0,gridsize=[float(h),float(w)])
        inputs_0 = F.grid_sample(torch.Tensor(inputs[0]).permute(2,0,1)[np.newaxis], vgrid[np.newaxis])[0].permute(1,2,0)
        if self.order == 0:
            target_0 = F.grid_sample(torch.Tensor(target).permute(2,0,1)[np.newaxis],    vgrid[np.newaxis], mode='nearest')[0].permute(1,2,0)
        else:    
            target_0 = F.grid_sample(torch.Tensor(target).permute(2,0,1)[np.newaxis],    vgrid[np.newaxis])[0].permute(1,2,0)

        mask_0 = target[:,:,2:3].copy()
        mask_0[mask_0==0]=np.nan
        if self.order == 0:
            mask_0 = F.grid_sample(torch.Tensor(mask_0).permute(2,0,1)[np.newaxis],    vgrid[np.newaxis], mode='nearest')[0].permute(1,2,0)
        else:
            mask_0 = F.grid_sample(torch.Tensor(mask_0).permute(2,0,1)[np.newaxis],    vgrid[np.newaxis])[0].permute(1,2,0)
        mask_0[torch.isnan(mask_0)] = 0

        vgrid = self.grid_transform(meshgrid, transmat1,gridsize=[float(h),float(w)])
        inputs_1 = F.grid_sample(torch.Tensor(inputs[1]).permute(2,0,1)[np.newaxis], vgrid[np.newaxis])[0].permute(1,2,0)

        # flow
        pos = target_0[:,:,:2] + self.grid_transform(meshgrid, transmat0,normalize=False)
        pos = self.grid_transform(pos.permute(2,0,1),transmat1_inv,normalize=False)
        if target_0.shape[2]>=4:
            # scale
            exp = target_0[:,:,3:] * scale1 / scale0
            target = torch.cat([  (pos[:,:,0] - meshgrid[0]).unsqueeze(-1), 
                              (pos[:,:,1] - meshgrid[1]).unsqueeze(-1),
                               mask_0,
                               exp], -1)
        else:
            target = torch.cat([  (pos[:,:,0] - meshgrid[0]).unsqueeze(-1),
                              (pos[:,:,1] - meshgrid[1]).unsqueeze(-1),
                               mask_0], -1)            
#                               target_0[:,:,2].unsqueeze(-1) ], -1)
        inputs = [np.asarray(inputs_0), np.asarray(inputs_1)]
        target = np.asarray(target)
        return inputs,target