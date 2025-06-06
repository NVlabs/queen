#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.


import torch
from torch import nn
import numpy as np
import re
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixOffCenter

def imageName_from_Path(path):
    if "camera_" in path:
        match = re.match(r'.*(camera_\d*).*\/(.*)\.png',path)
    else:
        match = re.match(r'.*(cam\d*).*\/(.*)\.png',path)
    return match.group(1)+'_'+match.group(2)

def camName_from_Path(path):
    if "camera_" in path:
        match = re.match(r'.*(camera_\d*).*\/(.*)\.png',path)
    else:
        match = re.match(r'.*(cam\d*).*\/(.*)\.png',path)
    return match.group(1)


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 frame_idx = 0, znear=0.01, zfar=100.0
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = zfar
        self.znear = znear

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class SequentialCamera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, uid, image_wh,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 image=None, image_path=None, gt_alpha_mask=None, frame_idx = 0,
                 znear=0.01, zfar=100.0, principle_point=None
                 ):
        super(SequentialCamera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_path = image_path
        self.image_name = imageName_from_Path(image_path) if image_path else None
        self.frame_idx = frame_idx

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image
        self.image_width = image_wh[0]
        self.image_height = image_wh[1]

        self.zfar = zfar
        self.znear = znear

        self.trans = trans
        self.scale = scale

        self.colors_precomp = None
        self.cov2d_precomp = None
        self.gt_depth = None
        self.err_scale = None
        self.pix_thresh_vals = None

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        if principle_point is None:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        else:
            self.projection_matrix = getProjectionMatrixOffCenter(znear=self.znear, zfar=self.zfar, 
                                                                   fovX=self.FoVx, fovY=self.FoVy,
                                                                   cx=principle_point[0], cy=principle_point[1],
                                                                   width=self.image_width, height=self.image_height).transpose(0,1).cuda()
            
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
        if self.original_image is not None:
            self.update_image(self.original_image, self.image_path, frame_idx, gt_alpha_mask)

    def update_image(self, image, image_path, frame_idx, gt_alpha_mask):
        assert self.image_height == image.shape[1] and self.image_width == image.shape[2]
        self.image_path = image_path
        self.image_name = imageName_from_Path(image_path) if image_path else None
        self.frame_idx = frame_idx
        new_image = image.clamp(0.0, 1.0).to(self.data_device)
        if frame_idx>1:
            self.image_diff = torch.abs(self.original_image-new_image)
            self.image_diff.requires_grad_(False)
        self.original_image = new_image
        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)

class MiniCam:
    def __init__(self, uid, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, camera_center=None, frame_idx=0):
        self.uid = uid
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        if camera_center is None:
            self.camera_center = view_inv[3][:3]
        else:
            self.camera_center = camera_center
        self.frame_idx = frame_idx

