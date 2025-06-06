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
import math
import numpy as np
from typing import NamedTuple
from utils.image_utils import coords_grid
import torch.nn.functional as F

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getRT(W2C, translate=np.array([.0, .0, .0]), scale=1.0):
    C2W = np.linalg.inv(W2C.transpose(0, 1))
    cam_center = C2W[:3,3]
    cam_center = cam_center/scale - translate
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    R = Rt[:3,:3].transpose()
    T = Rt[:3, 3]
    return R, T

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getProjectionMatrixOffCenter(znear, zfar, fovX, fovY, 
                                 cx, cy, width, height):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = 2*(cx/width)-1
    P[1, 2] = 2*(cy/height)-1
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getProjectionMatrixInv(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    P[0, 0] = (right - left)/(2.0*znear)
    P[1, 1] = (top - bottom)/(2*znear)
    P[2, 3] = 1
    P[3, 2] = -(zfar-znear)/(zfar*znear)
    P[3, 3] = 1/znear
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def get_pairwise_distances(X,Y, metric='euclidean'):
    dot_pdt = torch.mm(X,Y.t())
    if metric=='euclidean': 
        square_norm_1 = torch.sum(X**2, dim = 1).unsqueeze(1)
        square_norm_2 = torch.sum(Y**2, dim = 1).unsqueeze(1)
        return torch.sqrt(torch.abs(square_norm_1 + square_norm_2.t() - 2.0*dot_pdt))
    elif metric=='cosine':
        Xnorm = X/torch.norm(X, p=2, dim=1).unsqueeze(1)
        Ynorm = Y/torch.norm(Y, p=2, dim=1).unsqueeze(1)
        return torch.abs(1.0 - torch.mm(Xnorm, Ynorm.t()))

def knn_gpu(X,Y, k, device):
    n = X.shape[0]
    batchsize = 1024
    n_batches = n//1024 +(1 if n%1024!=0 else 0)
    knn = np.zeros((n,k))
    X = torch.Tensor(X).to(device)
    for i in range(n_batches):
        dists = get_pairwise_distances(X[i*batchsize:(i+1)*batchsize],Y)
        _, indices = torch.topk(dists, k, dim=1, largest=False, sorted=True)
        knn[i*batchsize:(i+1)*batchsize] = indices.detach().cpu().numpy()
    return knn


def adjust_depths(cameras, colmap_xyz):


    downsample_scale = 6
    _,orig_H,orig_W = cameras[0].original_image.shape
    n_cams = len(cameras)
    downsample_size = (int(orig_H/downsample_scale),int(orig_W/downsample_scale))
    H,W = downsample_size
    xyz_2d = coords_grid(1,H,W, device='cuda')[0].permute(1,2,0).view(-1,2)
    # xyz = torch.cat((xyz,torch.zeros_like(xyz[:,0:1]),torch.ones_like(xyz[:,0:1])),dim=-1) # N x 4
    xyz_2d[:,0] = xyz_2d[:,0]/(0.5*W)+(1/W-1)
    xyz_2d[:,1] = xyz_2d[:,1]/(0.5*H)+(1/H-1)

    xyz_2d = torch.tensor(xyz_2d, requires_grad=True)

    znears = torch.tensor([cam.znear for cam in cameras],requires_grad = True, dtype=torch.float32, device='cuda')
    zfars = torch.tensor([cam.zfar for cam in cameras],requires_grad = True, dtype=torch.float32, device='cuda')
    fovx = torch.tensor([cam.FoVx for cam in cameras],requires_grad = True, dtype=torch.float32, device='cuda')
    fovy = torch.tensor([cam.FoVy for cam in cameras],requires_grad = True, dtype=torch.float32, device='cuda')
    # torch.nn.init.ones_(scales)
    # torch.nn.init.zeros_(shifts)
    iterations = 300
    params = [{'name': 'znears', 'params': znears, 'lr':1.0e-1},
              {'name': 'zfars', 'params': zfars, 'lr':5.0e-1},
              {'name': 'fovx', 'params': fovx, 'lr':1.0e-1},
              {'name': 'fovy', 'params': fovy, 'lr':1.0e-1}]
    loss = torch.nn.L1Loss()
    optimizer = torch.optim.SGD(params, weight_decay=0.0, nesterov=True, momentum=0.9)


    indices = torch.randint(0, n_cams, (iterations,1))
    # indices = indices[indices[:,0]!=indices[:,1]][:iterations]
    # indices = indices[indices!=gt_idx][:iterations]

    from tqdm import tqdm
    progress_bar = tqdm(range(1, iterations+1), desc="Depth adjustment progress")
    losses = []
    for it in range(iterations):
        src_idx = indices[it]
        with torch.no_grad():
            src_depth = cameras[src_idx].gt_depth.unsqueeze(0).unsqueeze(0)
            src_rt = cameras[src_idx].world_view_transform.T
            src_depth = F.interpolate(src_depth, size=downsample_size, mode='bilinear',
                                    align_corners=True)[0]
        view = getProjectionMatrix(znears[src_idx], zfars[src_idx], fovx[src_idx], fovy[src_idx]).cuda()
        viewinv = getProjectionMatrixInv(znears[src_idx], zfars[src_idx], fovx[src_idx], fovy[src_idx]).cuda()
        breakpoint()
            
        xyz = torch.cat((xyz_2d,src_depth.view(-1,1), torch.ones_like(src_depth.view(-1,1))),dim=1)

        vinv = torch.linalg.inv(view).unsqueeze(0) # 1 x 4 x 4
        rtinv = torch.linalg.inv(src_rt).unsqueeze(0)
        homw = 1/(torch.linalg.vecdot(vinv[0:1,3,:], xyz)+1.0e-6)
        cam_src_xyz = torch.matmul(vinv,(xyz*homw.unsqueeze(-1)).unsqueeze(-1)).squeeze(-1)
        world_src_xyz = torch.matmul(rtinv, cam_src_xyz.unsqueeze(-1)).squeeze(-1)

        batch_size = 10000
        n_batches = colmap_xyz.shape[0]//batch_size +(1 if colmap_xyz.shape[0]%batch_size!=0 else 0)

        loss = 0.0
        for i in range(n_batches):
            dists = get_pairwise_distances(colmap_xyz[i*batch_size:(i+1)*batch_size], world_src_xyz[:,:3])
            loss += torch.min(dists,dim=1)[0].sum()
        loss = loss/colmap_xyz.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if it%100==0:
            progress_bar.set_postfix({"Loss": loss.item()})
            progress_bar.update(100)
        # with torch.no_grad():
        #     scales[gt_idx] = 1
        #     shifts[gt_idx] = 0
        
    breakpoint()

    progress_bar.close()

    import matplotlib.pyplot as plt
    plt.clf()
    plt.plot(losses)
    plt.show()
    plt.savefig('./output/add_depth_v2/losses.png')

    breakpoint()
    fused_point_cloud, fused_color = None, None
    world_coords_colmap = torch.cat((self._xyz,torch.ones_like(self._xyz[:,0:1])),dim=1)