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
import torch.nn as nn
import pathlib
import torchvision
from PIL import Image
import cv2
import numpy as np
import torch.nn.functional as F
from utils.general_utils import kthvalue
from typing import Union, List, Optional, BinaryIO
    

def bilinear_interpolate_torch(im, x, y):
    dtype = torch.cuda.FloatTensor
    dtype_long = torch.cuda.LongTensor
    x0 = torch.floor(x).type(dtype_long)
    x1 = x0 + 1
    
    y0 = torch.floor(y).type(dtype_long)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1]-1)
    x1 = torch.clamp(x1, 0, im.shape[1]-1)
    y0 = torch.clamp(y0, 0, im.shape[0]-1)
    y1 = torch.clamp(y1, 0, im.shape[0]-1)
    Ia = im[ y0, x0 ]#[0]
    Ib = im[ y1, x0 ]#[0]
    Ic = im[ y0, x1 ]#[0]
    Id = im[ y1, x1 ]#[0]
    
    wa = (x1.type(dtype)-x) * (y1.type(dtype)-y)
    wb = (x1.type(dtype)-x) * (y-y0.type(dtype))
    wc = (x-x0.type(dtype)) * (y1.type(dtype)-y)
    wd = (x-x0.type(dtype)) * (y-y0.type(dtype))

    return torch.t((torch.t(Ia)*wa)) + torch.t(torch.t(Ib)*wb) + torch.t(torch.t(Ic)*wc) + torch.t(torch.t(Id)*wd)

def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid

def coords_grid_proj(camera, downsample_scale=1):
    _,orig_H,orig_W = camera.original_image.shape
    downsample_size = (int(orig_H/downsample_scale),int(orig_W/downsample_scale))
    H,W = downsample_size
    xyz = coords_grid(1,H,W, device='cuda')[0].permute(1,2,0).view(-1,2)
    xyz = torch.cat((xyz,torch.zeros_like(xyz[:,0:1]),torch.ones_like(xyz[:,0:1])),dim=-1) # N x 4
    xyz[:,0] = xyz[:,0]/(0.5*W)+(1/W-1)
    xyz[:,1] = xyz[:,1]/(0.5*H)+(1/H-1)
    return xyz

def bilinear_sample(img, sample_coords, mode='bilinear', padding_mode='zeros', return_mask=False):
    # img: [B, C, H, W]
    # sample_coords: [B, 2, H, W] in image scale
    if sample_coords.size(1) != 2:  # [B, H, W, 2]
        sample_coords = sample_coords.permute(0, 3, 1, 2)

    b, _, h, w = sample_coords.shape

    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    img = F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    if return_mask:
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (y_grid <= 1)  # [B, H, W]

        return img, mask

    return img

def flow_warp(feature, flow, grid=None, padding_mode='zeros'):
    b, c, h, w = feature.size()
    assert flow.size(1) == 2

    if grid is None:
        grid = coords_grid(b, h, w).to(flow.device) # [B, 2, H, W]

    return bilinear_sample(feature, grid+flow, padding_mode=padding_mode, return_mask=False)

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    maxi = 255
    mse = (((torch.round(img1*maxi) - torch.round(img2*maxi))) ** 2).mean()
    return 20 * torch.log10(maxi / torch.sqrt(mse))

def scale_images(image): # [N, C, H, W]
    num_images = image.shape[0]
    min_vals = image.view(num_images,-1).min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
    max_vals = image.view(num_images,-1).max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
    scaled = (image-min_vals)/(max_vals-min_vals)
    return scaled, min_vals, max_vals

def resize_dims(orig_size, scale=1.0):
    if scale == 1.0:
        return orig_size
    assert scale>0.0 and scale<1.0

    rescaled_size = (int(orig_size[0]*scale), int(orig_size[1]*scale))
    return rescaled_size

@torch.no_grad()
def save_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[str, pathlib.Path, BinaryIO],
) -> None:
    ndarr = torch.round(tensor*255).permute(1,2,0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp)

def write_depth(path, depth, grayscale, bits=1):
    """Write depth map to png file.

    Args:
        path (str): filepath without extension
        depth (array): depth
        grayscale (bool): use a grayscale colormap?
    """
    if not grayscale:
        bits = 1

    if not np.isfinite(depth).all():
        depth=np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        print("WARNING: Non-finite depth values present")

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.dtype)

    if not grayscale:
        out = cv2.applyColorMap(np.uint8(out), cv2.COLORMAP_INFERNO)

    if bits == 1:
        cv2.imwrite(path + ".png", out.astype("uint8"))
    elif bits == 2:
        cv2.imwrite(path + ".png", out.astype("uint16"))

    return

def downsample_image(original_image, scale=1.0):

    assert scale<=1.0, "Scale must be <= 1.0"
    if scale == 1.0:
        return original_image
    else:
        return nn.functional.interpolate(original_image.unsqueeze(0), scale_factor=scale, mode='bilinear', align_corners=False).squeeze(0)

def resize_image(original_image, scale=1.0):

    assert scale<=1.0, "Scale must be <= 1.0"
    if scale == 1.0:
        return original_image
    else:
        _, image_height, image_width = original_image.shape
        resized = nn.functional.interpolate(original_image.unsqueeze(0), scale_factor=scale, mode='bilinear', align_corners=False).squeeze(0)
        return nn.functional.interpolate(resized.unsqueeze(0), size=(image_height, image_width), mode='bilinear', align_corners=False).squeeze(0)

def blur_image(scale, transform):

    assert scale<=1.0, "Scale must be <= 1.0"
    if scale == 1.0:
        return torch.nn.Identity()
    else:
        kernel_radius=int(transform.split("_")[1])
        cur_radius = round(kernel_radius*(1-scale))
        kernel_size=2*cur_radius+1
        sigma=float(transform.split("_")[2])
        if sigma == 0:
            cur_sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        else:
            cur_sigma = sigma*(1-scale)
        transform = torchvision.transforms.GaussianBlur((kernel_size,kernel_size), sigma=cur_sigma).cuda()
        return transform
    
@torch.no_grad()
def get_flow(model, image1, image2, padding_factor = 16, inference_size = None, flow_batch = -1):


    if image1.dim() == 3:
        image1 = image1[None]
    if image2.dim() == 3:
        image2 = image2[None]

    
    ori_size = image1.shape[-2:]

    # resize before inference
    if inference_size is not None:
        assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
        nearest_size = [int(np.ceil(inference_size[-2] / padding_factor)) * padding_factor,
                        int(np.ceil(inference_size[-1] / padding_factor)) * padding_factor]
    else:
        nearest_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
                        int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]
        

    image1 = F.interpolate(image1, size=nearest_size, mode='bilinear',
                            align_corners=True)
    image2 = F.interpolate(image2, size=nearest_size, mode='bilinear',
                            align_corners=True)
    
    flow = None
    batchsize = image1.shape[0] if flow_batch==-1 else flow_batch
    n_batches = np.ceil(image1.shape[0]/batchsize).astype(np.int32)

    for batch_idx in range(n_batches):
        curimage1 = image1[batch_idx*batchsize:(batch_idx+1)*batchsize]
        curimage2 = image2[batch_idx*batchsize:(batch_idx+1)*batchsize]
        results_dict = model(curimage1, curimage2,
                                attn_type='swin',
                                attn_splits_list=[2],
                                corr_radius_list=[-1],
                                prop_radius_list=[-1],
                                num_reg_refine=1,
                                task='flow',
                                pred_bidir_flow=False,
                                )

        flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]
        # flow_pr = flow_pr[batchsize:]

        # resize back
        if nearest_size[-2]!=ori_size[-2] or nearest_size[-1]!=ori_size[-1] or inference_size is not None:
            flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                    align_corners=True)
            
        flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / nearest_size[-1]
        flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / nearest_size[-2]

        if flow is None:
            flow = flow_pr
        else:
            flow = torch.cat((flow,flow_pr),dim=0)

    return flow


@torch.no_grad()
def get_depth(model, image1, pose_data,  padding_factor=16, inference_size=None, batchsize=None):
    
    intrinsics, poses = pose_data
    if poses.dim()==3:
        assert poses.shape[0] == image1.shape[0]

    if image1.dim() == 3:
        image1 = image1[None]

    if batchsize is None:
        batchsize = image1.shape[0]
    num_batches = image1.shape[0] // batchsize + (1 if image1.shape[0]%batchsize != 0 else 0)

    ori_size = image1.shape[-2:]

    # resize before inference
    if inference_size is not None:
        assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
        nearest_size = [int(np.ceil(inference_size[-2] / padding_factor)) * padding_factor,
                        int(np.ceil(inference_size[-1] / padding_factor)) * padding_factor]
    else:
        nearest_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
                        int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]
        
    scale_y, scale_x = nearest_size[-2]/ori_size[-2], nearest_size[-1]/ori_size[-1]
    intrinsics[:,0] *= scale_y
    intrinsics[:,1] *= scale_x

    image1 = F.interpolate(image1, size=nearest_size, mode='bilinear',
                            align_corners=True)
        
    pr_depth = None
    for batch_idx in range(num_batches):
        pred_depth = model(image1[batch_idx*batchsize:(batch_idx+1)*batchsize], 
                           image1[batch_idx*batchsize:(batch_idx+1)*batchsize],
                            attn_type='swin',
                            attn_splits_list=[2],
                            prop_radius_list=[-1],
                            num_reg_refine=1,
                            intrinsics=intrinsics[batch_idx*batchsize:(batch_idx+1)*batchsize],
                            pose=poses[batch_idx*batchsize:(batch_idx+1)*batchsize],
                            min_depth=1. / 10,
                            max_depth=1. / 0.5,
                            num_depth_candidates=64,
                            pred_bidir_depth=False,
                            depth_from_argmax=False,
                            task='depth',
                            )['flow_preds'][-1]   # [1, H, W]
        if batch_idx == 0:
            pr_depth = pred_depth
        else:
            pr_depth = torch.cat((pr_depth,pred_depth),dim=0)
    
    # pred_depth = infer_depth_multi(model, image1, (intrinsics.float(), pose.float()))
        
    # resize back
    if nearest_size[-2]!=ori_size[-2] or nearest_size[-1]!=ori_size[-1] or inference_size is not None:
        pr_depth = F.interpolate(pr_depth.unsqueeze(1), size=ori_size, mode='bilinear',
                                align_corners=True)

    return pr_depth


@torch.no_grad()
def get_mask(prev_frame_views, cur_frame_views, flow, thresh):
    n_cams = prev_frame_views.shape[0]
    num_quantized = 50
    flow_norm = torch.norm(flow,dim=1,keepdim=True)

    _, _, H, W = prev_frame_views.shape

    err_map = torch.norm(prev_frame_views - cur_frame_views,dim=1,keepdim=True).view(prev_frame_views.shape[0],-1) # N_views x H*W*3
    total_mse = err_map.pow(2).mean(dim=-1, keepdim=True).sqrt() # N_views x 1
    pct = []

    minvals, maxvals = err_map.min(dim=-1,keepdim=True)[0], err_map.max(dim=-1,keepdim=True)[0]
    minvals = (maxvals-minvals)*0.0+minvals
    maxvals = (maxvals-minvals)*0.1+minvals
    thresh_values = torch.tensor(np.linspace(minvals.detach().cpu().numpy(),maxvals.detach().cpu().numpy(),num_quantized)).squeeze(-1).permute(1,0).to(err_map).int() #  N_views x num_quantized

    masks = err_map.unsqueeze(-1)>thresh_values.unsqueeze(1) # N_views x  H*W*3 x num_quantized
    mse = torch.sqrt((err_map.unsqueeze(-1)*masks.float()).pow(2).sum(dim=1)/(H*W*1)) # N_views x num_quantized
    psnr_drop = mse/(total_mse-mse)


    x = masks.sum(dim=1)/(H*W)
    y =  psnr_drop

    dist = torch.abs(x/y-1.0)
    dist = torch.sqrt(x.pow(2)+y.pow(2))
    indices = torch.argmin(dist,dim=1)

    thresh_values = thresh_values[torch.arange(thresh_values.shape[0]),indices]
    return err_map.view(prev_frame_views.shape[0],1,H,W)>thresh_values.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), thresh_values.float()


    total_mse = err_map.pow(2).mean().sqrt() # N_views x 1
    thresh_values = torch.linspace((err_map.max()-err_map.min())*0.1+err_map.min(),err_map.max(),100).to(err_map)
    for thresh in thresh_values:
        static_mse = torch.sqrt(err_map[err_map<thresh].pow(2).sum()/(H*W*3))
        static_pct = static_mse.item()/total_mse.item()
        pct.append(-20*np.log10(static_pct))

    thresh_values = thresh_values[np.argmin(np.abs(np.array(pct)))]
    return err_map>thresh_values, thresh_values

def l1_loss(img1, img2):
    """Calculate L1 loss between two images.
    
    Args:
        img1 (torch.Tensor): First image, shape (B, C, H, W)
        img2 (torch.Tensor): Second image, shape (B, C, H, W)
        
    Returns:
        torch.Tensor: L1 loss value
    """
    return torch.abs(img1 - img2).mean()

    