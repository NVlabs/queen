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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    flow3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    pixel_mask,
    color_mask,
    cov_mask,
    update_mask,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        flow3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        pixel_mask,
        color_mask,
        cov_mask,
        update_mask,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        flow3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        pixel_mask,
        color_mask,
        cov_mask,
        update_mask,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            flow3D,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            pixel_mask,
            color_mask,
            cov_mask,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            raster_settings.render_depth,
            True, # Always render alpha for forward, needed for backward
            raster_settings.render_flow
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, flow2D, depth, alpha, infl, count_infl, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, flow2D, depth, alpha, infl, count_infl, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)


        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, flow3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, 
                              binningBuffer, imgBuffer, alpha, update_mask, pixel_mask)
        return color, flow2D, infl, count_infl, depth, alpha, radii

    @staticmethod
    def backward(ctx, grad_out_color, grad_out_flow2D, grad_infl, grad_count_infl, grad_depth, grad_alpha, grad_radii):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, flow3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, alpha, update_mask, pixel_mask = ctx.saved_tensors

        pixel_mult = (pixel_mask>0.0).unsqueeze(0) if pixel_mask.shape[0]>0 else 1
        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                flow3D,
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                update_mask,
                pixel_mask,
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color*pixel_mult, 
                grad_out_flow2D*pixel_mult if raster_settings.render_flow else torch.Tensor([]),
                grad_depth*pixel_mult if raster_settings.render_depth else torch.Tensor([]),
                grad_alpha*pixel_mult if raster_settings.backward_alpha else torch.Tensor([]),
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                alpha,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_flow3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_flow3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_flow3D if raster_settings.render_flow else None,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None, # pixel_mask
            None, # color_mask
            None, # cov_mask
            None, # update_mask
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    render_depth: bool
    backward_alpha: bool
    render_flow: bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, flow3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None, pixel_mask = None, 
                color_mask = None, cov_mask = None, update_mask = None):
        
        raster_settings = self.raster_settings

        if color_mask is None and ((shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None)):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        elif (color_mask is not None and colors_precomp is None) :
            raise Exception('Color mask provided for reading/writing from colors_precomp but colors_precomp is not provided!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        if pixel_mask is None:
            pixel_mask = torch.Tensor([])
        if color_mask is None:
            color_mask = torch.Tensor([]).bool()
        if cov_mask is None:
            cov_mask = torch.Tensor([]).bool()

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            flow3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp, 
            pixel_mask,
            color_mask,
            cov_mask,
            update_mask,
            raster_settings,
        )

