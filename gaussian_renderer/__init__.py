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
import diff_gaussian_rasterization
import gaussian_rasterization_grad
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from scene.cameras import SequentialCamera

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, image_shape = None, update_mask=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        # Gradient retention failed, continue without it
        pass

    if image_shape is None:
        image_shape = (3, viewpoint_camera.image_height, viewpoint_camera.image_width)

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = diff_gaussian_rasterization.GaussianRasterizationSettings(
        image_height=image_shape[1],
        image_width=image_shape[2],
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = diff_gaussian_rasterization.GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    if update_mask is not None:
        rendered_image, radii = rasterizer(
            means3D = means3D[update_mask],
            means2D = means2D[update_mask],
            shs = shs[update_mask],
            colors_precomp = colors_precomp[update_mask] if colors_precomp else colors_precomp,
            opacities = opacity[update_mask],
            scales = scales[update_mask],
            rotations = rotations[update_mask],
            cov3D_precomp = cov3D_precomp[update_mask] if cov3D_precomp else cov3D_precomp)
    else: 
        rendered_image, radii = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}




def render_mask(viewpoint_camera: SequentialCamera, pc : GaussianModel, pipe, bg_color : torch.Tensor, 
                scaling_modifier = 1.0, override_color = None, image_shape = None, pixel_mask=None, 
                color_mask=None, cov_mask=None, render_depth=False, backward_alpha=False, render_flow=False,
                retain_grad=False, update_mask=None, gaussian_mask=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        # Gradient retention failed, continue without it
        pass

    if image_shape is None:
        image_shape = (3, viewpoint_camera.image_height, viewpoint_camera.image_width)

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = gaussian_rasterization_grad.GaussianRasterizationSettings(
        image_height=image_shape[1],
        image_width=image_shape[2],
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        render_depth=render_depth,
        backward_alpha=backward_alpha,
        render_flow=render_flow
    )
    rasterizer = gaussian_rasterization_grad.GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    flow3D  = pc.get_flow
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    if color_mask is not None:
        colors_precomp = viewpoint_camera.colors_precomp
        assert color_mask.shape[0] == colors_precomp.shape[0]
        
    if update_mask is None:
        update_mask = pc.mask_all
    else:
        update_mask = pc.mask_all*update_mask
    assert update_mask.shape[0] == means3D.shape[0]
    assert update_mask.shape[1] == 7
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    
    if pc.frame_idx>1:
        assert means3D.shape[0] == flow3D.shape[0]
        assert means2D.shape[0] == flow3D.shape[0]
        assert means2D.shape[0] == shs.shape[0]
        assert opacity.shape[0] == shs.shape[0]
        assert opacity.shape[0] == rotations.shape[0]
        assert update_mask.shape[0] == rotations.shape[0]
        if pixel_mask is not None:
            assert pixel_mask.shape[-2] == image_shape[1] and pixel_mask.shape[-1] == image_shape[2]
    
    if gaussian_mask is not None:        
        rendered_image, flow2D, infl, count_infl, depth, alpha, radii = rasterizer(
            means3D = means3D[gaussian_mask],
            flow3D = flow3D[gaussian_mask],
            means2D = means2D[gaussian_mask],
            shs = shs[gaussian_mask],
            colors_precomp = colors_precomp[gaussian_mask] if colors_precomp else colors_precomp,
            opacities = opacity[gaussian_mask],
            scales = scales[gaussian_mask],
            rotations = rotations[gaussian_mask],
            cov3D_precomp = cov3D_precomp[gaussian_mask] if cov3D_precomp else cov3D_precomp,
            pixel_mask = pixel_mask,
            color_mask = color_mask[gaussian_mask] if color_mask else color_mask, # 1 means read from colors_precomp, 0 means write to colors_precomp, None means neither
            cov_mask = cov_mask[gaussian_mask] if cov_mask else cov_mask,
            update_mask=update_mask[gaussian_mask],
            )
    else:
        rendered_image, flow2D, infl, count_infl, depth, alpha, radii = rasterizer(
            means3D = means3D,
            flow3D = flow3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            pixel_mask = pixel_mask,
            color_mask = color_mask, # 1 means read from colors_precomp, 0 means write to colors_precomp, None means neither
            cov_mask = cov_mask,
            update_mask=update_mask,
            )
    if retain_grad:
        rendered_image.retain_grad()
        alpha.retain_grad()
        infl.retain_grad()
        depth.retain_grad()
        flow2D.retain_grad()


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    render_pkg = {"render": rendered_image,
                  "flow": None,
                  "alpha": alpha,
                  "influence": infl,
                  "count_influence": count_infl,
                  "viewspace_points": screenspace_points,
                  "visibility_filter" : radii > 0,
                  "radii": radii,
                  "depth": None}
    
    if raster_settings.render_depth:
        render_pkg["depth"] = depth[0]

    if raster_settings.render_flow:
        render_pkg["flow"] = flow2D

    return render_pkg




def render_mask_shift(viewpoint_camera: SequentialCamera, pc : GaussianModel, pipe, bg_color : torch.Tensor, 
                scaling_modifier = 1.0, override_color = None, image_shape = None, pixel_mask=None, 
                color_mask=None, cov_mask=None, render_depth=False, backward_alpha=False, render_flow=False,
                retain_grad=False, update_mask=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0

    if image_shape is None:
        image_shape = (3, viewpoint_camera.image_height, viewpoint_camera.image_width)

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = gaussian_rasterization_grad.GaussianRasterizationSettings(
        image_height=image_shape[1],
        image_width=image_shape[2],
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        render_depth=render_depth,
        backward_alpha=backward_alpha,
        render_flow=False
    )

    rasterizer = gaussian_rasterization_grad.GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz.detach()
    flow3D  = pc.get_flow
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    if color_mask is not None:
        colors_precomp = viewpoint_camera.colors_precomp
        assert color_mask.shape[0] == colors_precomp.shape[0]
        
    if update_mask is None:
        update_mask = pc.mask_all
    else:
        update_mask = pc.mask_all*update_mask
    assert update_mask.shape[0] == means3D.shape[0]
    assert update_mask.shape[1] == 7
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    
    
    rendered_image, flow2D, infl, count_infl, depth, alpha, radii = rasterizer(
        means3D = means3D+flow3D,
        flow3D = flow3D.detach(),
        means2D = means2D.detach(),
        shs = shs.detach(),
        colors_precomp = colors_precomp,
        opacities = opacity.detach(),
        scales = scales.detach(),
        rotations = rotations.detach(),
        cov3D_precomp = cov3D_precomp,
        pixel_mask = pixel_mask,
        color_mask = color_mask, # 1 means read from colors_precomp, 0 means write to colors_precomp, None means neither
        cov_mask = cov_mask,
        update_mask=update_mask,
        )
    if retain_grad:
        rendered_image.retain_grad()
        alpha.retain_grad()
        infl.retain_grad()
        depth.retain_grad()
        flow2D.retain_grad()

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    render_pkg = {"render": rendered_image,
                  "flow": None,
                  "alpha": alpha,
                  "influence": infl,
                  "count_influence": count_infl,
                  "visibility_filter" : radii > 0,
                  "radii": radii,
                  "depth": None}
    
    if raster_settings.render_depth:
        render_pkg["depth"] = depth[0]

    if raster_settings.render_flow:
        render_pkg["flow"] = flow2D

    return render_pkg
