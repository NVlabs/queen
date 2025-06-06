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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import math
import pickle
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud, focal2fov, fov2focal, getWorld2View2, getProjectionMatrix
from collections import OrderedDict
import torch.nn.functional as F
from utils.general_utils import strip_symmetric, build_scaling_rotation, warp_depth
from utils.image_utils import coords_grid
from utils.graphics_utils import knn_gpu
from utils.compress_utils import CompressedLatents, init_latents
from arguments import QuantizeParams, ModelParams
from scene.decoders import LatentDecoder, DecoderIdentity, DecoderLayer, LatentDecoderRes, Gate
import time

class GaussianModel:
    """3D Gaussian Splatting model with quantized latent representations and temporal consistency."""

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, latent_args: QuantizeParams, model_args: ModelParams, frame_idx : int = 1, use_xyz_legacy: bool = False):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self.use_xyz_legacy = use_xyz_legacy
        print(f"Using xyz_legacy mode: {self.use_xyz_legacy}")

        # Initialize latent parameter storage
        self._latents = OrderedDict([(n,torch.empty(0)) for n in latent_args.param_names])

        # Gaussian tracking and optimization state
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        
        # Attribute masks for selective updates
        self.mask_xyz = torch.empty(0)
        self.mask_features_dc = torch.empty(0)
        self.mask_features_rest = torch.empty(0)
        self.mask_scaling = torch.empty(0)
        self.mask_rotation = torch.empty(0)
        self.mask_opacity = torch.empty(0)
        self.mask_flow = torch.empty(0)
        self.init_probs = None
        self.added_mask = None

        # Random number generator for splitting operations
        self.split_generator = torch.Generator(device="cuda")
        self.split_generator.manual_seed(latent_args.seed)

        self.param_names = latent_args.param_names
        self.mapping = None

        # Previous frame attributes
        self.prev_atts = OrderedDict({param_name:None for param_name in self.param_names})
        self.prev_latents = OrderedDict({param_name:None for param_name in self.param_names})
        self.prev_atts_initial = OrderedDict({param_name:None for param_name in self.param_names})

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.frame_idx = frame_idx

        # Freeze states for different attributes
        self.frz_xyz = "none"
        self.frz_features_dc =  "none"
        self.frz_features_rest =  "none"
        self.frz_scaling =  "none"
        self.frz_rotation =  "none"
        self.frz_opacity =  "none"
        self.frz_flow = "none"
        self.gate_atts = None
        self.latent_args = latent_args
        self.model_args = model_args
        self.setup_functions()
        self.setup_decoders(latent_args)

    def setup_decoders(self, latent_args: QuantizeParams, verbose=False):
        """Initialize latent decoders for each Gaussian attribute based on quantization settings."""
        self.feature_dims = OrderedDict([
            ("xyz", 3),
            ("f_dc", 3),
            ("f_rest", 3 * ((self.max_sh_degree + 1) ** 2 - 1)),
            ("sc", 3),
            ("rot", 4),
            ("op", 1),
            ("flow", 3)
        ])
        self.latent_decoders = OrderedDict()
        for i, param_name in enumerate(self.param_names):
            self.latent_decoders[param_name] = DecoderIdentity()
            if latent_args.quant_type[i] == 'sq':
                self.latent_decoders[param_name] = LatentDecoder(
                    latent_dim=latent_args.latent_dim[i],
                    feature_dim=self.feature_dims[param_name],
                    ldecode_matrix=latent_args.ldecode_matrix[i],
                    latent_norm=latent_args.latent_norm[i],
                    num_layers_dec=latent_args.num_layers_dec[i],
                    hidden_dim_dec=latent_args.hidden_dim_dec[i],
                    activation=latent_args.activation[i],
                    use_shift=latent_args.use_shift[i],
                    ldec_std=latent_args.ldec_std[i],
                    final_activation=latent_args.final_activation[i],
                ).cuda()
            if verbose:
                print(f"GaussianModel: Created {latent_args.quant_type[i]} decoder for {param_name}")

    def capture(self):
        return (
            self.active_sh_degree,
            self._latents,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.infl_accum,
            self.denom,
            self.infl_denom,
            OrderedDict([(n, l.state_dict()) for n,l in self.latent_decoders.items()]),
            self.gate_atts.state_dict() if self.gate_atts is not None else None,
            self.prev_atts,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.frame_idx,
            self.get_masks,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._latents,
        self.max_radii2D, 
        xyz_gradient_accum, 
        infl_accum, 
        denom,
        ldec_dicts,
        gate_att_dict,
        prev_atts,
        opt_dict, 
        self.spatial_lr_scale, 
        self.frame_idx,
        mask_atts) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.infl_accum = infl_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        for n in ldec_dicts:
            self.latent_decoders[n].load_state_dict(ldec_dicts[n])

        self.mask_xyz.data = mask_atts["xyz"]
        self.mask_features_dc.data = mask_atts["f_dc"]
        self.mask_features_rest.data = mask_atts["f_rest"]
        self.mask_scaling.data = mask_atts["sc"]
        self.mask_rotation.data = mask_atts["rot"]
        self.mask_opacity.data = mask_atts["op"]
        self.mask_flow.data = mask_atts["flow"]
        if self.gate_atts is not None:
            self.gate_atts.load_state_dict(gate_att_dict)
        self.prev_atts = prev_atts

    def restore_fps(self, model_args, training_args, start_frame_idx):
        (self.active_sh_degree, 
        self._latents,
        self.max_radii2D, 
        xyz_gradient_accum, 
        infl_accum, 
        denom,
        infl_denom,
        ldec_dicts,
        gate_att_dict,
        prev_atts,
        opt_dict, 
        self.spatial_lr_scale, 
        self.frame_idx,
        mask_atts) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.infl_accum = infl_accum
        self.denom = denom
        if start_frame_idx>1:
            self.frame_idx = 2
            self.update_residuals()
        for n in ldec_dicts:
            self.latent_decoders[n].load_state_dict(ldec_dicts[n])

        self.mask_xyz.data = mask_atts["xyz"]
        self.mask_features_dc.data = mask_atts["f_dc"]
        self.mask_features_rest.data = mask_atts["f_rest"]
        self.mask_scaling.data = mask_atts["sc"]
        self.mask_rotation.data = mask_atts["rot"]
        self.mask_opacity.data = mask_atts["op"]
        self.mask_flow.data = mask_atts["flow"]
        if self.gate_atts is not None:
            self.gate_atts.load_state_dict(gate_att_dict)
        self.prev_atts = prev_atts
    
    @property
    def get_atts(self):
        return self._latents
    
    @property
    def get_decoded_atts(self):
        return OrderedDict({"xyz"     : self._xyz,
                            "f_dc"    : self._features_dc,
                            "f_rest"  : self._features_rest,
                            "sc"      : self._scaling,
                            "rot"     : self._rotation,
                            "op"      : self._opacity,
                            "flow"    : self._flow})
    
    @property
    def get_masks(self):
        return OrderedDict({"xyz"     : self.mask_xyz, 
                            "f_dc"    : self.mask_features_dc, 
                            "f_rest"  : self.mask_features_rest, 
                            "sc"      : self.mask_scaling, 
                            "rot"     : self.mask_rotation,
                            "op"      : self.mask_opacity,
                            "flow"    : self.mask_flow})
    
    @property
    def get_frz(self):
        return OrderedDict({"xyz"     : self.frz_xyz, 
                            "f_dc"    : self.frz_features_dc, 
                            "f_rest"  : self.frz_features_rest, 
                            "sc"      : self.frz_scaling, 
                            "rot"     : self.frz_rotation,
                            "op"      : self.frz_opacity,
                            "flow"    : self.frz_flow})
    @property
    def mask_all(self):
        mask_all = torch.cat((self.mask_xyz, self.mask_features_dc, self.mask_features_rest,
                              self.mask_scaling, self.mask_rotation, self.mask_opacity, self.mask_flow),dim=1)
        return mask_all
    
    @property
    def _xyz(self):
        """Get xyz coordinates with optional gating and temporal consistency."""

        if self.use_xyz_legacy:
            return self._xyz_legacy()
        else:
            return self._xyz_fixed()
    
    def _xyz_legacy(self):
        # Decode latents to get xyz attribute
        xyz = self.latent_decoders["xyz"](self._latents["xyz"])
        
        # Apply gating if previous frame attributes exist and gating is enabled
        if self.prev_atts["xyz"] is not None and self.gate_atts is not None and self.gate_params["xyz"]:
            xyz = self.gate_atts(xyz-self.prev_atts["xyz"])+self.prev_atts["xyz"]
        return xyz

    def _xyz_fixed(self):
        # Decode latents to get xyz attribute
        xyz = self.latent_decoders["xyz"](self._latents["xyz"])
        
        # Apply gating if previous frame attributes exist and gating is enabled
        if self.prev_atts["xyz"] is not None and self.gate_atts is not None and self.gate_params["xyz"]:
            # Use gated residual from previous frame
            try:
                xyz = self.gate_atts(xyz-self.xyz_before[self.mapping])+self.xyz_before[self.mapping]
            except:
                # Handle mapping errors gracefully
                print(f"Warning: xyz mapping error - xyz shape: {xyz.shape}, "
                      f"prev_atts shape: {self.prev_atts['xyz'].shape}, "
                      f"xyz_before shape: {self.xyz_before.shape}, "
                      f"mapping max: {self.mapping.max()}")
                raise
        return xyz


    @property
    def _ungated_xyz_res(self):
        """Get ungated xyz residual for regularization."""
        xyz = self.latent_decoders["xyz"](self._latents["xyz"])
        return xyz-self.prev_atts["xyz"]
    
    @property
    def _features_dc(self):
        """Get DC (0th order) spherical harmonics features with optional gating."""
        if isinstance(self.latent_decoders["f_dc"], DecoderIdentity):
            features_dc = self._latents["f_dc"]
        else:
            features_dc = self.latent_decoders["f_dc"](self._latents["f_dc"])
            features_dc = features_dc.reshape(features_dc.shape[0], 1, 3)
        
        # Apply gating if enabled
        if self.prev_atts["f_dc"] is not None and self.gate_atts is not None and self.gate_params["f_dc"]:
            features_dc = self.gate_atts(features_dc-self.prev_atts["f_dc"])+self.prev_atts["f_dc"]
        return features_dc  # shape (N, 1, 3)
    
    @property
    def _features_rest(self):
        """Get higher-order spherical harmonics features with optional gating."""
        if isinstance(self.latent_decoders["f_rest"], DecoderIdentity):
            features_rest = self._latents["f_rest"]
        else:
            features_rest = self.latent_decoders["f_rest"](self._latents["f_rest"])
            features_rest = features_rest.reshape(features_rest.shape[0], (self.max_sh_degree + 1) ** 2 - 1, 3)
        
        # Apply gating if enabled
        if self.prev_atts["f_rest"] is not None and self.gate_atts is not None and self.gate_params["f_rest"]:
            features_rest = self.gate_atts(features_rest-self.prev_atts["f_rest"])+self.prev_atts["f_rest"]
        return features_rest  # shape (N, C-1, 3)
    
    @property
    def _scaling(self):
        """Get scaling parameters with optional gating."""
        scaling = self.latent_decoders["sc"](self._latents["sc"])
        if self.prev_atts["sc"] is not None and self.gate_atts is not None and self.gate_params["sc"]:
            scaling = self.gate_atts(scaling-self.prev_atts["sc"])+self.prev_atts["sc"]
        return scaling
    
    @property
    def _rotation(self):
        """Get rotation quaternions with optional gating."""
        rot = self.latent_decoders["rot"](self._latents["rot"])
        if self.prev_atts["rot"] is not None and self.gate_atts is not None and self.gate_params["rot"]:
            rot = self.gate_atts(rot-self.prev_atts["rot"])+self.prev_atts["rot"]
        return rot
    
    @property
    def _opacity(self):
        """Get opacity values with optional gating."""
        op = self.latent_decoders["op"](self._latents["op"])
        if self.prev_atts["op"] is not None and self.gate_atts is not None and self.gate_params["op"]:
            op = self.gate_atts(op-self.prev_atts["op"])+self.prev_atts["op"]
        return op
    
    @property
    def _flow(self):
        """Get optical flow vectors with optional gating."""
        flow = self.latent_decoders["flow"](self._latents["flow"])
        if self.prev_atts["flow"] is not None and self.gate_atts is not None and self.gate_params["flow"]:
            flow = self.gate_atts(flow-self.prev_atts["flow"])+self.prev_atts["flow"]
        return flow
    
    @property
    def mask_cov(self):
        return torch.logical_or(self.mask_scaling, self.mask_rotation)
    
    @property
    def mask_color(self):
        return torch.logical_or(self.mask_features_dc, self.mask_features_rest)
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_flow(self):
        return self._flow
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    
    def parameters(self):
        return list(self._latents.values()) + \
                [param for decoder in self.latent_decoders.values() for param in list(decoder.parameters())]
    
    def named_parameters(self):
        parameter_dict = self._latents
        for n, decoder in self.latent_decoders.items():
            parameter_dict.update(
                {n+'.'+param_name:param for param_name, param in dict(decoder.named_parameters()).items()}
                )
        return parameter_dict
    
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def norm_decoders(self, att_name=None):
        for param_name in self.param_names:
            if att_name is not None and param_name!=att_name:
                continue
            decoder = self.latent_decoders[param_name]
            if not isinstance(decoder, DecoderIdentity) and decoder.norm!="none":
                decoder.normalize(self._latents[param_name])
                
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, ignore_colors: bool = False):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        if ignore_colors:
            fused_color = RGB2SH(torch.tensor(0.5 * np.ones_like(pcd.points)).float().cuda())
        else:
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        scales = torch.clamp(scales, -10, 4)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        flow = torch.zeros_like(fused_point_cloud)
        
        ###################################### Init latents ##########################################
        self._latents = OrderedDict([(n,None) for n in self.param_names])

        init = self.latent_decoders["xyz"].invert(fused_point_cloud)
        self._latents["xyz"] = nn.Parameter(init.requires_grad_(True))

        init = features[:,:,0:1].transpose(1, 2).contiguous()
        if self.latent_args.f_dc_invert_type == "autoenc" and not isinstance(self.latent_decoders["f_dc"], DecoderIdentity):
            init, decoder_state_dict = init_latents(self.latent_args, fused_color.flatten(1).cuda(),"f_dc", lambda_distortion=0.0)
            self.latent_decoders["f_dc"].load_state_dict(decoder_state_dict)
        elif not isinstance(self.latent_decoders["f_dc"], DecoderIdentity):
            init = self.latent_decoders["f_dc"].invert(fused_color.flatten(start_dim=1).contiguous().cuda())
        self._latents["f_dc"] = nn.Parameter(init.requires_grad_(True))

        init = features[:,:,1:].transpose(1, 2).contiguous()
        if isinstance(self.latent_decoders["f_rest"], LatentDecoder):
            init = torch.zeros((features.size(0),self.latent_decoders["f_rest"].latent_dim)).to(init).contiguous()
        self._latents["f_rest"] = nn.Parameter(init.requires_grad_(True))

        if self.latent_args.sc_invert_type == "autoenc" and not isinstance(self.latent_decoders["sc"], DecoderIdentity):
            init, decoder_state_dict = init_latents(self.latent_args, scales,"sc", lambda_distortion=0.0)
            self.latent_decoders["sc"].load_state_dict(decoder_state_dict)
        else:
            init = self.latent_decoders["sc"].invert(scales)
        self._latents["sc"] = nn.Parameter(init.requires_grad_(True))

        if self.latent_args.rot_invert_type == "autoenc" and not isinstance(self.latent_decoders["rot"], DecoderIdentity):
            init, decoder_state_dict = init_latents(self.latent_args, rots,"rot", lambda_distortion=0.0)
            self.latent_decoders["rot"].load_state_dict(decoder_state_dict)
        else:
            init = self.latent_decoders["rot"].invert(rots)
        self._latents["rot"] = nn.Parameter(init.requires_grad_(True))

        if self.latent_args.op_invert_type == "autoenc" and not isinstance(self.latent_decoders["op"], DecoderIdentity):
            init, decoder_state_dict = init_latents(self.latent_args, opacities,"op", lambda_distortion=0.0)
            self.latent_decoders["op"].load_state_dict(decoder_state_dict)
        else:
            init = self.latent_decoders["op"].invert(opacities)
        self._latents["op"] = nn.Parameter(init.requires_grad_(True))

        if not isinstance(self.latent_decoders["flow"], DecoderIdentity):
            self._latents["flow"] = nn.Parameter(torch.zeros(flow.shape[0],self.latent_decoders["flow"].latent_dim), 
                                                 requires_grad=True, device=flow.device)
        else:
            self._latents["flow"] = nn.Parameter(flow.requires_grad_(True))

        ##########################################################################################

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.mask_xyz.data = torch.ones_like(self._opacity).bool()
        self.mask_features_dc.data = torch.ones_like(self._opacity).bool()
        self.mask_features_rest.data= torch.ones_like(self._opacity).bool()
        self.mask_scaling.data = torch.ones_like(self._opacity).bool()
        self.mask_rotation.data = torch.ones_like(self._opacity).bool()
        self.mask_opacity.data = torch.ones_like(self._opacity).bool()
        self.mask_flow.data = torch.ones_like(self._opacity).bool()


    def create_from_depth(self, cameras, spatial_lr_scale, downsample_scale=2, alpha_thresh=0.1, renderFunc=None):
        self.spatial_lr_scale = spatial_lr_scale

        _,orig_H,orig_W = cameras[0].original_image.shape
        downsample_size = (int(orig_H/downsample_scale),int(orig_W/downsample_scale))
        H,W = downsample_size
        xyz = coords_grid(1,H,W, device='cuda')[0].permute(1,2,0).view(-1,2)
        xyz = torch.cat((xyz,torch.zeros_like(xyz[:,0:1]),torch.ones_like(xyz[:,0:1])),dim=-1) # N x 4
        xyz[:,0] = xyz[:,0]/(0.5*W)+(1/W-1)
        xyz[:,1] = xyz[:,1]/(0.5*H)+(1/H-1)
        fused_point_cloud, fused_color = None, None
        world_xyz_colmap = torch.cat((self._xyz,torch.ones_like(self._xyz[:,0:1])),dim=1)
        net_points = 0
        for idx in range(len(cameras)):
            camera = cameras[idx]

            #################################################### Colmap coords ####################################################
            
            with torch.no_grad():
                world_xyz_colmap = torch.cat((self._xyz,torch.ones_like(self._xyz[:,0:1])),dim=1)
                cam_xyz_colmap = torch.matmul(camera.world_view_transform.T.unsqueeze(0),world_xyz_colmap.unsqueeze(-1)).squeeze(-1)
                cam_hom_colmap = torch.matmul(camera.projection_matrix.T.unsqueeze(0),cam_xyz_colmap.unsqueeze(-1)).squeeze(-1)
                cam_proj_colmap = cam_hom_colmap[:,:3]/cam_hom_colmap[:,3:]

                in_frustum = torch.logical_and(torch.all(cam_proj_colmap<1,dim=1),torch.all(cam_proj_colmap>-1,dim=1))
                in_frustum_depth = torch.logical_and(cam_proj_colmap[:,2]>0,cam_proj_colmap[:,2]<=1.0)

                cam_proj_colmap_filtered = cam_proj_colmap[in_frustum*in_frustum_depth]
                cam_xyz_colmap_filtered = cam_xyz_colmap[in_frustum*in_frustum_depth]
                world_xyz_colmap_filtered = world_xyz_colmap[in_frustum*in_frustum_depth]


            ######################################### Scaling values for inverse depth #############################################

            with torch.no_grad():
                image, depth = camera.original_image, camera.gt_depth # Actually inverse_depth for midas

                # Grid sample expects a  N x H x W x 2 grid for some reason. Create an array with first N values populated and reshape
                grid = torch.zeros(1,H,W,2).reshape(-1,2)
                grid[:cam_proj_colmap_filtered.shape[0]] = cam_proj_colmap_filtered[:,:2]

                # Sample from network produced inverse depth at those coordinates
                img = F.grid_sample(depth.unsqueeze(0).unsqueeze(0), grid.to(depth).reshape(1,H,W,2), mode='bilinear', padding_mode='zeros', align_corners=True)
                # Use only the coordinates at populated grid locations
                inverse_depths = img.reshape(-1,1)[:cam_proj_colmap_filtered.shape[0]]

                new_img = torch.zeros_like(depth)
                img_coords = torch.floor((grid[:cam_proj_colmap_filtered.shape[0]]+1)*0.5*torch.tensor([W,H]).to(grid).unsqueeze(0)).long()
                new_img[img_coords[:,1],img_coords[:,0]] = inverse_depths.flatten()

                B = cam_proj_colmap_filtered[:,2] # Get the z projected coordinates from the colmap as the GT depth
                B = (1 - B*(camera.zfar-camera.znear)/camera.zfar)/camera.znear
                A = torch.cat((inverse_depths,torch.ones_like(inverse_depths)),dim=1)
                out = torch.linalg.lstsq(A,B)
                
                pred_depth = 1/(A[:,0]*out.solution[0] + out.solution[1])
                gt_depth = 1/B

            ###################################### Subsample and index locations for depth #############################################

            with torch.no_grad():
                Zc = 1/(depth*out.solution[0]+out.solution[1]).view(-1,1)
                Xc = Zc*math.tan((camera.FoVx/2))*xyz[:,0:1]
                Yc = Zc*math.tan((camera.FoVy/2))*xyz[:,1:2]
                cam_xyz = torch.cat((Xc, Yc, Zc, torch.ones_like(Xc)),dim=1)
                world_xyz = torch.matmul(torch.linalg.inv(camera.world_view_transform.T).unsqueeze(0),cam_xyz.unsqueeze(-1)).squeeze(-1)[:,:3]

                cam_hom= torch.matmul(camera.projection_matrix.T.unsqueeze(0),cam_xyz.unsqueeze(-1)).squeeze(-1)
                depth = cam_hom[:,2]/cam_hom[:,3]
                camera.gt_depth = depth.reshape(orig_H, orig_W)

                render_pkg = renderFunc(viewpoint_camera=camera,pc=self)
                _, alpha = render_pkg["depth"], render_pkg["alpha"]
                alpha = F.interpolate(alpha.unsqueeze(0), size=downsample_size, mode='bilinear',
                                        align_corners=True)[0]
                alpha_mask = (alpha<alpha_thresh).reshape(-1) # 1 implies we add points from our GT depth, 0 implies neighborhood points already exist from colmap
                if alpha_mask.sum()==0:
                    continue
                assert alpha_mask.sum()!=alpha_mask.numel()

            # New points to add in empty regions
            num_new = alpha_mask.sum()/(~alpha_mask).numel()*((in_frustum*in_frustum_depth).sum())
            if num_new == 0:
                continue

            world_xyz_new = world_xyz[alpha_mask] # Only points corresponding to new areas without colmap init

            indices = torch.randperm(world_xyz_new.shape[0])[:num_new.long()]
            world_xyz_subsampled = world_xyz_new[indices]
            net_points += world_xyz_subsampled.shape[0]

            ################################################### Color #########################################################

            image_new = image.permute(1,2,0).view(-1,3)[alpha_mask,:] # Get corresponding pixels at the new areas
            image_subsampled = image_new[indices]

            color = RGB2SH(image_subsampled)
            features = torch.zeros((color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0 ] = color
            features[:, 3:, 1:] = 0.0

            if fused_point_cloud is None :
                fused_point_cloud = world_xyz_subsampled
                fused_color = features
            else:
                fused_point_cloud = torch.cat((fused_point_cloud,world_xyz_subsampled))
                fused_color = torch.cat((fused_color,features))

        if fused_color is None:
            return
            
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        new_xyz = self.latent_decoders["xyz"].invert(fused_point_cloud)
        if isinstance(self.latent_decoders["f_dc"],DecoderIdentity):
            new_features_dc = fused_color[:,:,0:1].transpose(1, 2).contiguous()
        else:
            new_features_dc = self.latent_decoders["f_dc"].invert(fused_color[:,:,0].contiguous().cuda())
        if isinstance(self.latent_decoders["f_rest"],DecoderIdentity):
            new_features_rest = fused_color[:,:,1:].transpose(1, 2).contiguous()
        else:
            new_features_rest = torch.zeros((fused_color.size(0),self.latent_decoders["f_rest"].latent_dim)).to(fused_color).contiguous()

        new_scaling = self.latent_decoders["sc"].invert(scales)
        new_rotation = self.latent_decoders["rot"].invert(rots)
        new_opacity = self.latent_decoders["op"].invert(opacities)
        new_flow = torch.zeros_like(fused_point_cloud)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_flow)

        print("\nAdded {} points!".format(net_points))

    def create_from_depth_immersive(self, cameras, spatial_lr_scale, downsample_scale=2, alpha_thresh=0.1, renderFunc=None):
        self.spatial_lr_scale = spatial_lr_scale

        _,orig_H,orig_W = cameras[0].original_image.shape
        downsample_size = (int(orig_H/downsample_scale),int(orig_W/downsample_scale))
        H,W = downsample_size
        xyz = coords_grid(1,H,W, device='cuda')[0].permute(1,2,0).view(-1,2)
        xyz = torch.cat((xyz,torch.zeros_like(xyz[:,0:1]),torch.ones_like(xyz[:,0:1])),dim=-1) # N x 4
        xyz[:,0] = xyz[:,0]/(0.5*W)+(1/W-1)
        xyz[:,1] = xyz[:,1]/(0.5*H)+(1/H-1)
        fused_point_cloud, fused_color = None, None
        world_xyz_colmap = torch.cat((self._xyz,torch.ones_like(self._xyz[:,0:1])),dim=1)
        net_points = 0
        for idx in range(len(cameras)):
            camera = cameras[idx]

            #################################################### Colmap coords ####################################################
            
            with torch.no_grad():
                world_xyz_colmap = torch.cat((self._xyz,torch.ones_like(self._xyz[:,0:1])),dim=1)
                cam_xyz_colmap = torch.matmul(camera.world_view_transform.T.unsqueeze(0),world_xyz_colmap.unsqueeze(-1)).squeeze(-1)
                cam_hom_colmap = torch.matmul(camera.projection_matrix.T.unsqueeze(0),cam_xyz_colmap.unsqueeze(-1)).squeeze(-1)
                cam_proj_colmap = cam_hom_colmap[:,:3]/cam_hom_colmap[:,3:]

                in_frustum = torch.logical_and(torch.all(cam_proj_colmap<1,dim=1),torch.all(cam_proj_colmap>-1,dim=1))
                in_frustum_depth = cam_proj_colmap[:,2]>0

                cam_proj_colmap_filtered = cam_proj_colmap[in_frustum*in_frustum_depth]
                cam_xyz_colmap_filtered = cam_xyz_colmap[in_frustum*in_frustum_depth]
                world_xyz_colmap_filtered = world_xyz_colmap[in_frustum*in_frustum_depth]

            # X, Y, Z = cam_xyz_colmap_filtered[:,0], cam_xyz_colmap_filtered[:,1], cam_xyz_colmap_filtered[:,2]
            # xp, yp, zp, wp = 1/math.tan((camera.FoVx/2))*X/Z, 1/math.tan((camera.FoVx/2))*Y/Z, camera.zfar/(camera.zfar-camera.znear)*(1-camera.znear/Z), Z

            ######################################### Scaling values for inverse depth #############################################

            with torch.no_grad():
                image, depth = camera.original_image, camera.gt_depth # Actually inverse_depth for midas

                # Grid sample expects a  N x H x W x 2 grid for some reason. Create an array with first N values populated and reshape
                grid = torch.zeros(1,H,W,2).reshape(-1,2)
                grid[:cam_proj_colmap_filtered.shape[0]] = cam_proj_colmap_filtered[:,:2]

                # Sample from network produced inverse depth at those coordinates
                img = F.grid_sample(depth.unsqueeze(0).unsqueeze(0), grid.to(depth).reshape(1,H,W,2), mode='bilinear', padding_mode='zeros', align_corners=True)
                # Use only the coordinates at populated grid locations
                inverse_depths = img.reshape(-1,1)[:cam_proj_colmap_filtered.shape[0]]

                orig_depths = cam_xyz_colmap_filtered[:,2]
                inverse_gt_depths = 1/(orig_depths+camera.znear)
                out = torch.linalg.lstsq(torch.cat((inverse_depths,torch.ones_like(inverse_depths)),dim=1), inverse_gt_depths)

            ###################################### Subsample and index locations for depth #############################################

            with torch.no_grad():
                Zc = 1/(depth*out.solution[0]+out.solution[1]).view(-1,1)
                Xc = Zc*math.tan((camera.FoVx/2))*xyz[:,0:1]
                Yc = Zc*math.tan((camera.FoVy/2))*xyz[:,1:2]
                cam_xyz = torch.cat((Xc, Yc, Zc, torch.ones_like(Xc)),dim=1)
                world_xyz = torch.matmul(torch.linalg.inv(camera.world_view_transform.T).unsqueeze(0),cam_xyz.unsqueeze(-1)).squeeze(-1)[:,:3]

                cam_hom= torch.matmul(camera.projection_matrix.T.unsqueeze(0),cam_xyz.unsqueeze(-1)).squeeze(-1)
                depth = cam_hom[:,2]/cam_hom[:,3]
                camera.gt_depth = depth.reshape(orig_H, orig_W)

                render_pkg = renderFunc(viewpoint_camera=camera,pc=self)
                _, alpha = render_pkg["depth"], render_pkg["alpha"]
                alpha = F.interpolate(alpha.unsqueeze(0), size=downsample_size, mode='bilinear',
                                        align_corners=True)[0]
                alpha_mask = (alpha<alpha_thresh).reshape(-1) # 1 implies we add points from our GT depth, 0 implies neighborhood points already exist from colmap
                if alpha_mask.sum()==0:
                    continue
                assert alpha_mask.sum()!=alpha_mask.numel()

            # New points to add in empty regions
            num_new = alpha_mask.sum()/(~alpha_mask).numel()*((in_frustum*in_frustum_depth).sum())
            if num_new == 0:
                continue

            world_xyz_new = world_xyz[alpha_mask] # Only points corresponding to new areas without colmap init

            indices = torch.randperm(world_xyz_new.shape[0])[:num_new.long()]
            world_xyz_subsampled = world_xyz_new[indices]
            net_points += world_xyz_subsampled.shape[0]

            ################################################### Color #########################################################

            image_new = image.permute(1,2,0).view(-1,3)[alpha_mask,:] # Get corresponding pixels at the new areas
            image_subsampled = image_new[indices]

            color = RGB2SH(image_subsampled)
            features = torch.zeros((color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0 ] = color
            features[:, 3:, 1:] = 0.0

            if fused_point_cloud is None :
                fused_point_cloud = world_xyz_subsampled
                fused_color = features
            else:
                fused_point_cloud = torch.cat((fused_point_cloud,world_xyz_subsampled))
                fused_color = torch.cat((fused_color,features))

        if fused_color is None:
            return
            
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        new_xyz = self.latent_decoders["xyz"].invert(fused_point_cloud)
        if isinstance(self.latent_decoders["f_dc"],DecoderIdentity):
            new_features_dc = fused_color[:,:,0:1].transpose(1, 2).contiguous()
        else:
            new_features_dc = self.latent_decoders["f_dc"].invert(fused_color[:,:,0].contiguous().cuda())
        if isinstance(self.latent_decoders["f_rest"],DecoderIdentity):
            new_features_rest = fused_color[:,:,1:].transpose(1, 2).contiguous()
        else:
            new_features_rest = torch.zeros((fused_color.size(0),self.latent_decoders["f_rest"].latent_dim)).to(fused_color).contiguous()

        new_scaling = self.latent_decoders["sc"].invert(scales)
        new_rotation = self.latent_decoders["rot"].invert(rots)
        new_opacity = self.latent_decoders["op"].invert(opacities)
        new_flow = torch.zeros_like(fused_point_cloud)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_flow)


        print("\nAdded {} points!".format(net_points))

    @torch.no_grad()
    def update_points_flow(self):
        if type(self.latent_decoders["xyz"]) == LatentDecoderRes:
            self._latents["xyz"].data = self.latent_decoders["xyz"].invert(self._flow)*self.mask_xyz
        elif isinstance(self.latent_decoders["xyz"], DecoderIdentity) \
            or isinstance(self.latent_decoders["xyz"], LatentDecoder):
            if self.gate_atts is not None:
                new_xyz = self._xyz + self._flow*self.mask_xyz*self.gate_atts.gate.unsqueeze(-1)
            else:
                new_xyz = self._xyz+self._flow*self.mask_xyz
            self._latents["xyz"].data = self.latent_decoders["xyz"].invert(new_xyz)

        self._latents["flow"] *= 0

    def size(self):
        """Calculate compressed model size in bits for storage estimation."""
        with torch.no_grad():
            latents_size = ldec_size = 0
            frz, masks = self.get_frz, self.get_masks
            
            for param_name in self.param_names:
                mask = masks[param_name].flatten()
                if param_name == "flow":
                    continue
                    
                # Add decoder size
                ldec_size += self.latent_decoders[param_name].size()
                decoder = self.latent_decoders[param_name]
                
                # Calculate latent storage size based on decoder type
                if isinstance(decoder, DecoderIdentity) \
                    or (type(decoder)==LatentDecoderRes and decoder.identity):
                    p = self._latents[param_name]
                    mask_frac = 1.0
                    
                    # Apply masking based on freeze state
                    if frz[param_name] == "st":
                        mask_frac = mask.sum().item()/mask.numel()
                    elif frz[param_name] == "all":
                        mask_frac = 0.0
                        
                    # Apply gating compression if enabled
                    if self.gate_atts is not None and self.gate_params[param_name]:
                        assert frz[param_name] == "none"
                        gate = self.gate_atts.sample_gate(stochastic=False)
                        mask_frac = (gate!=0.0).sum().item()/gate.numel()

                    latents_size += p.numel()*torch.finfo(p.dtype).bits*mask_frac
                else:
                    if frz[param_name] == "all":
                        continue
                        
                    # Determine previous attribute for compression
                    if type(self.latent_decoders[param_name])==LatentDecoder:
                        if self.frame_idx == 1:
                            prev_att = None
                        else:
                            prev_att = torch.round(self.prev_latents[param_name])
                    elif type(self.latent_decoders[param_name])==LatentDecoderRes:
                        prev_att = None
                    else:
                        raise Exception(f"Unknown {param_name} decoder {type(self.latent_decoders[param_name])}")
                        
                    # Calculate entropy-based compression size
                    for dim in range(self._latents[param_name].size(1)):
                        if prev_att is not None:
                            weight = (torch.round(self._latents[param_name][:,dim])-prev_att[:,dim]).long()
                        else:
                            weight = torch.round(self._latents[param_name][:,dim]).long()
                        if frz[param_name] == "st":
                            weight = weight[mask]
                        
                        unique_vals, counts = torch.unique(weight, return_counts = True)
                        probs = counts/torch.sum(counts)

                        information_bits = torch.clamp(-1.0 * torch.log(probs + 1e-10) / np.log(2.0), 0, 1000)
                        size_bits = torch.sum(information_bits*counts).item()
                        latents_size += size_bits
                        
            # Add gate size if present
            if self.gate_atts is not None:
                latents_size += self.gate_atts.size()
                
        return ldec_size+latents_size
    
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.infl_accum = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.infl_denom = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.added_mask = None

        self.lr_scaling = OrderedDict()
        self.gate_params = OrderedDict()
        for i,param in enumerate(self.param_names):
            decoder = self.latent_decoders[param]
            if type(decoder) == DecoderIdentity or (type(decoder)== LatentDecoderRes and decoder.identity):
                self.lr_scaling[param] = 1.0
            else:
                self.lr_scaling[param] = training_args.latents_lr_scaling[i]
            self.gate_params[param] = self.latent_args.gate_params[i]!="none"

        self.orig_lr = OrderedDict({"xyz":training_args.position_lr_init, 
                                    "f_dc":training_args.features_dc_lr, 
                                    "f_rest":training_args.features_rest_lr, 
                                     "sc":training_args.scaling_lr, 
                                     "rot":training_args.rotation_lr, 
                                     "op":training_args.opacity_lr, 
                                     "flow":training_args.flow_lr})
        lr = {
                'xyz':training_args.position_lr_init * self.spatial_lr_scale*self.lr_scaling["xyz"],
                'f_dc':training_args.features_dc_lr*self.lr_scaling["f_dc"],
                'f_rest':training_args.features_rest_lr*self.lr_scaling["f_rest"],
                'sc':training_args.scaling_lr*self.lr_scaling["sc"],
                'rot':training_args.rotation_lr*self.lr_scaling["rot"],
                'op':training_args.opacity_lr*self.lr_scaling["op"],
                'flow':training_args.flow_lr*self.lr_scaling["flow"]
            }
        l = []
        for i,param in enumerate(self.param_names):
            l += [{'params': [self._latents[param]], 'lr': lr[param], "name": param}]
            if not isinstance(self.latent_decoders[param], DecoderIdentity):
                l += [{'params': self.latent_decoders[param].parameters(), 'lr': training_args.ldecs_lr[i], "name":f"ldec_{param}"}]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_masks(self, args, dynamic_mask):
        frz_args = {"xyz":args.frz_xyz, "f_dc":args.frz_f_dc, "f_rest":args.frz_f_rest, 
                    "op":args.frz_op, "rot":args.frz_rot, "sc":args.frz_sc, "flow":args.frz_flow}
        masks = self.get_masks
        for param_name in frz_args:
            if frz_args[param_name] not in ["none","all"]:
                assert dynamic_mask is not None
                masks[param_name].data = dynamic_mask
            elif frz_args[param_name] == "all":
                masks[param_name].data *= False
            elif frz_args[param_name] == "none":
                masks[param_name].data += True

    def freeze_atts(self, args):
        self.frz_xyz = args.frz_xyz
        self.frz_features_dc = args.frz_f_dc
        self.frz_features_rest = args.frz_f_rest
        self.frz_scaling = args.frz_sc
        self.frz_rotation = args.frz_rot
        self.frz_opacity = args.frz_op
        self.frz_flow = args.frz_flow

        # 0 is static content 1 is dynamic content
        frz, atts, masks = self.get_frz, self.get_atts, self.get_masks
        for att_name in atts:
            if torch.any(~masks[att_name]) and frz[att_name]!="none":
                if "f_" in att_name:
                    mask = masks[att_name][...,None]
                else:
                    mask = masks[att_name]
                if frz[att_name] == "all":
                    atts[att_name].requires_grad_(False)
                    for param in self.latent_decoders[att_name].parameters():
                        param.requires_grad_(False)
                elif frz[att_name] == "st":
                    self.latent_decoders[att_name].freeze_partial(masks[att_name].flatten())
                    continue
                    if "f_" in att_name:
                        atts[att_name].register_post_accumulate_grad_hook(lambda grad: grad.mul_(mask.unsqueeze(-1)))
                    else:
                        atts[att_name].register_hook(lambda grad: grad*(mask))
                else:
                    raise Exception('Undefined mode ', frz[att_name])

    def std_reg(self):
        net_std = 0.0
        for att_name in self.get_atts:
            decoder = self.latent_decoders[att_name]
            if "xyz" not in att_name and "flow" not in att_name and \
                type(decoder)!=DecoderIdentity:
                net_std += self._latents[att_name].std(dim=0).mean()
        return net_std
    
    def update_residuals(self):
        """Initialize residual encoders for temporal compression in subsequent frames."""
        atts = self.get_atts # All the latent variables
        
        # Initialize parent mapping for tracking Gaussian relationships
        self.xyz_before = self.get_xyz.clone()
        self.mapping =  torch.arange(self.xyz_before.shape[0], device="cuda")
        
        for i, att_name in enumerate(atts):
            if self.latent_args.quant_type[i] == 'sq_res':
                decoded_att = self.latent_decoders[att_name](self._latents[att_name])
                
                if self.frame_idx == 2:
                    # Switch from identity to residual decoder after frame 1
                    assert isinstance(self.latent_decoders[att_name], DecoderIdentity)
                    decoder = LatentDecoderRes(
                        latent_dim=self.latent_args.latent_dim[i],
                        feature_dim=self.feature_dims[att_name],
                        ldecode_matrix=self.latent_args.ldecode_matrix[i],
                        latent_norm=self.latent_args.latent_norm[i],
                        num_layers_dec=self.latent_args.num_layers_dec[i],
                        hidden_dim_dec=self.latent_args.hidden_dim_dec[i],
                        activation=self.latent_args.activation[i],
                        use_shift=self.latent_args.use_shift[i],
                        ldec_std=self.latent_args.ldec_std[i],
                        final_activation=self.latent_args.final_activation[i],
                    ).cuda()
                    self.latent_decoders[att_name] = decoder
                else:
                    # Verify residual decoder type for subsequent frames
                    decoder = self.latent_decoders[att_name]
                    assert isinstance(self.latent_decoders[att_name], LatentDecoderRes)

                self.latent_decoders[att_name].frame_idx = self.frame_idx

                # Initialize decoder with previous frame's decoded attributes
                if "f_" in att_name and self.frame_idx == 2:
                    decoder.init_decoded(decoded_att.reshape(decoded_att.shape[0], -1))  # SH coefficients in (N, X)
                else:
                    decoder.init_decoded(decoded_att)

                # Set up new latent parameters for current frame
                if type(self.latent_decoders[att_name])== LatentDecoderRes \
                    and self.latent_args.quant_after[i]>0.0:
                    # Start in identity mode, switch to quantized later
                    self.latent_decoders[att_name].identity = True
                    decoder = self.latent_decoders[att_name]
                    latent = torch.zeros_like(decoder.decoded_att)
                    self._latents[att_name] = nn.Parameter(latent.requires_grad_(True))
                else:
                    # Initialize with zero residuals
                    self._latents[att_name] = nn.Parameter(
                                                torch.zeros((self._latents[att_name].shape[0],
                                                                self.latent_args.latent_dim[i]), 
                                                            dtype=torch.float, 
                                                            device="cuda").requires_grad_(True)
                                                )

    def update_grads(self):
        frz, atts, masks = self.get_frz, self.get_atts, self.get_masks
        for att_name in atts:
            if frz[att_name] == 'st':
                mask = masks[att_name] if frz[att_name] == 'st' else ~masks[att_name]
                if "f_" in att_name:
                    atts[att_name].grad *= mask.unsqueeze(-1)
                else:
                    atts[att_name].grad *= mask

    def update_learning_rate(self, iteration, latent_args: QuantizeParams):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr*self.lr_scaling["xyz"]
            elif param_group["name"] in self.param_names:
                idx = self.param_names.index(param_group["name"])
                if latent_args.latent_scale_norm[idx] == "div":
                    lr = self.orig_lr[param_group["name"]]*self.lr_scaling[param_group["name"]]
                    lr /= self.latent_decoders[param_group["name"]].scale_norm()
                    param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path, mask):
        mkdir_p(os.path.dirname(path))

        if mask is not None:
            xyz = self._xyz[mask].detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self._features_dc[mask].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._features_rest[mask].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self._opacity[mask].detach().cpu().numpy()
            scale = self._scaling[mask].detach().cpu().numpy()
            rotation = self._rotation[mask].detach().cpu().numpy()
        else:
            xyz = self._xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self._opacity.detach().cpu().numpy()
            scale = self._scaling.detach().cpu().numpy()
            rotation = self._rotation.detach().cpu().numpy()

        vertex_ids = np.arange(xyz.shape[0])
        dtype_full = [(attribute, 'f4') for attribute in ['x', 'y', 'z', 'nx', 'ny', 'nz']]
        dtype_full.extend([(attribute, 'f4') for attribute in 
                        [f'f_dc_{i}' for i in range(f_dc.shape[1])]])
        dtype_full.extend([(attribute, 'f4') for attribute in 
                        [f'f_rest_{i}' for i in range(f_rest.shape[1])]])
        dtype_full.extend([('opacity', 'f4')])
        dtype_full.extend([(attribute, 'f4') for attribute in 
                        [f'scale_{i}' for i in range(scale.shape[1])]])
        dtype_full.extend([(attribute, 'f4') for attribute in 
                        [f'rot_{i}' for i in range(rotation.shape[1])]])
        dtype_full.append(('vertex_id', 'i4'))  # Add the vertex_id field

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements['x'] = xyz[:, 0]
        elements['y'] = xyz[:, 1]
        elements['z'] = xyz[:, 2]
        elements['nx'] = normals[:, 0]
        elements['ny'] = normals[:, 1]
        elements['nz'] = normals[:, 2]
        for i in range(f_dc.shape[1]):
            elements[f'f_dc_{i}'] = f_dc[:, i]
        for i in range(f_rest.shape[1]):
            elements[f'f_rest_{i}'] = f_rest[:, i]
        elements['opacity'] = opacities[:, 0]  # Fix the shape here
        for i in range(scale.shape[1]):
            elements[f'scale_{i}'] = scale[:, i]
        for i in range(rotation.shape[1]):
            elements[f'rot_{i}'] = rotation[:, i]
        elements['vertex_id'] = vertex_ids

        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_compressed_pkl(self, path, latent_args):
        mkdir_p(os.path.dirname(path))

        latents = OrderedDict()
        decoder_state_dict = OrderedDict()
        decoder_args = OrderedDict()
                
        for i,attribute in enumerate(self.param_names):
            if isinstance(self.latent_decoders[attribute], DecoderIdentity):
                if self.prev_atts[attribute] is not None and self.gate_atts is not None and self.gate_params[attribute]:
                    xyz = self.latent_decoders["xyz"](self._latents["xyz"])
                    prev = self.xyz_before[self.mapping]
                    residual_xyz = xyz - prev
                    
                    # Get ungated indices and their residual values
                    ungated_xyz_indices = self.gate_atts.sample_gate(stochastic=False).nonzero(as_tuple=True)[0]
                    # compute the number of bits needed to store the max value in ungated_xyz_indices
                    max_value = ungated_xyz_indices.max()
                    num_bits = max_value.item().bit_length()
                    if num_bits < 8:
                        ungated_xyz_indices = ungated_xyz_indices.type(torch.int8)
                    elif num_bits < 16:
                        ungated_xyz_indices = ungated_xyz_indices.type(torch.short)
                    elif num_bits < 32:
                        ungated_xyz_indices = ungated_xyz_indices.type(torch.int)
                    else:
                        ungated_xyz_indices = ungated_xyz_indices.type(torch.long)
                    # Store the gated residuals, not raw residuals
                    ungated_residuals = self.gate_atts(residual_xyz)[ungated_xyz_indices]
                    compressed_residuals = CompressedLatents()
                    compressed_residuals.compress(ungated_residuals, scale=10000.0)

                    # Store minimal data needed for reconstruction
                    latents[attribute] = {
                        'mapping': self.mapping,
                        'ungated_indices': ungated_xyz_indices,
                        'ungated_residuals_compressed': compressed_residuals
                        # '_xyz_debug': self._xyz,
                        # 'xyz_before_debug': self.xyz_before,
                        # 'prev_att_debug': self.prev_atts[attribute]
                    }
                    
                    # Verify reconstruction matches _xyz
                    reconstructed_xyz = prev.clone()
                    reconstructed_xyz[ungated_xyz_indices] += ungated_residuals
                    # assert torch.allclose(self._xyz, reconstructed_xyz, rtol=1e-5, atol=1e-5), "Reconstruction verification failed!"
                else:
                    # For non-gated DecoderIdentity attributes, just store the latents directly
                    latents[attribute] = self._latents[attribute].detach().cpu()
            else:
                latent = self._latents[attribute].detach().cpu()
                compressed_obj = CompressedLatents()
                compressed_obj.compress(latent)
                latents[attribute] = compressed_obj
                decoder_args[attribute] = {
                    'latent_dim': latent_args.latent_dim[i],
                    'feature_dim': self.feature_dims[attribute],
                    'ldecode_matrix': latent_args.ldecode_matrix[i],
                    'latent_norm': latent_args.latent_norm[i],
                    'num_layers_dec': latent_args.num_layers_dec[i],
                    'hidden_dim_dec': latent_args.hidden_dim_dec[i],
                    'activation': latent_args.activation[i],
                    'use_shift': latent_args.use_shift[i],
                    'ldec_std': latent_args.ldec_std[i]
                }
                decoder_state_dict[attribute] = self.latent_decoders[attribute].state_dict().copy()

                # manually add the decoded_att to the state_dict
                if hasattr(self.latent_decoders[attribute], 'decoded_att'):
                    decoder_state_dict[attribute]['decoded_att'] = self.latent_decoders[attribute].decoded_att.detach().cpu()

        save_state = {
                         'latents': latents,
                         'decoder_state_dict': decoder_state_dict,
                         'decoder_args': decoder_args,
                         'latent_decoders_dict': {attr: type(self.latent_decoders[attr]).__name__ for attr in self.param_names}
            }

        with open(path,'wb') as f:
            pickle.dump(save_state, f)

    def load_compressed_pkl(self, path):
        with open(path,'rb') as f:
            data = pickle.load(f)
            latents = data['latents']
            decoder_state_dict = data['decoder_state_dict']
            decoder_args = data['decoder_args']
            latent_decoders_dict = data['latent_decoders_dict']
                        
        # First verify the number of gaussians matches
        num_gaussians = None
        for attribute in latents:
            if isinstance(latents[attribute], dict) and 'num_gaussians' in latents[attribute]:
                if num_gaussians is None:
                    num_gaussians = latents[attribute]['num_gaussians']
                elif num_gaussians != latents[attribute]['num_gaussians']:
                    raise ValueError(f"Inconsistent number of gaussians in compressed data: {num_gaussians} vs {latents[attribute]['num_gaussians']}")
        
        # Initialize gate if needed
        if num_gaussians is not None:
            if self.gate_atts is None:
                self.gate_atts = Gate(num_gaussians, 
                                    gamma=self.model_args.gate_gamma,
                                     eta=self.model_args.gate_eta,
                                     lr=self.model_args.gate_lr,
                                     temp=self.model_args.gate_temp).cuda()
        
        # Then proceed with loading attributes
        for i, attribute in enumerate(latents):
            if self.latent_args.gate_params[i] == 'on':
                # if the gate_params is on, that means we are loading gated attributes
                # Reconstruct prev_atts from sparse difference
                if self.prev_atts[attribute] is not None:
                    prev_att = self.prev_atts[attribute]
                else:
                    prev_att = torch.zeros([num_gaussians, 3], device="cuda")

                prev_att_mapping = latents[attribute]["mapping"].cuda()

                reconstructed = prev_att[prev_att_mapping].clone() 
                ungated_xyz_indices = latents[attribute]['ungated_indices']
                ungated_residuals = latents[attribute]['ungated_residuals_compressed'].uncompress(scale=10000.0).cuda()
                reconstructed[ungated_xyz_indices] += ungated_residuals
                
                # Verify reconstruction matches _xyz
                # original_xyz = latents[attribute]['_xyz_debug']
                # assert torch.allclose(original_xyz, reconstructed, rtol=1e-5, atol=1e-5), "Reconstruction verification failed!"
                
                self.mapping = prev_att_mapping
                
                self._latents[attribute] = nn.Parameter(reconstructed.requires_grad_(False))
            else:
                if self.prev_atts[attribute] is not None:
                    prev_att = self.prev_atts[attribute]
                else:
                    prev_att = torch.zeros(latents[attribute]["shape"], device="cuda")
                remapped_prev_att = prev_att[prev_att_mapping]  

                # then we define it based on the decoder type
                if latent_decoders_dict[attribute] == "LatentDecoder":
                    self.latent_decoders[attribute] = LatentDecoder(**decoder_args[attribute]).cuda()
                    self.latent_decoders[attribute].load_state_dict(decoder_state_dict[attribute])
                    self._latents[attribute] = nn.Parameter(latents[attribute].uncompress().cuda().requires_grad_(False))
                    
                elif latent_decoders_dict[attribute] == "DecoderIdentity":
                    # Identity decoder (no compression)
                    self.latent_decoders[attribute] = DecoderIdentity().cuda()
                    self._latents[attribute] = nn.Parameter(latents[attribute].cuda().requires_grad_(False))
                    
                elif latent_decoders_dict[attribute] == "LatentDecoderRes":
                    # Residual decoder with previous frame reference
                    self.latent_decoders[attribute] = LatentDecoderRes(**decoder_args[attribute]).cuda()
                    
                    # Extract and initialize with previous frame's decoded attributes
                    decoded_att = decoder_state_dict[attribute]['decoded_att'].clone().cuda()
                    del decoder_state_dict[attribute]['decoded_att']  # Remove from state dict before loading
                    
                    self.latent_decoders[attribute].load_state_dict(decoder_state_dict[attribute])
                    self.latent_decoders[attribute].init_decoded(decoded_att)
                    self.latent_decoders[attribute].identity = False  # Enable quantized mode
                    self.latent_decoders[attribute].frame_idx = self.frame_idx
                    self._latents[attribute] = nn.Parameter(latents[attribute].uncompress().cuda().requires_grad_(False))
                    
                    # Verify decoder functionality
                    test_output = self.latent_decoders[attribute](self._latents[attribute])
                    expected_dim = decoder_args[attribute]['feature_dim']
                    if test_output.shape[-1] != expected_dim:
                        print(f"Warning: {attribute} decoder output shape {test_output.shape} != expected {expected_dim}")
        
        self.active_sh_degree = self.max_sh_degree

    def decode_latents(self):
        with torch.no_grad():
            for param in self.param_names:
                if not isinstance(self.latent_decoders[param], DecoderIdentity):
                    decoded = self.latent_decoders[param](self._latents[param])
                    if param == "f_rest":
                        self._latents[param].data = decoded.reshape(self._latents[param].shape[0], 
                                                                                 (self.max_sh_degree + 1) ** 2 - 1, 3)
                    elif param == "f_dc":
                        self._latents[param].data = decoded.reshape(self._latents[param].shape[0], 1, 3)
                    else:
                        self._latents[param].data = decoded
                    self.latent_decoders[param] = DecoderIdentity()

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        if type(self.latent_decoders["op"]) == LatentDecoderRes:
            opacities_new = self.latent_decoders["op"].invert(opacities_new-self.latent_decoders["op"].decoded_att)
        else:
            opacities_new = self.latent_decoders["op"].invert(opacities_new)

        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "op")
        self._latents["op"] = optimizable_tensors["op"]

    def load_ply(self, path, verbose=False):
        plydata = PlyData.read(path)
        if verbose:
            print(f"GaussianModel::load_ply(): loaded gaussian from ply file at: {path}")

        # positions and opacities
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        # SH dc (0-freq) values
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # SH "ac" values
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        expected_extra_f_dim = 3 * (self.max_sh_degree + 1) ** 2 - 3
        if len(extra_f_names) == expected_extra_f_dim:
            if verbose:
                print(f"GaussianModel::load_ply(): parsed SH extra f dim matches expectation: {expected_extra_f_dim}")
        elif len(extra_f_names) > expected_extra_f_dim:
            if verbose:
                print(f"GaussianModel::load_ply(): parsed SH extra f dim ({len(extra_f_names)}) exceeds expectation ({expected_extra_f_dim}).")
        else:
            raise RuntimeError(f"GaussianModel::load_ply(): parsed SH extra f dim ({len(extra_f_names)}) does not reach expectation ({expected_extra_f_dim})")

        features_extra = np.zeros((xyz.shape[0], expected_extra_f_dim))
        for idx, attr_name in enumerate(extra_f_names):
            # if provided features (extra_f_names) have higher dim than what we need (expected_extra_f_dim), ignore them.
            if idx < expected_extra_f_dim:
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P, F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        # scales
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # rotations
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # initialize the attributes
        self._latents["xyz"] = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._latents["f_dc"] = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._latents["f_rest"] = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._latents["rot"] = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._latents["sc"] = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._latents["op"] = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))

        # extra stuff
        flow = torch.zeros_like(torch.from_numpy(xyz)).float().cuda()
        self._latents["flow"] = nn.Parameter(flow.clone().requires_grad_(True))

        for param in self.param_names:
            self.latent_decoders[param] = DecoderIdentity()  # by doing so, we initialize the Gaussian with no decoding.
            if hasattr(self,"gate_params"):
                self.gate_params[param] = False
        self.gate_atts = None

        self.active_sh_degree = self.max_sh_degree

        # extra stuff
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.mask_xyz.data = torch.ones_like(self._opacity).bool()
        self.mask_features_dc.data = torch.ones_like(self._opacity).bool()
        self.mask_features_rest.data= torch.ones_like(self._opacity).bool()
        self.mask_scaling.data = torch.ones_like(self._opacity).bool()
        self.mask_rotation.data = torch.ones_like(self._opacity).bool()
        self.mask_opacity.data = torch.ones_like(self._opacity).bool()
        self.mask_flow.data = torch.ones_like(self._opacity).bool()
        if verbose:
            print(f"GaussianModel::load_ply(): initialized Gaussian attributes from: {path}")


    def replace_tensor_to_optimizer(self, tensor, name, lr=None):
        optimizable_tensors = {}
        assert "ldec" not in name, "Latent decoder params cannot be replaced!"
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                if lr is not None:
                    group['lr'] = lr
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if "ldec" in group["name"]:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        for name in self.param_names:
            self._latents[name] = optimizable_tensors[name]

        for mask_name, mask in self.get_masks.items():
            mask.data = mask[valid_points_mask]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.infl_accum = self.infl_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.infl_denom = self.infl_denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        if self.frame_idx>1:
            if self.gate_atts is not None:
                self.gate_atts.prune_params(valid_points_mask)
            masks = self.get_masks
            for att_name in masks:
                decoder = self.latent_decoders[att_name]
                if type(decoder) == LatentDecoder and self.get_frz[att_name] == 'st':
                    decoder.mask = masks[att_name].flatten()
                elif type(decoder) == LatentDecoderRes:
                    if self.get_frz[att_name] == 'st':
                        decoder.mask = masks[att_name].flatten()
                    decoder.decoded_att = decoder.decoded_att[valid_points_mask]
                if self.prev_atts[att_name] is not None:
                    self.prev_atts[att_name] = self.prev_atts[att_name][valid_points_mask]
                if self.prev_latents[att_name] is not None:
                    self.prev_latents[att_name] = self.prev_latents[att_name][valid_points_mask]
            if self.init_probs is not None:
                self.init_probs = self.init_probs[valid_points_mask]
            self.added_mask = self.added_mask[valid_points_mask]
                    
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        masks = self.get_masks
        for group in self.optimizer.param_groups:
            if "ldec" in group["name"]:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

            
            masks[group['name']].data = torch.cat((masks[group['name']],
                                              torch.ones(extension_tensor.shape[0],1).bool().to(masks[group['name']].device)), dim=0)

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_flow,
                              clone_mask=None, split_mask=None, N=2):
        
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "op": new_opacities,
        "sc" : new_scaling,
        "rot" : new_rotation,
        "flow":  new_flow}

        num_new_points = new_xyz.shape[0]
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        for param in self.param_names:
            self._latents[param] = optimizable_tensors[param]

        self.xyz_gradient_accum = torch.zeros((self._latents["xyz"].shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._latents["xyz"].shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self._latents["xyz"].shape[0]), device="cuda")

        self.infl_accum = torch.cat((self.infl_accum,torch.zeros(new_xyz.shape[0]).to(self.infl_accum)))
        self.infl_denom = torch.cat((self.infl_denom,torch.zeros(new_xyz.shape[0]).to(self.infl_denom)))

        assert (clone_mask is None or split_mask is None)
        if self.frame_idx>1:
            if self.gate_atts is not None:
                self.gate_atts.add_params(clone_mask=clone_mask, split_mask=split_mask, N=N)
            masks = self.get_masks
            for att_name in masks:
                decoder = self.latent_decoders[att_name]
                if type(decoder) == LatentDecoder and self.get_frz[att_name] == 'st':
                    decoder.mask = masks[att_name].flatten()
                elif type(decoder) == LatentDecoderRes:
                    if self.get_frz[att_name] == 'st':
                        decoder.mask = masks[att_name].flatten()
                    if clone_mask is not None:
                        selected_decoded_att = decoder.decoded_att[clone_mask]
                        decoder.decoded_att = torch.cat((decoder.decoded_att,selected_decoded_att), dim=0)
                    if split_mask is not None:
                        selected_decoded_att = decoder.decoded_att[split_mask]
                        if "f_" in att_name and decoder.identity:
                            new_att = selected_decoded_att.repeat(N,1,1)
                        else:
                            new_att = selected_decoded_att.repeat(N,1)
                        decoder.decoded_att = torch.cat((decoder.decoded_att, new_att), dim=0)
                    
                if self.prev_atts[att_name] is not None:
                    if clone_mask is not None:
                        prev_att = self.prev_atts[att_name][clone_mask]
                        prev_latent = torch.zeros_like(self.prev_latents[att_name][clone_mask])
                    if split_mask is not None:
                        if self.prev_atts[att_name].dim() == 2:
                            prev_att = self.prev_atts[att_name][split_mask].repeat(N,1)
                        elif self.prev_atts[att_name].dim() == 3:
                            prev_att = self.prev_atts[att_name][split_mask].repeat(N,1,1)
                        if self.prev_latents[att_name].dim() == 2:
                            prev_latent = torch.zeros_like(self.prev_latents[att_name][split_mask].repeat(N,1))
                        elif self.prev_latents[att_name].dim() == 3:
                            prev_latent = torch.zeros_like(self.prev_latents[att_name][split_mask].repeat(N,1,1))
                    self.prev_atts[att_name] = torch.cat((self.prev_atts[att_name],prev_att), dim=0)
                    self.prev_latents[att_name] = torch.cat((self.prev_latents[att_name],prev_latent), dim=0)
            if self.init_probs is not None:
                if clone_mask is not None:
                    self.init_probs = torch.cat((self.init_probs, self.init_probs[clone_mask]),dim=0)
                if split_mask is not None:
                    self.init_probs = torch.cat((self.init_probs, self.init_probs[split_mask].repeat(N)),dim=0)


    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, is_dynamic=False):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        if selected_pts_mask.sum().item() == 0:
            return
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds, generator=self.split_generator)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        residual = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
        new_xyz = residual + self.get_xyz[selected_pts_mask].repeat(N, 1)
        if type(self.latent_decoders["xyz"]) == LatentDecoderRes:
            decoded_att = self.latent_decoders["xyz"].decoded_att[selected_pts_mask].repeat(N,1)
            new_xyz = self.latent_decoders["xyz"].invert(new_xyz-decoded_att)
        else:
            new_xyz = self.latent_decoders["xyz"].invert(new_xyz)
        
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        if type(self.latent_decoders["sc"])==LatentDecoderRes:
            decoded_att = self.latent_decoders["sc"].decoded_att[selected_pts_mask].repeat(N,1)
            new_scaling = self.latent_decoders["sc"].invert(new_scaling-decoded_att)
        else:
            new_scaling = self.latent_decoders["sc"].invert(new_scaling)

        new_rotation = self._latents["rot"][selected_pts_mask].repeat(N,1)

        if isinstance(self.latent_decoders["f_dc"],DecoderIdentity) \
            or (type(self.latent_decoders["f_dc"])==LatentDecoderRes \
                and self.latent_decoders["f_dc"].identity):
            new_features_dc = self._latents["f_dc"][selected_pts_mask].repeat(N,1,1)
        else:
            new_features_dc = self._latents["f_dc"][selected_pts_mask].repeat(N,1)

        if isinstance(self.latent_decoders["f_rest"],DecoderIdentity)\
            or (type(self.latent_decoders["f_rest"])==LatentDecoderRes \
                and self.latent_decoders["f_rest"].identity):
            new_features_rest = self._latents["f_rest"][selected_pts_mask].repeat(N,1,1)
        else:
            new_features_rest = self._latents["f_rest"][selected_pts_mask].repeat(N,1)

        new_opacity = self._latents["op"][selected_pts_mask].repeat(N,1)
        new_flow = self._latents["flow"][selected_pts_mask].repeat(N,1)


        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_flow,
                                   split_mask=selected_pts_mask, N=N)
        self.added_mask = torch.cat((self.added_mask,torch.ones(new_xyz.shape[0]).bool().to(self._latents["xyz"].device)))

        if is_dynamic:
            new_mapping = torch.cat((self.mapping, torch.nonzero(selected_pts_mask).repeat(N,1).flatten()), dim=0)
            if new_mapping.max() >= self.xyz_before.shape[0]:
                new_mapping[new_mapping >= self.xyz_before.shape[0]] = new_mapping[new_mapping[new_mapping >= self.xyz_before.shape[0]]]
            # self.mapping = torch.cat((self.mapping, torch.nonzero(selected_pts_mask).repeat(N,1).flatten()), dim=0) # TODO should this update to new_mapping?
            self.mapping = new_mapping


        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))

        # have to update the new_mapping after all the densification and pruning
        # to address the case where points are pruned and then their parents
        # point to elements that no longer exist
        # so we need to update the mapping to the original parent indices
        self.prune_points(prune_filter)
        if is_dynamic:
            new_mapping = self.mapping[~prune_filter]
            if new_mapping.max() >= self.xyz_before.shape[0]:
                new_mapping[new_mapping >= self.xyz_before.shape[0]] = new_mapping[new_mapping[new_mapping >= self.xyz_before.shape[0]]]
            self.mapping =  new_mapping


    def densify_and_clone(self, grads, grad_threshold, scene_extent, is_dynamic=False):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        if self.added_mask is None:
            self.added_mask = torch.zeros(self._latents["xyz"].shape[0]).bool().to(self._latents["xyz"].device)

        if selected_pts_mask.sum().item() == 0:
            return

        new_xyz = self._latents["xyz"][selected_pts_mask]
        new_features_dc = self._latents["f_dc"][selected_pts_mask]
        new_features_rest = self._latents["f_rest"][selected_pts_mask]
        new_scaling = self._latents["sc"][selected_pts_mask]
        new_rotation = self._latents["rot"][selected_pts_mask]
        new_opacities = self._latents["op"][selected_pts_mask]
        new_flow = self._latents["flow"][selected_pts_mask]


        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_flow,
                                   clone_mask=selected_pts_mask)
        self.added_mask = torch.cat((self.added_mask,torch.ones(new_xyz.shape[0]).bool().to(self._latents["xyz"].device)))

        if is_dynamic:
            new_mapping = torch.cat((self.mapping, torch.nonzero(selected_pts_mask).flatten()), dim=0)
            # if the new mapping includes items that are not in xyz_before, need to reference their parents instead
            if new_mapping.max() >= self.xyz_before.shape[0]:
                new_mapping[new_mapping >= self.xyz_before.shape[0]] = new_mapping[new_mapping[new_mapping >= self.xyz_before.shape[0]]]
            self.mapping = new_mapping



    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            
            
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
        
    def densify_dynamic(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        self.densify_and_clone(grads, max_grad, extent, is_dynamic=True)
        self.densify_and_split(grads, max_grad, extent, is_dynamic=True)
        if self.mapping.max() >= self.xyz_before.shape[0]:
           self.mapping[self.mapping >= self.xyz_before.shape[0]] = self.mapping[self.mapping[self.mapping >= self.xyz_before.shape[0]]]

        
        # prune any big points
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        big_points_vs = self.max_radii2D > max_screen_size
        big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
        prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        prune_mask = prune_mask*self.added_mask # Prune only added points
        
        self.prune_points(prune_mask)

        # modify the mapping after pruning
        self.mapping = self.mapping[~prune_mask]
        if self.mapping.max() >= self.xyz_before.shape[0]:
            self.mapping[self.mapping >= self.xyz_before.shape[0]] = self.mapping[self.mapping[self.mapping >= self.xyz_before.shape[0]]]

        
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def add_influence_stats(self, infl_tensor):
        self.infl_accum += infl_tensor
        self.infl_denom += 1

    @torch.no_grad()
    def influence_prune(self, infl_threshold):
        out = self.infl_accum/self.infl_denom
        out[out.isnan()] = 0.0
        prune_mask = out<=infl_threshold
            
        self.prune_points(prune_mask)
        new_mapping = self.mapping[~prune_mask]
        if new_mapping.max() >= self.xyz_before.shape[0]:
            new_mapping[new_mapping >= self.xyz_before.shape[0]] = new_mapping[new_mapping[new_mapping >= self.xyz_before.shape[0]]]
        self.mapping = new_mapping
        self.infl_accum *= 0
        self.infl_denom *= 0

    def copy(self):
        """Create a deep copy of the GaussianModel instance.
        
        Returns:
            GaussianModel: A new instance with copied attributes and parameters.
        """
        # Create new instance with same initialization parameters
        new_model = GaussianModel(self.max_sh_degree, self.latent_args, self.model_args, self.frame_idx, self.use_xyz_legacy)
        
        # Copy basic attributes
        new_model.active_sh_degree = self.active_sh_degree
        new_model.spatial_lr_scale = self.spatial_lr_scale
        new_model.percent_dense = self.percent_dense
        
        # Copy latents
        for param_name in self.param_names:
            new_model._latents[param_name] = nn.Parameter(self._latents[param_name].data.clone().requires_grad_(True))
        
        # Copy decoder state dicts
        atts = self.get_atts # All the latent variables
        for i, att_name in enumerate(atts):
            if not isinstance(self.latent_decoders[param_name], DecoderIdentity):
                # first check if the decoder is a DecoderIdentity
                if isinstance(new_model.latent_decoders[param_name], DecoderIdentity):
                    decoder = LatentDecoderRes(
                        latent_dim=self.latent_args.latent_dim[i],
                        feature_dim=self.feature_dims[att_name],
                        ldecode_matrix=self.latent_args.ldecode_matrix[i],
                        latent_norm=self.latent_args.latent_norm[i],
                        num_layers_dec=self.latent_args.num_layers_dec[i],
                        hidden_dim_dec=self.latent_args.hidden_dim_dec[i],
                        activation=self.latent_args.activation[i],
                        use_shift=self.latent_args.use_shift[i],
                        ldec_std=self.latent_args.ldec_std[i],
                        final_activation=self.latent_args.final_activation[i],
                    ).cuda()
                    new_model.latent_decoders[att_name] = decoder
                else: 
                    new_model.latent_decoders[param_name].load_state_dict(
                        self.latent_decoders[param_name].state_dict()
                    )
        
        # Copy masks
        new_model.mask_xyz.data = self.mask_xyz.data.clone()
        new_model.mask_features_dc.data = self.mask_features_dc.data.clone()
        new_model.mask_features_rest.data = self.mask_features_rest.data.clone()
        new_model.mask_scaling.data = self.mask_scaling.data.clone()
        new_model.mask_rotation.data = self.mask_rotation.data.clone()
        new_model.mask_opacity.data = self.mask_opacity.data.clone()
        new_model.mask_flow.data = self.mask_flow.data.clone()
        
        # Copy freeze states
        new_model.frz_xyz = self.frz_xyz
        new_model.frz_features_dc = self.frz_features_dc
        new_model.frz_features_rest = self.frz_features_rest
        new_model.frz_scaling = self.frz_scaling
        new_model.frz_rotation = self.frz_rotation
        new_model.frz_opacity = self.frz_opacity
        new_model.frz_flow = self.frz_flow
        
        # Copy previous attributes and latents
        for param_name in self.param_names:
            if self.prev_atts[param_name] is not None:
                new_model.prev_atts[param_name] = self.prev_atts[param_name].clone()
            if self.prev_latents[param_name] is not None:
                new_model.prev_latents[param_name] = self.prev_latents[param_name].clone()
        
        # Copy gate attributes if they exist
        if self.gate_atts is not None:
            new_model.gate_atts = self.gate_atts.copy()
            new_model.gate_params = self.gate_params.copy()
        
        # Copy other tensors
        new_model.max_radii2D = self.max_radii2D.clone()
        new_model.xyz_gradient_accum = self.xyz_gradient_accum.clone()
        new_model.infl_accum = self.infl_accum.clone()
        new_model.denom = self.denom.clone()
        new_model.infl_denom = self.infl_denom.clone()
        
        # Copy mapping and xyz_before if they exist and are not None
        if hasattr(self, 'mapping') and self.mapping is not None:
            new_model.mapping = self.mapping.clone()
        if hasattr(self, 'xyz_before') and self.xyz_before is not None:
            new_model.xyz_before = self.xyz_before.clone()
        
        # Copy added_mask if it exists and is not None
        if hasattr(self, 'added_mask') and self.added_mask is not None:
            new_model.added_mask = self.added_mask.clone()
        
        # Copy init_probs if it exists and is not None
        if hasattr(self, 'init_probs') and self.init_probs is not None:
            new_model.init_probs = self.init_probs.clone()
        
        return new_model
