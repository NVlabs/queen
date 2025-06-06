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


from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, config: dict, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            if key in config:
                value = config[key]
            elif key.startswith("_") and key[1:] in config:
                value = config[key[1:]]
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class PrefixParamGroup:
    def __init__(self, parser: ArgumentParser, config: dict,  prefixes:str, name : str, fill_none = False):
        group = parser.add_argument_group(name)

        for key, value in vars(self).copy().items():
            t = type(value)
            if key in config:
                value = config[key]
            elif key.startswith("_") and key[1:] in config:
                value = config[key[1:]]
            value = value if not fill_none else None 
            if t == bool:
                group.add_argument("--" + key, default=value, action="store_true")
            elif t == list:
                for i, prefix in enumerate(prefixes):
                    if prefix+'_'+key in config:
                        cur_value = config[prefix+'_'+key]
                        value[i] = cur_value
                    group.add_argument("--" + prefix+'_'+key, default=value[i], type=type(value[0]))
                    setattr(self, prefix+'_'+key, value[i])
            else:
                group.add_argument("--" + key, default=value, type=t)

        self.prefixes = prefixes

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        
        for key, value in vars(self).items():
            if key == "prefixes":
                continue
            t = type(value)
            if t == list:
                assert len(value) == len(self.prefixes)
                updated_value = []
                for i,param_group in enumerate(self.prefixes):
                    updated_value.append(getattr(group, param_group+'_'+key))
                setattr(group, key, updated_value)

        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, config, sentinel=False):
        self.sh_degree = 2
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self.img_fmt = "png"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.test_indices = [0]
        self.test_interval = 250
        self.log_interval = 100
        self.test_saved_pkl = False
        self.znear = 5.0
        self.zfar = 300.0
        self.log_images = False
        self.log_ply = False
        self.log_compressed = False
        self.max_frames = 300
        self.start_idx = 0
        self.seed = 0
        self.adaptive_iters = False
        self.adaptive_update = False
        self.adaptive_render = False
        self.adaptive_update_period = 0.0
        self.world_scale = 40.0
        self.timed = False

        self.use_wandb = False

        self.wandb_project = "queen-dev"
        self.wandb_entity = "nvr-amri"
        self.wandb_run_name = "test-run"
        self.wandb_mode = "online"
        self.wandb_log_images = False
        self.wandb_tags = ""
        self.wandb_log_interval = 10

        self.flow_model_ckpt = 'unimatch/pretrained/gmflow-scale1-mixdata-train320x576-4c3a6e9a.pth'
        # self.depth_model_ckpt = 'unimatch/pretrained/gmdepth-scale1-regrefine1-resumeflowthings-scannet-90325722.pth'
        self.depth_model_ckpt = 'MiDaS/weights/dpt_beit_large_512.pt'
        self.depth_init = False # Use GT depth for initializing points with colmap
        self.depth_thresh = 0.15 # Threshold for alpha mask for reinitializing with new points
        self.depth_scale = 1.0
        self.depth_pix_range = 20
        self.depth_num_comp = 5
        self.depth_tolerance = 0.01
        self.depth_pair_interval = 200
        
        self.update_mask = "diff" # Choose from ["flow2d", "diff", "viewspace_diff", "none"]
        self.update_loss = "mae" # Choose from ["mae","mse","ssim"] for loss for viewspace_diff gradients
        # self.freeze_thresh = 0.07
        self.gaussian_update_thresh = 0.07 # (0-255)/255 range for diff, pixel shift for flow2d, viewspace gradients for viewspace_diff
        self.pixel_update_thresh = 0.07  # if using a different threshold for the adaptive pixel thresholding
        self.flow_scale = 0.7 # Scale for flow prediction if flow2d
        self.flow_batch = -1 # Scale for flow prediction if flow2d
        self.dilate_size = 48

        self.flow_update = False # Use scene flow deltas to update points for next frame
        self.use_gt_flow = False # If False, compute flow between current render and next GT, else between current GT and enxt
        self.flow_loss_type = "warp" # Choose from ["warp", "render"] -> render is loss from next image render, warp is loss from 2d flow warping

        self.frz_xyz = "none" # ["none","st","all"]
        self.frz_f_dc = "none"  
        self.frz_f_rest = "none"  
        self.frz_sc = "none"  
        self.frz_rot = "none" 
        self.frz_op = "none"  
        self.frz_flow = "none"

        self.gate_temp = 0.5
        self.gate_gamma = -0.1
        self.gate_eta = 1.1
        self.gate_lr = 1.0e-4
        self.gate_lambda_l2 = 0.0
        self.gate_lambda_l0 = 0.1

        super().__init__(parser, config, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser, config):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, config, "Pipeline Parameters")

class OptimizationParamsRest(ParamGroup):
    def __init__(self, parser, config):
        self.epochs_rest = 3
        self.position_lr_init_rest = 0.00016
        self.position_lr_final_rest = 0.0000016
        self.position_lr_delay_mult_rest = 0.01
        self.position_lr_max_steps_rest = 30_000
        self.features_dc_lr_rest = 0.0025
        self.features_rest_lr_rest = 0.0025/20
        self.opacity_lr_rest = 0.05
        self.scaling_lr_rest = 0.001
        self.rotation_lr_rest = 0.005
        self.flow_lr_rest = 0.0001
        self.percent_dense_rest = 0.01
        self.lambda_depthssim_rest = 0.0
        self.lambda_dssim_rest = 0.2
        self.lambda_fdssim_rest = 0.0
        self.lambda_depth_rest = 0.0
        self.lambda_alpha_rest = 0.0
        self.lambda_flow_rest = 0.0
        self.lambda_consistency_rest = 0.0
        self.lambda_tv_rest = 0.0
        self.lambda_posres_rest = 0.0
        self.color_from_iter_rest = 0
        self.depth_from_iter_rest = 4000
        self.depth_until_iter_rest = 10000
        self.alpha_from_iter_rest = 1000
        self.flow_from_iter_rest = 0.0
        self.calc_dense_stats_rest = 0
        self.min_opacity_rest = 0.01
        self.densification_interval_rest = 5
        self.opacity_reset_interval_rest = 600
        self.densify_from_epoch_rest = 0
        self.densify_until_epoch_rest = -1.0
        self.densify_grad_threshold_rest = 0.0002
        self.prune_from_iter_rest = 0
        self.prune_until_iter_rest = -1
        self.prune_interval_rest = 3000
        self.prune_threshold_rest = 0.0
        self.size_threshold_rest = 20
        self.random_background_rest = False
        self.resize_scale_rest = 1.0
        self.resize_period_rest = 0.0
        self.transform_rest = 'downsample'
        self.prune_interval_rest = 8000
        self.weight_decay_rest = 0.0

        self.xyz_ldecs_lr_rest = 1.0e-4
        self.f_dc_ldecs_lr_rest = 1.0e-4
        self.f_rest_ldecs_lr_rest = 1.0e-4
        self.op_ldecs_lr_rest = 1.0e-4
        self.sc_ldecs_lr_rest = 1.0e-4
        self.rot_ldecs_lr_rest = 1.0e-4
        self.flow_ldecs_lr_rest = 1.0e-4

        self.xyz_lr_scaling_rest = 1.0
        self.f_dc_lr_scaling_rest = 1.0
        self.f_rest_lr_scaling_rest = 1.0
        self.op_lr_scaling_rest = 1.0
        self.sc_lr_scaling_rest = 1.0
        self.rot_lr_scaling_rest = 1.0
        self.flow_lr_scaling_rest = 1.0

        super().__init__(parser, config, "Optimization Parameters Rest")

    def extract(self, args):
        g = super().extract(args)
        g.latents_lr_scaling_rest = [g.xyz_lr_scaling_rest, 
                                   g.f_dc_lr_scaling_rest, 
                                   g.f_rest_lr_scaling_rest, 
                                   g.sc_lr_scaling_rest, 
                                   g.rot_lr_scaling_rest, 
                                   g.op_lr_scaling_rest,
                                   g.flow_lr_scaling_rest]
        g.ldecs_lr_rest = [g.xyz_ldecs_lr_rest,
                         g.f_dc_ldecs_lr_rest,
                         g.f_rest_ldecs_lr_rest,
                         g.sc_ldecs_lr_rest,  
                         g.rot_ldecs_lr_rest,
                         g.op_ldecs_lr_rest,
                         g.flow_ldecs_lr_rest]
        return g

class OptimizationParamsInitial(ParamGroup):
    def __init__(self, parser, config):
        self.epochs = 500
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 10_000
        self.features_dc_lr = 0.0025
        self.features_rest_lr = 0.0025/20
        self.opacity_lr = 0.05
        self.scaling_lr = 0.001
        self.rotation_lr = 0.001
        self.flow_lr = 0.0001
        self.percent_dense = 0.01
        self.lambda_depthssim = 0.0
        self.lambda_fdssim = 0.0
        self.lambda_dssim = 0.2
        self.lambda_depth = 0.0
        self.lambda_alpha = 0.0
        self.lambda_flow = 0.0
        self.lambda_consistency = 0.0
        self.lambda_tv = 0.0
        self.lambda_posres = 0.0
        self.color_from_iter = 0
        self.depth_from_iter = 4000
        self.depth_until_iter = 4000
        self.alpha_from_iter = 1000
        self.flow_from_iter = 0.5
        self.calc_dense_stats = 0
        self.min_opacity = 0.005
        self.densification_interval = 5
        self.opacity_reset_interval = 600
        self.densify_from_epoch = 50
        self.densify_until_epoch = 0.8
        self.densify_grad_threshold = 0.0002
        self.prune_from_iter = 10000
        self.prune_until_iter = -1
        self.prune_interval = 3000
        self.prune_threshold = 1.0e-5
        self.size_threshold = 20
        self.random_background = False
        self.resize_scale = 1.0
        self.resize_period = 0.0
        self.transform = 'downsample'
        self.prune_interval = 8000
        self.weight_decay = 0.0

        self.xyz_ldecs_lr = 1.0e-4
        self.f_dc_ldecs_lr = 1.0e-4
        self.f_rest_ldecs_lr = 1.0e-4
        self.op_ldecs_lr = 1.0e-4
        self.sc_ldecs_lr = 1.0e-4
        self.rot_ldecs_lr = 1.0e-4
        self.flow_ldecs_lr = 1.0e-4

        self.xyz_lr_scaling = 1.0
        self.f_dc_lr_scaling = 1.0
        self.f_rest_lr_scaling = 1.0
        self.op_lr_scaling = 1.0
        self.sc_lr_scaling = 1.0
        self.rot_lr_scaling = 1.0
        self.flow_lr_scaling = 1.0

        super().__init__(parser, config, "Optimization Parameters Initial")

    def extract(self, args):
        g = super().extract(args)
        g.latents_lr_scaling = [g.xyz_lr_scaling, 
                                   g.f_dc_lr_scaling, 
                                   g.f_rest_lr_scaling, 
                                   g.sc_lr_scaling, 
                                   g.rot_lr_scaling, 
                                   g.op_lr_scaling,
                                   g.flow_lr_scaling]
        g.ldecs_lr = [g.xyz_ldecs_lr,
                         g.f_dc_ldecs_lr,
                         g.f_rest_ldecs_lr,
                         g.sc_ldecs_lr,
                         g.rot_ldecs_lr,
                         g.op_ldecs_lr,
                         g.flow_ldecs_lr]
        return g
    
class OptimizationParams:
    def __init__(self, opt_init, opt_rest, frame_idx=1):
        self.opt_init = vars(opt_init)
        self.opt_rest = vars(opt_rest)
        self.frame_idx = frame_idx
        self.set_params(frame_idx)

    def set_params(self, frame_idx):
        self.frame_idx = frame_idx
        for param_name in self.opt_init:
            value = self.opt_init[param_name] if frame_idx == 1 else self.opt_rest[param_name+'_rest']
            setattr(self, param_name, value)

class QuantizeParams(PrefixParamGroup): 
    def __init__(self, parser, config, sentinel=False):
        self.latent_dim = [3,1,16,3,4,1,3]
        self.latent_norm = ["none", "none", "none", "none", "none", "none", "none"]
        self.quant_type = ['none']*7
        self.latent_scale_norm = ["none"]*7
        self.ldecode_matrix = ["learnable"]*7
        self.use_shift = [1]*7
        self.ldec_std = [1.0]*7
        self.num_layers_dec = [0]*7
        self.hidden_dim_dec = [3,1,16,3,4,1,1]
        self.activation = ["relu"]*7
        self.final_activation = ["none"]*7
        self.param_names = ["xyz", "f_dc", "f_rest", "sc", "rot", "op", "flow"]
        self.invert_type = ["dec", "dec", "dec", "dec", "dec", "dec", "dec"]
        self.freeze_after = [1.0]*7
        self.freeze_before = [0.0]*7
        self.gate_params = ["none"]*7

        self.quant_after = [0.0]*7
        super().__init__(parser, config, self.param_names, "Latent Parameters", sentinel)

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except (TypeError, FileNotFoundError):
        print("Config file not found")

    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
