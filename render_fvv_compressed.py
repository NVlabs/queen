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
from scene import Scene
import os
import yaml
import socket
import sys
import hashlib
from collections import defaultdict
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
import torch.utils.benchmark as benchmark
from utils.general_utils import safe_state
from utils.image_utils import psnr, l1_loss
from utils.loss_utils import ssim
import numpy as np

from argparse import ArgumentParser
from gaussian_renderer import GaussianModel
from utils.loader_utils import MultiViewVideoDataset, SequentialMultiviewSampler
from scene.decoders import DecoderIdentity
from arguments import ModelParams, PipelineParams, OptimizationParams, QuantizeParams, OptimizationParamsInitial, OptimizationParamsRest, get_combined_args
from utils.compress_utils import search_for_max_iteration

def do_system(arg):
    print(f"==== running: {arg}")
    err=os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)

def compare_gaussians(ply_gaussians, compressed_gaussians):
    """Compare attributes between PLY and compressed gaussians."""
    differences = {}
    
    # First log the size differences
    print(f"\nSize comparison:")
    print(f"PLY gaussians: {ply_gaussians.get_xyz.shape[0]}")
    print(f"Compressed gaussians: {compressed_gaussians.get_xyz.shape[0]}")
    
    # Find the minimum size to compare
    min_size = min(ply_gaussians.get_xyz.shape[0], compressed_gaussians.get_xyz.shape[0])
    
    # Get all attribute names that exist in both models
    common_attrs = set(ply_gaussians.get_atts) & set(compressed_gaussians.get_atts)
    
    for att_name in common_attrs:
        try:
            # Get the raw attributes first
            ply_att = ply_gaussians.get_decoded_atts[att_name]
            compressed_att = compressed_gaussians.get_decoded_atts[att_name]
            
            # Ensure we're working with tensors
            if not isinstance(ply_att, torch.Tensor):
                ply_att = torch.tensor(ply_att, device=ply_gaussians.get_xyz.device)
            if not isinstance(compressed_att, torch.Tensor):
                compressed_att = torch.tensor(compressed_att, device=compressed_gaussians.get_xyz.device)
            
            # Ensure same device
            compressed_att = compressed_att.to(ply_att.device)
            
            # Take only the minimum size
            ply_att = ply_att[:min_size]
            compressed_att = compressed_att[:min_size]
            
            # Calculate absolute and relative differences
            abs_diff = torch.abs(ply_att - compressed_att)
            rel_diff = abs_diff / (torch.abs(ply_att) + 1e-8)  # Add small epsilon to avoid division by zero
            
            differences[att_name] = {
                'max_abs_diff': abs_diff.max().item(),
                'mean_abs_diff': abs_diff.mean().item(),
                'max_rel_diff': rel_diff.max().item(),
                'mean_rel_diff': rel_diff.mean().item(),
                'size_mismatch': ply_gaussians.get_xyz.shape[0] != compressed_gaussians.get_xyz.shape[0],
                'shape': list(ply_att.shape)
            }
        except Exception as e:
            # print(f"Warning: Could not compare attribute {att_name}: {str(e)}")
            differences[att_name] = {
                'error': str(e),
                'size_mismatch': True
            }
    
    return differences

def compare_rendered_images(ply_img, pkl_img):
    """Compare rendered images from PLY and PKL representations."""
    metrics = {}
    
    # Convert to float and ensure same device
    ply_img = ply_img.float()
    pkl_img = pkl_img.float().to(ply_img.device)
    
    # Calculate image metrics
    metrics['psnr'] = psnr(ply_img, pkl_img).item()
    metrics['ssim'] = ssim(ply_img, pkl_img).item()
    metrics['l1'] = l1_loss(ply_img, pkl_img).item()
    
    # Calculate absolute difference image
    abs_diff = torch.abs(ply_img - pkl_img)
    metrics['max_abs_diff'] = abs_diff.max().item()
    metrics['mean_abs_diff'] = abs_diff.mean().item()
    
    return metrics, abs_diff

def render_fvv(dataset: ModelParams, opt: OptimizationParams, pipeline: PipelineParams, qp:QuantizeParams, args,
             skip_train: bool, skip_test: bool, render_compressed: bool):
    
    with torch.no_grad():
        # Create the gaussian model and scene, initialized with frame 1 images from dataset
        qp.seed = dataset.seed
        gaussians = GaussianModel(dataset.sh_degree, qp, dataset)
        scene = Scene(dataset, gaussians, 
                      train_image_data= None, test_image_data=None, N_video_views=args.num_video_views)
        
        gaussians.training_setup(opt)
        
        # Create output directories
        if render_compressed:
            spiral_dir = os.path.join(dataset.model_path, 'spiral_compressed')
        else:
            spiral_dir = os.path.join(dataset.model_path, 'spiral_rendered')
        os.makedirs(spiral_dir, exist_ok=True)
        
        # Create comparison directories if in comparison mode
        if args.render_compare:
            compare_dir = os.path.join(dataset.model_path, 'comparison')
            os.makedirs(compare_dir, exist_ok=True)
            os.makedirs(os.path.join(compare_dir, 'ply'), exist_ok=True)
            os.makedirs(os.path.join(compare_dir, 'pkl'), exist_ok=True)
            os.makedirs(os.path.join(compare_dir, 'diff'), exist_ok=True)
            os.makedirs(os.path.join(compare_dir, 'metrics'), exist_ok=True)
            
            # Create log file for comparison metrics
            with open(os.path.join(compare_dir, 'comparison_metrics.csv'), 'w') as f:
                f.write("Frame,Attribute,MaxAbsDiff,MeanAbsDiff,MaxRelDiff,MeanRelDiff,SizeMismatch,Shape\n")
            
            # Create log file for image metrics
            with open(os.path.join(compare_dir, 'image_metrics.csv'), 'w') as f:
                f.write("Frame,PSNR,SSIM,L1,MaxAbsDiff,MeanAbsDiff\n")
        
        cameras = scene.getVideoCameras()
        for start_frame_idx in tqdm(range(1,dataset.max_frames+1), desc="Rendering spiral"):
            gaussians.frame_idx = start_frame_idx
            scene.model_path = os.path.join(args.model_path,'frames',str(start_frame_idx).zfill(4))
            
            if start_frame_idx == 1:
                scene.loaded_iter = search_for_max_iteration(os.path.join(scene.model_path, "point_cloud"))
                scene.gaussians.load_ply(os.path.join(scene.model_path,
                                                                "point_cloud.ply"))
            else:
                if args.render_compare:
                    # Load both PLY and PKL versions for comparison
                    ply_gaussians = scene.gaussians.copy()
                    scene.loaded_iter = search_for_max_iteration(os.path.join(scene.model_path, "point_cloud"))
                    ply_gaussians.load_ply(os.path.join(scene.model_path,
                                                                "point_cloud.ply"))
                    
                    # Load compressed version
                    scene.gaussians.load_compressed_pkl(os.path.join(scene.model_path,
                                                                "compressed",
                                                                "point_cloud.pkl"))
                    
                    # Compare gaussian attributes
                    differences = compare_gaussians(ply_gaussians, scene.gaussians)
                    
                    # Log gaussian differences
                    with open(os.path.join(compare_dir, 'comparison_metrics.csv'), 'a') as f:
                        for att_name, diff_stats in differences.items():
                            if 'error' in diff_stats:
                                f.write(f"{start_frame_idx},{att_name},ERROR,{diff_stats['error']}\n")
                            else:
                                f.write(f"{start_frame_idx},{att_name},{diff_stats['max_abs_diff']:.6f},{diff_stats['mean_abs_diff']:.6f},{diff_stats['max_rel_diff']:.6f},{diff_stats['mean_rel_diff']:.6f},{diff_stats['size_mismatch']},{diff_stats['shape']}\n")
                    
                    # Render from both versions
                    camera = cameras[start_frame_idx-1]
                    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
                    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                    
                    # Render PLY version
                    ply_img = render(camera, ply_gaussians, pipeline, background)["render"]
                    torchvision.utils.save_image(ply_img, os.path.join(compare_dir, 'ply', f"{start_frame_idx:04d}.png"))
                    
                    # Render PKL version
                    pkl_img = render(camera, scene.gaussians, pipeline, background)["render"]
                    torchvision.utils.save_image(pkl_img, os.path.join(compare_dir, 'pkl', f"{start_frame_idx:04d}.png"))
                    
                    # Compare rendered images
                    img_metrics, diff_img = compare_rendered_images(ply_img, pkl_img)
                    
                    # Save difference image
                    torchvision.utils.save_image(diff_img, os.path.join(compare_dir, 'diff', f"{start_frame_idx:04d}.png"))
                    
                    # Log image metrics
                    with open(os.path.join(compare_dir, 'image_metrics.csv'), 'a') as f:
                        f.write(f"{start_frame_idx},{img_metrics['psnr']:.6f},{img_metrics['ssim']:.6f},{img_metrics['l1']:.6f},{img_metrics['max_abs_diff']:.6f},{img_metrics['mean_abs_diff']:.6f}\n")
                                        
                    print("\nImage metrics:")
                    print(f"  PSNR: {img_metrics['psnr']:.6f}")
                    print(f"  SSIM: {img_metrics['ssim']:.6f}")
                    print(f"  L1: {img_metrics['l1']:.6f}")
                    print(f"  Max absolute difference: {img_metrics['max_abs_diff']:.6f}")
                    print(f"  Mean absolute difference: {img_metrics['mean_abs_diff']:.6f}")
                    
                elif render_compressed:
                    scene.gaussians.load_compressed_pkl(os.path.join(scene.model_path,
                                                                "compressed",
                                                                "point_cloud.pkl"))
                    
                else:
                    scene.loaded_iter = search_for_max_iteration(os.path.join(scene.model_path, "point_cloud"))
                    scene.gaussians.load_ply(os.path.join(scene.model_path,
                                                                "point_cloud",
                                                                "iteration_" + str(scene.loaded_iter),
                                                                "point_cloud.ply"))

            bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                
            if "immersive" in dataset.source_path and False:
                for idx  in tqdm(range(300), desc="Rendering spiral"):
                    camera = cameras[idx]
                    img = render(camera, gaussians, pipeline, background)["render"]
                    torchvision.utils.save_image(img, os.path.join(spiral_dir, str(idx).zfill(4)+ ".png"))
            else:
                camera = cameras[start_frame_idx-1]
                img = render(camera, gaussians, pipeline, background)["render"]
                torchvision.utils.save_image(img, os.path.join(spiral_dir, str(start_frame_idx).zfill(4)+ ".png"))
            
            # Update previous frame's attributes and latents for next frame's residual encoding
            if (render_compressed or args.render_compare) and start_frame_idx != dataset.max_frames:
                # Used for residual encoding of next frame
                for att_name in gaussians.get_atts:
                    gaussians.prev_atts[att_name] = gaussians.get_decoded_atts[att_name].clone()
                    gaussians.prev_latents[att_name] = gaussians.get_atts[att_name].clone()
                    gaussians.prev_atts[att_name].requires_grad_(False)
                    gaussians.prev_latents[att_name].requires_grad_(False)

        cmd = 'ffmpeg -y -thread_queue_size 1024 -framerate 30 -pattern_type glob -i "%s/*.png" -vb 20M -pix_fmt yuv420p %s/output.mp4' % (spiral_dir, spiral_dir)
        print(cmd)
        do_system(cmd)
            
if __name__ == "__main__":

    print('Running on ', socket.gethostname())
    # Config file is used for argument defaults. Command line arguments override config file.
    config_path = sys.argv[sys.argv.index("--config")+1] if "--config" in sys.argv else None
    if config_path:
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        config = {}
    config = defaultdict(lambda: {}, config)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")

    lp = ModelParams(parser, config['model_params'])
    op_i = OptimizationParamsInitial(parser, config['opt_params_initial'])
    op_r = OptimizationParamsRest(parser, config['opt_params_rest'])
    pp = PipelineParams(parser, config['pipe_params'])
    qp = QuantizeParams(parser, config['quantize_params'])

    parser.add_argument('--config', type=str, default=None)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--num_spirals", type=int, default=2)
    parser.add_argument("--num_video_views", type=int, default=300)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--render_compressed", action="store_true")
    parser.add_argument("--render_compare", action="store_true", help="Render and compare both PLY and PKL representations")
    args = parser.parse_args(sys.argv[1:])
    
    # Merge optimization args for initial and rest and change accordingly
    op = OptimizationParams(op_i.extract(args), op_r.extract(args))

    print("Rendering " + args.model_path)
    safe_state(args.quiet)

    lp_args = lp.extract(args)
    pp_args = pp.extract(args)
    qp_args = qp.extract(args)

    render_fvv(lp_args, op, pp_args, qp_args, args, args.skip_train, args.skip_test, args.render_compressed)


