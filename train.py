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


import os
import sys
import torch
import socket
from random import randint, Random
from utils.loss_utils import l1_loss, ssim, l2_loss, tv_loss, lp_loss, DepthRelLoss, mse_loss
from gaussian_renderer import render, network_gui, render_mask, render_mask_shift
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import cv2
import copy
import uuid
import json
import time
import yaml
import hashlib
import functools
import torchvision
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
import torchvision.transforms.functional as F
from utils.image_utils import psnr, save_image
from scene.cameras import camName_from_Path, imageName_from_Path
from argparse import ArgumentParser, Namespace
from utils.general_utils import DecayScheduler, kthvalue
from utils.graphics_utils import adjust_depths
from utils.image_utils import resize_image, downsample_image, blur_image, get_mask, write_depth, coords_grid, flow_warp, coords_grid_proj, get_depth, resize_dims
from utils.loader_utils import MultiViewVideoDataset
from utils.loader_utils import SequentialMultiviewSampler, MultiViewVideoDataset
from arguments import ModelParams, PipelineParams, OptimizationParams, QuantizeParams, OptimizationParamsInitial, OptimizationParamsRest
from scene.utils import get_depth_model, get_depth_poses
from torchmetrics.functional.regression import pearson_corrcoef
from MiDaS.run import process
from scene.decoders import LatentDecoder, LatentDecoderRes, Gate
from generate_video_all import symlink

# Disable tqdm to make pdb easier to use
# Set to False to disable progress bars for debugging
enable_tqdm = True
enable_debug = False

EPS = 1.0e-7
try:
    from torch.utils.tensorboard import SummaryWriter
    if not ('SLURM_PROCID' in os.environ and os.environ['SLURM_PROCID']!='0'):
        TENSORBOARD_FOUND = True
    else:
        TENSORBOARD_FOUND = False
except ImportError:
    TENSORBOARD_FOUND = False

try:
    import wandb
    if not ('SLURM_PROCID' in os.environ and os.environ['SLURM_PROCID']!='0'):
        WANDB_FOUND = True
    else:
        WANDB_FOUND = False
except ImportError:
    WANDB_FOUND = False

def training(dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams, qp:QuantizeParams, testing_iterations: list, 
             saving_iterations: list, checkpoint_iterations, checkpoint: str, debug_from, args):
    """Main training function for QUEEN compressed Gaussian splatting."""
    wandb_enabled = WANDB_FOUND and dataset.use_wandb
    tb_writer = prepare_output_and_logger(args)
    generator = Random(dataset.seed)
    qp.seed = dataset.seed

    qp.use_shift = [bool(el) for el in qp.use_shift]

    # Create dataset and loader for training and testing at each time instance
    train_image_dataset = MultiViewVideoDataset(dataset.source_path, split='train', test_indices=dataset.test_indices,
                                                max_frames=dataset.max_frames, start_idx=dataset.start_idx, img_format=dataset.img_fmt)
    test_image_dataset = MultiViewVideoDataset(dataset.source_path, split='test', test_indices=dataset.test_indices, 
                                               max_frames=dataset.max_frames, start_idx=dataset.start_idx, 
                                               img_format=dataset.img_fmt)

    train_sampler = SequentialMultiviewSampler(train_image_dataset)
    if test_image_dataset.n_cams > 0:
        test_sampler = SequentialMultiviewSampler(test_image_dataset)

    train_loader = iter(torch.utils.data.DataLoader(train_image_dataset, batch_size=train_image_dataset.n_cams, 
                                                    sampler=train_sampler, num_workers=4))
    if test_image_dataset.n_cams > 0:
        test_loader = iter(torch.utils.data.DataLoader(test_image_dataset, batch_size=test_image_dataset.n_cams, 
                                                        sampler=test_sampler, num_workers=4))
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    print(f"training(): dataset.white_background set to {dataset.white_background}")
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    # Initial set of images to initialize camera and camera parameters
    # Image dimensions should remain constant throughout the video
    print(f"training(): loading data for the first frame...")
    tic = time.time()
    train_data = next(train_loader)
    train_images, train_paths = train_data
    if test_image_dataset.n_cams > 0:
        test_data = next(test_loader)
        test_images, test_paths = test_data
        test_image_data = {'image':test_images.cuda(),'path':test_paths,'frame_idx':0}
    else:
        print('No test cameras found, disabling testing.')
        test_images, test_paths = None, None
        test_image_data = {'image':None,'path':None,'frame_idx':0}

    train_image_data = {'image':train_images.cuda(),'path':train_paths,'frame_idx':0}
    
    print(f"training(): data loaded in {float(time.time() - tic):.2f} sec")

    # Create the gaussian model and scene, initialized with frame 1 images from dataset
    gaussians = GaussianModel(dataset.sh_degree, qp, dataset, use_xyz_legacy=args.use_xyz_legacy)

    max_frames = args.max_frames
    scene = Scene(
        dataset,
        gaussians,
        train_image_data= train_image_data,
        test_image_data=test_image_data,
        N_video_views=max_frames
    )
    # Setup training arguments
    gaussians.training_setup(opt)

    # Spiral cameras
    video_cameras = scene.getVideoCameras()

    # Metadata used by various components
    train_cameras = scene.getTrainCameras()
    n_frames, n_cams = train_image_dataset.n_frames, train_image_dataset.n_cams
    print(f"training(): running with {n_frames} frames from {n_cams} cameras")
    opt.iterations = opt.epochs*n_cams
    print(f"training(): opt.iterations set to {opt.iterations}")
    _,H,W = train_cameras[0].original_image.shape

    cur_frame_views = train_image_data['image']
    prev_frame_views = cur_frame_views

    # Vary number of iterations based on frame difference in json file
    if dataset.adaptive_iters and n_frames>1:
        frame_diff = json.load(open(os.path.join(dataset.source_path,'frame_diff.json'),'r'))['l2']
        frame_diff = np.array(frame_diff[:n_frames-1])
        epochs_rest = opt.opt_rest['epochs_rest']
        mult = np.clip(frame_diff/frame_diff.mean(),1/4,4) # between 0.25 to 4
        mult = mult/mult.mean()
        frame_epochs = np.ceil((mult*epochs_rest)).astype(np.int32)
        frame_iters = np.concatenate((np.array([opt.iterations]),frame_epochs*n_cams))
    else:
        frame_iters = np.array([opt.iterations]+[opt.opt_rest['epochs_rest']*n_cams]*(n_frames-1))
    if opt.lambda_depth>0.0 or dataset.depth_init:

        ## MiDas model for monocular depth estimation
        depth_model, transform, net_w, net_h = get_depth_model(dataset)
        for camera in train_cameras:
            gt_image = camera.original_image.permute(1,2,0).detach().cpu().numpy()
            image = transform({"image": gt_image})["image"]
            with torch.no_grad():
                prediction = process(torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
                                     depth_model, 'dpt_beit_large_512', image, (net_w, net_h), 
                                     gt_image.shape[1::-1],
                                     False, False)
                camera.gt_depth = torch.tensor(prediction).cuda()

        # Add points to gaussian model using the monocular depth
        if dataset.depth_init:
            gaussians.create_from_depth_immersive(cameras=train_cameras, spatial_lr_scale=gaussians.spatial_lr_scale, downsample_scale=1,
                                        alpha_thresh=dataset.depth_thresh, renderFunc = functools.partial(render_mask, 
                                                                                        pipe=pipe, 
                                                                                        bg_color=bg, 
                                                                                        image_shape=camera.original_image.shape, 
                                                                                        color_mask=None, 
                                                                                        render_depth=True))
            
        # Loss function for relative depth
        depth_loss_fn = DepthRelLoss(camera.original_image.shape[1], camera.original_image.shape[2],
                                     pix_diff=dataset.depth_pix_range, num_comp=dataset.depth_num_comp, 
                                     tolerance=dataset.depth_tolerance)

    # Progressive training scheduler - OBSOLETE: Remove in future cleanup
    resize_scale_sched = DecayScheduler(
                                        total_steps=int(opt.resize_period*(opt.iterations+1)),
                                        decay_name='cosine',
                                        start=opt.resize_scale,
                                        end=1.0,
                                        )

    start_frame_idx = 1
    training_metrics = []
    net_elapsed_time = 0.0
    net_iter_time = 0.0


    training_start = time.time()

    # Define video-wide metrics for wandb logging
    if wandb_enabled:
        wandb.define_metric("frame_idx")
        wandb.define_metric("frame/num_iterations", step_metric="frame_idx")
        wandb.define_metric("frame/test/loss_viewpoint/psnr", step_metric="frame_idx")
        wandb.define_metric("frame/test/loss_viewpoint/loss", step_metric="frame_idx")
        wandb.define_metric("frame/val/loss_viewpoint/psnr", step_metric="frame_idx")
        wandb.define_metric("frame/val/loss_viewpoint/loss", step_metric="frame_idx")
        wandb.define_metric("frame/size", step_metric="frame_idx")
        wandb.define_metric("frame/num_points", step_metric="frame_idx")
        wandb.define_metric("frame/update_points", step_metric="frame_idx")
        wandb.define_metric("frame/iter_time", step_metric="frame_idx")
        wandb.define_metric("frame/iter_time_io", step_metric="frame_idx")
        wandb.define_metric("frame/elapsed", step_metric="frame_idx")

    if opt.lambda_flow > 0.0:
        grid = coords_grid(1,H,W, device='cuda')

    if enable_tqdm:
        progress_bar_frame = tqdm(range(1, n_frames+1), desc="Training progress")
        progress_bar_frame.update(start_frame_idx-1)
    else:
        progress_bar_frame = None
        frame_counter = 0

    # start frame index loop
    for frame_idx in range(start_frame_idx, n_frames+1):

        # Frame-wise metrics for wandb logging
        if wandb_enabled and frame_idx <= 2:
            frame_str = f"{str(frame_idx).zfill(4)}"
            iter_metric = "iter_"+frame_str
            frame_str = "frame_"+frame_str
            wandb.define_metric(iter_metric)
            wandb.define_metric(frame_str+"/test/loss_viewpoint/best_psnr", step_metric=iter_metric)
            wandb.define_metric(frame_str+"/test/loss_viewpoint/psnr", step_metric=iter_metric)
            wandb.define_metric(frame_str+"/test/loss_viewpoint/l1_loss", step_metric=iter_metric)
            wandb.define_metric(frame_str+"/val/loss_viewpoint/psnr", step_metric=iter_metric)
            wandb.define_metric(frame_str+"/val/loss_viewpoint/l1_loss", step_metric=iter_metric)

            wandb.define_metric(frame_str+"/train_loss_patches/l1_loss", step_metric=iter_metric)
            wandb.define_metric(frame_str+"/train_loss_patches/total_loss", step_metric=iter_metric)
            wandb.define_metric(frame_str+"/num_points", step_metric=iter_metric)
            wandb.define_metric(frame_str+"/update_points", step_metric=iter_metric)
            wandb.define_metric(frame_str+"/elapsed", step_metric=iter_metric)
            wandb.define_metric(frame_str+"/size", step_metric=iter_metric)
            

        first_iter = 1
        scene.model_path = os.path.join(args.model_path,'frames',str(dataset.start_idx + frame_idx).zfill(4))

        os.makedirs(scene.model_path,exist_ok=True)

        ema_loss_for_log, cur_size, best_psnr = 0.0, 0.0, 0.0
        metrics = {'val':{'psnr':0.0, 'loss':0.0}, 'test':{'psnr':0.0, 'loss': 0.0}}
        camera_idx_stack = []
        report = None

        if dataset.timed:
            torch.cuda.synchronize()
        frame_start_io = time.time()
        frame_time_io = 0.0

        try: 
            # Pre-load data for next frame
            next_train_data = next(train_loader)
            next_train_images, next_train_paths = next_train_data[0].cuda(), next_train_data[1]
            next_frame_views = next_train_images

            orig_size = cur_frame_views.shape[-2:]
            rescaled_size = resize_dims(orig_size, dataset.flow_scale)

        except StopIteration:
            assert frame_idx == n_frames
            opt.lambda_flow = 0.0

        if dataset.timed:
            torch.cuda.synchronize()
        frame_start = time.time()
        frame_time = 0.0

        # Update a bunch of variables and models for each new frame
        if frame_idx > 1:

            # Initialize gate probabilities based on gradient differences or frame differences
            if dataset.update_mask == "viewspace_diff":
                # Compute viewspace gradient differences for gate initialization
                grad_diff = torch.zeros(gaussians.get_xyz.shape[0],1).to(gaussians._xyz)
                denom = torch.zeros(gaussians.get_xyz.shape[0],1).to(gaussians._xyz)
                gaussians.optimizer.zero_grad(set_to_none=True)
                for cam_idx, camera in enumerate(train_cameras):
                    render_pkg = render_mask(camera, gaussians, pipe, bg, image_shape=gt_image.shape)
                    camera.prev_rendered = render_pkg["render"].detach()
                    image, viewspace_point_tensor = render_pkg["render"], render_pkg["viewspace_points"]
                    visibility_filter = render_pkg["visibility_filter"]
                    cur_gt_image = cur_frame_views[cam_idx]
                    prev_gt_image = prev_frame_views[cam_idx]
                    if dataset.update_loss == "mae":
                        Ll1 = mse_loss(image, cur_gt_image)
                        Ll1_prev = mse_loss(image, prev_gt_image)
                    elif dataset.update_loss == "mse":
                        Ll1 = mse_loss(image, cur_gt_image)
                        Ll1_prev = mse_loss(image, prev_gt_image)
                    elif dataset.update_loss == "ssim":
                        Ll1 = 1.0-ssim(image, cur_gt_image)
                        Ll1_prev = 1.0-ssim(image, prev_gt_image)
                    elif dataset.update_loss == "mae_orig":
                        Ll1 = l1_loss(image, cur_gt_image)
                        Ll1_prev = l1_loss(image, prev_gt_image)
                    cur_loss = Ll1-Ll1_prev
                    cur_loss.backward()
                    cur_grad = viewspace_point_tensor.grad[visibility_filter,:2].clone()
                    with torch.no_grad():
                        viewspace_point_tensor.grad *= 0
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    grad_diff[visibility_filter] += torch.norm(cur_grad,dim=-1,keepdim=True)
                    denom[visibility_filter] += 1

                grad_diff[grad_diff.isnan()] = 0.0

                with torch.no_grad():
                    if dataset.adaptive_render and dataset.adaptive_update_period>0.0:
                        for camera in train_cameras:
                            grad_mask = (grad_diff.flatten()>dataset.pixel_update_thresh)
                            render_pkg = render_mask(camera, scene.gaussians, pipe, bg, 
                                                    gaussian_mask=grad_mask)
                            alphamask = (render_pkg["alpha"]>0.5).float()
                            camera.orig_mask = alphamask
                            mask_down = torch.nn.functional.max_pool2d(alphamask.unsqueeze(0), (dataset.dilate_size,dataset.dilate_size))
                            mask_dilate = torch.nn.functional.interpolate(mask_down, size=(alphamask.shape[-2],alphamask.shape[-1]))
                            camera.mask = (mask_dilate.squeeze(0).squeeze(0)>0).float()

                gaussian_mask = grad_diff>dataset.gaussian_update_thresh


            with torch.no_grad():

                # Load optimizer hyperparams (initial or rest) based on frame index
                opt.set_params(frame_idx)
                opt.iterations = frame_iters[frame_idx-1]
                opt.epochs = (opt.iterations//n_cams)
                gaussians.frame_idx = frame_idx
                # Create decoder and latents for quantized residuals if first time
                # Else reset latent values to 0
                gaussians.update_residuals()
                # Redefine the optimizer and other tracked variables for the gaussian model
                gaussians.training_setup(opt)
                # Load the current test data (Preloaded data for next frame is only for training)
                train_images, train_paths = cur_train_images, cur_train_paths
                if dataset.timed:
                    torch.cuda.synchronize()
                frame_time += time.time() - frame_start
                if test_image_dataset.n_cams > 0:
                    test_data = next(test_loader)
                    test_images, test_paths = test_data[0].cuda(), test_data[1]
                else:
                    if frame_idx == start_frame_idx:
                        print('No test cameras found, disabling testing.')
                    test_images, test_paths = None, None
                if dataset.timed:
                    torch.cuda.synchronize()
                frame_start = time.time()
                train_image_data = {'image':train_images,'path':train_paths}
                test_image_data = {'image':test_images,'path':test_paths}
            
                # Update the images and paths for all cameras in the scene with new frame index
                scene.updateCameraImages(args, train_image_data, test_image_data, frame_idx, resolution_scales=[1.0])
                train_cameras = scene.getTrainCameras()

                # If using a frame difference or 2d flow mask for gate initialization and adaptive masked training
                if dataset.update_mask =="diff":
                    flow_norm = torch.norm((prev_frame_views-cur_frame_views),dim=1,keepdim=True)/np.sqrt(3) # normalize across rgb
                        
                    # Mask if using fixed threshold
                    flow_mask = flow_norm>dataset.pixel_update_thresh


                    if dataset.adaptive_render and dataset.adaptive_update_period>0.0:
                        bg = torch.rand((3), device="cuda") if opt.random_background else background
                        gaussian_mask = torch.zeros_like(gaussians.mask_xyz)
                        # Freeze mask by back projecting pixel mask
                        net_influence = None
                        for idx,camera in enumerate(train_cameras):
                            render_pkg = render_mask(camera, gaussians, pipe, bg, image_shape=camera.original_image.shape, 
                                                    pixel_mask=flow_mask[idx].float(), render_depth=False)
                            influence = render_pkg["influence"]
                            if net_influence is None:
                                net_influence = influence
                            else:
                                net_influence += influence
                            gaussian_mask = torch.logical_or(gaussian_mask,influence[...,None]>0)


                        # Pixel mask by rerendering gaussian mask 
                        # (otherwise directly use the 2d mask as
                        for idx,camera in enumerate(train_cameras):
                            alphamask = flow_mask[idx].float()
                            camera.orig_mask = alphamask
                            mask_down = torch.nn.functional.max_pool2d(alphamask.unsqueeze(0), (dataset.dilate_size,dataset.dilate_size))
                            mask_dilate = torch.nn.functional.interpolate(mask_down, size=(alphamask.shape[-2],alphamask.shape[-1]))
                            camera.mask = (mask_dilate.squeeze(0).squeeze(0)>0).float()

                    if (dataset.gaussian_update_thresh != dataset.pixel_update_thresh) or \
                        not (dataset.adaptive_render and dataset.adaptive_update_period>0.0):
                        flow_mask = flow_norm>dataset.gaussian_update_thresh
                        # Rerun backprojection if we want to use a different threshold for our gaussian mask
                        gaussian_mask = torch.zeros_like(gaussians.mask_xyz)
                        # Freeze mask by back projecting pixel mask
                        net_influence = None
                        for idx,camera in enumerate(train_cameras):
                            render_pkg = render_mask(camera, gaussians, pipe, bg, image_shape=camera.original_image.shape, 
                                                    pixel_mask=flow_mask[idx].float(), render_depth=False)
                            influence = render_pkg["influence"]
                            gaussian_mask = torch.logical_or(gaussian_mask,influence[...,None]>0)
                            if net_influence is None:
                                net_influence = influence
                            else:
                                net_influence += influence
                
                gaussians.update_masks(dataset, None if dataset.update_mask == "none" else gaussian_mask)
                gaussians.freeze_atts(dataset)

                if dataset.adaptive_render and dataset.adaptive_update_period>0.0:
                    adaptive_update_epochs = np.ceil(opt.epochs*dataset.adaptive_update_period).astype(np.int32)
                    pix_thresh_vals = torch.ones(adaptive_update_epochs*n_cams)*dataset.pixel_update_thresh

                    if opt.iterations>pix_thresh_vals.shape[0]:
                        addn_pix_vals = torch.zeros(opt.iterations-pix_thresh_vals.shape[0]).to(pix_thresh_vals)
                        pix_thresh_vals = torch.cat((pix_thresh_vals,addn_pix_vals),dim=0)
                    assert pix_thresh_vals.shape[0] == opt.iterations
                else:
                    pix_thresh_vals = None

                # Initialize gate probabilities based on computed differences
                if any([gating!="none" for gating in qp.gate_params]):
                    if dataset.update_mask == "viewspace_diff":
                        # Use gradient differences for gate initialization
                        init_probs = grad_diff/(grad_diff+grad_diff.median())
                        gaussians.init_probs = init_probs.flatten()
                    elif dataset.update_mask == "diff":
                        # Use frame differences for gate initialization  
                        init_probs = net_influence/(net_influence+net_influence.mean())
                        gaussians.init_probs = init_probs.flatten()
                    else:
                        gaussians.init_probs = None
                    if gaussians.gate_atts is None:
                        gaussians.gate_atts = Gate(gaussians._xyz.shape[0], 
                                                  gamma=dataset.gate_gamma,
                                                  eta=dataset.gate_eta,
                                                  lr = dataset.gate_lr, 
                                                  temp=dataset.gate_temp,
                                                  lambda_l2=dataset.gate_lambda_l2, 
                                                  lambda_l0=dataset.gate_lambda_l0, 
                                                  init_probs=gaussians.init_probs)
                        gaussians.gate_atts.train()
                    else:
                        gaussians.gate_atts.reset_params(init_probs=gaussians.init_probs)
                        gaussians.gate_atts.train()

                if dataset.flow_update and opt.lambda_flow>0.0:
                    gaussians.update_points_flow()
                prev_frame_views = cur_frame_views

        if enable_tqdm and frame_idx == 1:
            progress_bar_iter = tqdm(range(first_iter, opt.iterations+1), 
                                     desc="Frame iteration progress")
        else:
            progress_bar_iter = None

        if dataset.timed:
            torch.cuda.synchronize()
        frame_time += time.time()- frame_start
        frame_start = time.time()
        frame_time_io += time.time() - frame_start_io
        frame_start_io = time.time()

        # Start training iteration loop for current frame
        for iteration in range(first_iter, opt.iterations + 1):        

            if enable_debug:
                print(f"DEBUG: started iteration {iteration}")

            if dataset.timed:
                torch.cuda.synchronize()
            iter_start = time.time()

            # Handle quantization and freezing of latent parameters
            if frame_idx>1:
                for i, att_name in enumerate(gaussians.param_names):
                    decoder = gaussians.latent_decoders[att_name]
                    # Switch from Identity Decoder to quantized encoding at specified iteration
                    if iteration == np.ceil(qp.quant_after[i]*opt.iterations) and type(decoder) == LatentDecoderRes:
                        decoder.identity = False
                        latent = gaussians._latents[att_name].data
                        if "f_" in att_name:
                            latent = latent.reshape(latent.shape[0],-1)
                        quant_latents = decoder.invert(latent)
                        new_lr = opt.latents_lr_scaling[i]*gaussians.orig_lr[att_name]
                        optimizable_tensors = gaussians.replace_tensor_to_optimizer(quant_latents, att_name,
                                                                                    lr=new_lr)
                        gaussians._latents[att_name] = optimizable_tensors[att_name]

                    # Handle parameter freezing schedule
                    assert qp.freeze_before[i]<= qp.freeze_after[i]
                    freeze_before_iter = np.ceil(qp.freeze_before[i]*opt.iterations)
                    freeze_after_iter = np.ceil(qp.freeze_after[i]*opt.iterations)
                    frz = gaussians.get_frz
                    if iteration==first_iter and iteration<freeze_before_iter:
                        gaussians.get_masks[att_name] *= False
                    elif iteration == freeze_before_iter:
                        gaussians.get_masks[att_name] += True
                        if frz[att_name] == "st":
                            # NOTE: might fail with densification
                            gaussians.get_masks[att_name] *= gaussian_mask

                    if iteration==(freeze_after_iter+1):
                        gaussians.get_masks[att_name] *= False
            
            gaussians.update_learning_rate(iteration, qp)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            # Pick a random Camera
            if not camera_idx_stack:
                camera_idx_stack = list(range(n_cams))
            cam_idx = camera_idx_stack.pop(generator.randint(0, len(camera_idx_stack)-1))
            viewpoint_cam = train_cameras[cam_idx]

            # Render

            bg = torch.rand((3), device="cuda") if opt.random_background else background

            # Loss
            gt_image = viewpoint_cam.original_image
            if opt.transform == "resize":
                gt_image = resize_image(gt_image, resize_scale_sched(iteration))
            elif "blur" in opt.transform and resize_scale_sched(iteration)!=1.0:
                if (iteration-1) % 100 == 0:
                    transform = blur_image(resize_scale_sched(iteration), opt.transform)
                gt_image = transform(gt_image)
            elif opt.transform == "downsample":
                gt_image = downsample_image(gt_image, resize_scale_sched(iteration))

            color_rw_mask = None

            # Initialize pixel_mask to None by default
            pixel_mask = None
            if frame_idx>1 and pix_thresh_vals is not None:
                if pix_thresh_vals[iteration-1]>0:
                    pixel_mask = viewpoint_cam.mask
            
            # render
            render_pkg = render_mask(viewpoint_cam, gaussians, pipe, bg, image_shape=gt_image.shape, 
                                     color_mask=color_rw_mask, render_depth=opt.lambda_depth>0.0, 
                                     backward_alpha=opt.lambda_alpha>0.0, 
                                     render_flow=opt.lambda_flow>0.0 and iteration > (opt.flow_from_iter*opt.iterations),
                                     pixel_mask=pixel_mask,
                                     update_mask=None)

            image, viewspace_point_tensor = render_pkg["render"], render_pkg["viewspace_points"]
            visibility_filter, radii = render_pkg["visibility_filter"], render_pkg["radii"]

            # Compute main reconstruction losses
            loss, Ll1 = torch.Tensor([0.0]).to(image.device), torch.Tensor([0.0]).to(image.device)
            if iteration>opt.color_from_iter:
                if pixel_mask is not None:
                    # Apply pixel mask for selective training
                    Ll1 = l1_loss(image*pixel_mask.unsqueeze(0), gt_image*pixel_mask.unsqueeze(0))
                    Lssim = ssim(image*pixel_mask.unsqueeze(0), gt_image*pixel_mask.unsqueeze(0))
                else:
                    Ll1 = l1_loss(image, gt_image)
                    Lssim = ssim(image, gt_image)

                loss += (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - Lssim)
            
            # Add regularization losses
            if opt.weight_decay>0.0:
                loss += opt.weight_decay * gaussians.std_reg()

            if gaussians.gate_atts is not None and gaussians.gate_atts.training:
                loss += gaussians.gate_atts.reg_loss(gaussians._ungated_xyz_res)

            if opt.lambda_posres>0.0:
                residual = gaussians.get_xyz-prev_xyz.detach()
                loss += opt.lambda_posres*torch.abs(residual).mean()
            if iteration > opt.alpha_from_iter and opt.lambda_alpha>0.0:
                loss += opt.lambda_alpha * l2_loss(render_pkg["alpha"],1.0)

            # Temporal flow consistency loss
            if opt.lambda_flow>0.0 and iteration > (opt.flow_from_iter*opt.iterations):
                if dataset.flow_loss_type == "render":
                    # Direct rendering approach for flow loss
                    render_pkg_flow = render_mask_shift(viewpoint_cam, gaussians, pipe, bg, image_shape=gt_image.shape)
                    next_image = render_pkg_flow["render"]
                    next_gt_image = next_frame_views[cam_idx]
                    if pixel_mask is not None:
                        next_Ll1 = l1_loss(next_image*pixel_mask.unsqueeze(0), next_gt_image*pixel_mask.unsqueeze(0))
                        next_Lssim = ssim(next_image*pixel_mask.unsqueeze(0), next_gt_image*pixel_mask.unsqueeze(0))
                    else:
                        next_Ll1 = l1_loss(next_image, next_gt_image)
                        next_Lssim = ssim(next_image, next_gt_image)
                    flow_loss = (1.0 - opt.lambda_fdssim) * next_Ll1 + opt.lambda_fdssim * (1.0 - next_Lssim)
                    loss += opt.lambda_flow * flow_loss

                elif dataset.flow_loss_type == "warp":
                    # Optical flow warping approach
                    rendered_flow = torch.clamp(render_pkg["flow"],-50,50) 
                    if dataset.use_gt_flow:
                        tgt_flow = cur_frame_views[cam_idx]
                    else:
                        tgt_flow = image.unsqueeze(0).detach()
                    warped = flow_warp(next_frame_views[cam_idx:cam_idx+1], rendered_flow.unsqueeze(0), grid)
                    if pixel_mask is not None:
                        flow_loss = (1-opt.lambda_dssim)*l1_loss(warped*pixel_mask.unsqueeze(0), 
                                                                 tgt_flow*pixel_mask.unsqueeze(0))+\
                                    opt.lambda_dssim*(1.0-ssim(warped*pixel_mask.unsqueeze(0), 
                                                                   tgt_flow*pixel_mask.unsqueeze(0)))
                    else:
                        flow_loss = (1-opt.lambda_dssim)*l1_loss(warped, tgt_flow)+\
                                    opt.lambda_dssim*(1.0-ssim(warped, tgt_flow))
                    loss += opt.lambda_flow * flow_loss  + opt.lambda_tv * tv_loss(rendered_flow)

            # Depth supervision loss (first frame only)
            if opt.lambda_depth>0.0 and iteration>opt.depth_from_iter and iteration<=opt.depth_until_iter and frame_idx == 1:
                pred_depth = render_pkg["depth"] 
                gt_depth = viewpoint_cam.gt_depth
                depth_loss = (1.0 - opt.lambda_depthssim) * depth_loss_fn(pred_depth, gt_depth)+ opt.lambda_depthssim * (1.0 - ssim(pred_depth.unsqueeze(0), gt_depth.unsqueeze(0)))
                loss += opt.lambda_depth * depth_loss + opt.lambda_tv * tv_loss(pred_depth)
                if iteration % dataset.depth_pair_interval == 0:
                    depth_loss_fn.resample_pairs()

            # Temporal consistency loss
            if opt.lambda_consistency>0.0:
                prev_image = viewpoint_cam.prev_rendered
                cur_image = render_pkg["render"]
                gt_diff = viewpoint_cam.image_diff
                # High consistency loss for low varying regions
                gt_diff = 1/(gt_diff+gt_diff.mean()) 
                # Normalize
                gt_diff = gt_diff/gt_diff.mean()
                consistency_loss = 1- l1_loss(prev_image*gt_diff, cur_image*gt_diff)
                loss += opt.lambda_consistency*consistency_loss
            loss.backward()
            if enable_debug:
                print(f'DEBUG ({iteration}): backpropagated')

            with torch.no_grad():
                if dataset.timed:
                    torch.cuda.synchronize()
                frame_time += time.time() - iter_start
                frame_time_io += time.time() - iter_start
                net_elapsed_time = time.time() - training_start
                # Log and save
                if dataset.test_interval>0:
                    is_test = (iteration % dataset.test_interval == 0) and frame_idx == 1
                else:
                    is_test = (iteration in testing_iterations) and frame_idx == 1
                if iteration == opt.iterations:
                    is_test = True
                    
                report = training_report(tb_writer, wandb_enabled, dataset, frame_idx, iteration, Ll1, loss, 
                                         l1_loss, cur_size, frame_time, is_test, scene, 
                                         render_mask, (pipe, background), prev_report=report)
                if enable_debug:
                    print(f'DEBUG ({iteration}): training_report done')

                if report:
                    if 'test' in report.keys():
                        report_configs = ['test','val']
                    else:
                        report_configs = ['val']
                    for config_name in report_configs:
                        metrics[config_name]['psnr'] = report[config_name]['psnr']
                        metrics[config_name]['loss'] = report[config_name]['l1']
                    if metrics['test']['psnr'] > best_psnr:
                        best_psnr = metrics['test']['psnr']
                        if wandb_enabled and frame_idx<=2:
                            wandb.log({frame_str+"/test/loss_viewpoint/best_psnr": best_psnr,
                                       iter_metric:iteration})
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

                if (iteration) % dataset.log_interval == 0 or iteration == opt.iterations:
                    cur_size = gaussians.size()/8/(10**6)
                    log_dict = {
                                "Loss": f"{ema_loss_for_log:.{5}f}",
                                "Num points": f"{gaussians._xyz.shape[0]}",
                                "Update points": f"{torch.count_nonzero(gaussians.mask_xyz)}" \
                                                if frame_idx>1 else f"{gaussians._xyz.shape[0]}",
                                "Size (MB)": f"{cur_size:.{2}f}",
                                "PSNR (Test)": f"{metrics['test']['psnr']:.{2}f}",
                                "PSNR (Val)": f"{metrics['val']['psnr']:.{2}f}",
                                }
                    if progress_bar_iter:
                        progress_bar_iter.set_postfix(log_dict)
                        progress_bar_iter.update(dataset.log_interval)
                if iteration == opt.iterations and progress_bar_iter:
                    progress_bar_iter.close()
                if (iteration in saving_iterations) or (iteration == opt.iterations and -1 in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    if args.save_format == "pkl":
                        # we don't save intermediate pkl files 
                        continue
                    elif args.save_format == "ply":
                        scene.save(iteration)
                    else:
                        raise ValueError(f"Invalid save format {args.save_format}")
                    symlink(os.path.join("..", "cfg_args"), 
                            os.path.abspath(os.path.join(scene.model_path, "cfg_args")))


                if dataset.timed:
                    torch.cuda.synchronize()
                iter_start = time.time()

                if iteration <=opt.prune_until_iter:
                    gaussians.add_influence_stats(render_pkg["influence"])

                if iteration>opt.prune_from_iter and iteration<=opt.prune_until_iter and iteration % opt.prune_interval == 0:
                    out = gaussians.infl_accum/gaussians.infl_denom
                    out[out.isnan()] = 0.0

                # Gaussian Densification
                if iteration <= (np.ceil(opt.densify_until_epoch*n_cams*opt.iterations)) and iteration>(opt.calc_dense_stats*n_cams):
                    # Track max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], 
                                                                         radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    # Wait for stats accumulation (at least 2 training epochs) before densifying
                    densify_from_epoch = max(opt.calc_dense_stats+2, opt.densify_from_epoch)

                    if iteration > (densify_from_epoch*n_cams) and iteration % (opt.densification_interval*n_cams) == 0:
                        size_threshold = opt.size_threshold if iteration > (opt.opacity_reset_interval*n_cams) else None
                        if frame_idx == 1:
                            # Standard densification for first frame
                            gaussians.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity, scene.cameras_extent, size_threshold)
                        else:
                            # Dynamic densification for subsequent frames
                            gaussians.densify_dynamic(opt.densify_grad_threshold, opt.min_opacity, scene.cameras_extent, opt.size_threshold)
                    
                    # Periodic opacity reset
                    if iteration % (opt.opacity_reset_interval*n_cams) == 0 or \
                        (dataset.white_background and iteration == (densify_from_epoch*n_cams)):
                        gaussians.reset_opacity()

                    if enable_debug:
                        print(f'DEBUG ({iteration}): densification done')

                # Pruning
                if iteration>opt.prune_from_iter and iteration<=opt.prune_until_iter and iteration % opt.prune_interval == 0:
                    gaussians.influence_prune(opt.prune_threshold)

                    if enable_debug:
                        print(f'DEBUG ({iteration}): pruning done')

                if dataset.timed:
                    torch.cuda.synchronize()
                frame_time += time.time()-iter_start
                frame_time_io += time.time()-iter_start
                with torch.no_grad():
                    if (opt.iterations - iteration) < (2*n_cams): # Save most recent render for final epochs
                        viewpoint_cam.prev_rendered = render_pkg["render"].detach()

                    if (opt.iterations - iteration)<(n_cams) and cam_idx == 0 and (dataset.log_images or dataset.log_compressed or dataset.log_ply):
                        
                        if dataset.log_images:
                            save_image(gt_image,os.path.join(scene.model_path, "gt.png"))

                        if dataset.log_ply:
                            scene.save(iteration, save_point_cloud=True)
                        
                        if dataset.log_compressed:
                            if frame_idx == 1:
                                scene.save(frame_idx, save_point_cloud=True)                                

                        if frame_idx>1 and (dataset.adaptive_render and dataset.adaptive_update_period>0.0) and dataset.update_mask!="none":
                            torchvision.utils.save_image(train_cameras[cam_idx].mask.unsqueeze(0)*gt_image,
                                                         os.path.join(scene.model_path, "mask.png"))
                            torchvision.utils.save_image(train_cameras[cam_idx].orig_mask.unsqueeze(0)*gt_image,
                                                         os.path.join(scene.model_path, "orig_mask.png"))
                        video_camera = video_cameras[frame_idx-1]
                        spiral_img = render(video_camera, gaussians, pipe, background)["render"]
                        if frame_idx == 1:
                            os.makedirs(os.path.join(dataset.model_path,"spiral"), exist_ok=True)
                        save_image(torch.clip(spiral_img, 0.0, 1.0),os.path.join(dataset.model_path, "spiral", f"{str(dataset.start_idx + frame_idx).zfill(4)}.png"))

                        if frame_idx == 1:
                            with torch.no_grad():
                                render_pkg = render_mask(viewpoint_cam, gaussians, pipe, bg, image_shape=gt_image.shape, 
                                                         color_mask=color_rw_mask, render_depth=True)
                                pred_depth = render_pkg["depth"]
                                render_depth = pred_depth.detach().cpu().numpy()

                            if opt.lambda_depth>0.0 or dataset.depth_init:
                                gt_depth = viewpoint_cam.gt_depth
                                gt_depth = gt_depth.detach().cpu().numpy()
                                gt_depth = (gt_depth-gt_depth.min())/(gt_depth.max()-gt_depth.min())
                                render_depth = (render_depth-render_depth.min())/(render_depth.max()-render_depth.min())
                                depth_ssim = ssim(torch.tensor(render_depth).cuda().unsqueeze(0), torch.tensor(gt_depth).cuda().unsqueeze(0)).item()
                                depth_psnr = psnr(torch.tensor(gt_depth).cuda().unsqueeze(0), torch.tensor(render_depth).cuda().unsqueeze(0)).item()
                                if wandb_enabled:
                                    wandb.run.summary["depth_SSIM"] = depth_ssim
                                    wandb.run.summary["depth_PSNR"] = depth_psnr
                                depth_err = np.abs(render_depth-gt_depth)
                                depth_err = torch.abs(render_pkg["depth"]-viewpoint_cam.gt_depth).detach().cpu().numpy()
                                torchvision.utils.save_image(torch.tensor(depth_err).unsqueeze(0),os.path.join(dataset.model_path,'err_depth_gray.png'))

                # Optimizer step
                if dataset.timed:
                    torch.cuda.synchronize()
                iter_start = time.time()
                if iteration <= opt.iterations:
                    # gaussians.update_grads()
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    if gaussians.gate_atts is not None and gaussians.gate_atts.training:
                        gaussians.gate_atts.step()
                        gaussians.gate_atts.clamp_params()
                if dataset.timed:
                    torch.cuda.synchronize()
                frame_time += time.time()-iter_start
                frame_time_io += time.time()-iter_start
                if enable_debug:
                    print(f'DEBUG ({iteration}): Optimizer step done')
        # end training loop for this frame

        if dataset.timed:
            torch.cuda.synchronize()
        frame_start = time.time()

        if dataset.log_compressed:
            if frame_idx == 1:
                scene.save(frame_idx, save_point_cloud=True)
            else: 
                gaussians.gate_atts.eval()
                scene.save_compressed(-1, qp)
                gaussians.gate_atts.train()


        # Update previous frame's attributes and latents for next frame's residual encoding
        if frame_idx != n_frames:
            # Used for residual encoding of next frame
            with torch.no_grad():
                for att_name in gaussians.get_atts:
                    prev_atts = gaussians.get_decoded_atts[att_name].clone()
                    prev_latents = gaussians.get_atts[att_name].clone()
                    gaussians.prev_atts[att_name] = prev_atts
                    gaussians.prev_latents[att_name] = prev_latents
                    gaussians.prev_atts[att_name].requires_grad_(False)
                    gaussians.prev_latents[att_name].requires_grad_(False)
                    gaussians.prev_atts_initial[att_name] = prev_atts.clone()
            cur_frame_views = next_frame_views
            cur_train_images = next_train_images
            cur_train_paths = next_train_paths
            prev_xyz = gaussians._xyz.clone()

        if dataset.timed:
            torch.cuda.synchronize()
        frame_time += time.time()-frame_start
        frame_time_io += time.time()-frame_start

        # Collect frame metrics for logging
        if test_image_dataset.n_cams > 0:
            frame_metrics = {
                "Frame index": frame_idx,
                "Loss": round(ema_loss_for_log,5),
                "Loss (Test)": round(metrics['test']['loss'].item(),5),
                "Loss (Val)": round(metrics['val']['loss'].item(),5),
                "Num points": gaussians._xyz.shape[0],
                "Update points": f"{torch.count_nonzero(gaussians.mask_xyz)}" \
                                    if frame_idx>1 else f"{gaussians._xyz.shape[0]}",
                "Size (MB)": round(cur_size,2),
                "PSNR (Test)": round(metrics['test']['psnr'].item(),2),
                "PSNR (Val)": round(metrics['val']['psnr'].item(),2),
                "Frame time": round(frame_time,2),
                "Frame time IO": round(frame_time_io,2),
                "Training time elapsed": round(net_elapsed_time,2),
            }
        else:
            # Not using test cameras
            frame_metrics = {
                "Frame index": frame_idx,
                "Loss": round(ema_loss_for_log,5),
                "Loss (Val)": round(metrics['val']['loss'].item(),5),
                "Num points": gaussians._xyz.shape[0],
                "Update points": f"{torch.count_nonzero(gaussians.mask_xyz)}" \
                                    if frame_idx>1 else f"{gaussians._xyz.shape[0]}",
                "Size (MB)": round(cur_size,2),
                "PSNR (Val)": round(metrics['val']['psnr'].item(),2),
                "Frame time": round(frame_time,2),
                "Frame time IO": round(frame_time_io,2),
                "Training time elapsed": round(net_elapsed_time,2),
            }

        training_metrics.append(frame_metrics)
        
        # Log to wandb if enabled
        if wandb_enabled:
            wandb.log({
                "frame/test/loss_viewpoint/psnr": metrics['test']['psnr'].item(),
                       "frame/test/loss_viewpoint/loss": metrics['test']['loss'].item(),
                       "frame/val/loss_viewpoint/psnr": metrics['val']['psnr'].item(),
                       "frame/val/loss_viewpoint/loss": metrics['val']['loss'].item(),
                       "frame/size": cur_size,
                       "frame/num_points": gaussians._xyz.shape[0],
                       "frame/update_points": torch.count_nonzero(gaussians.mask_xyz) \
                                              if frame_idx>1 else gaussians._xyz.shape[0],
                       "frame/iter_time": frame_time,
                       "frame/iter_time_io": frame_time_io,
                       "frame/elapsed": net_elapsed_time,
                       "frame/num_iterations": opt.iterations if frame_idx>1 else 0,
                       "frame_idx": frame_idx})


        # Compute and display average metrics
        if test_image_dataset.n_cams > 0:
            avg_metrics = {
                "Loss (Test)": round(sum([fm["Loss (Test)"] for fm in training_metrics])/len(training_metrics),5),
                "Loss (Val)": round(sum([fm["Loss (Val)"] for fm in training_metrics])/len(training_metrics),5),
                "PSNR (Test)": round(sum([fm["PSNR (Test)"] for fm in training_metrics])/len(training_metrics),2),
                "PSNR (Val)": round(sum([fm["PSNR (Val)"] for fm in training_metrics])/len(training_metrics),2),
                "Size (MB)": round(sum([fm["Size (MB)"] for fm in training_metrics])),
                "Frame time": round(sum([fm["Frame time"] for fm in training_metrics])/len(training_metrics),2),
                "Elapsed time": round(frame_metrics["Training time elapsed"],2),
            }
        else:
            avg_metrics = {
                "Loss (Val)": round(sum([fm["Loss (Val)"] for fm in training_metrics])/len(training_metrics),5),
                "PSNR (Val)": round(sum([fm["PSNR (Val)"] for fm in training_metrics])/len(training_metrics),2),
                "Size (MB)": round(sum([fm["Size (MB)"] for fm in training_metrics])),
                "Frame time": round(sum([fm["Frame time"] for fm in training_metrics])/len(training_metrics),2),
                "Elapsed time": round(frame_metrics["Training time elapsed"],2),
            }

        # Update progress display
        del frame_metrics["Training time elapsed"]
        if enable_tqdm:
            progress_bar_frame.set_postfix(frame_metrics)
            progress_bar_frame.update(1)
        else:
            frame_counter += 1
            print(f"frame {frame_counter} frame_metrics: {frame_metrics}")

        # End frame index loop
          
    with open(os.path.join(args.model_path,'training_metrics.json'),'w') as f:
        json.dump(training_metrics, f, indent=4) 

    with open(os.path.join(args.model_path, 'avg_metrics.json'),'w') as f:
        json.dump(avg_metrics, f) 

    if enable_tqdm:
        progress_bar_frame.close()

    # Display final results
    print('\nFinal average training metrics:')
    for k,v in avg_metrics.items():
        print(k+":"+ str(v))

    # Log final metrics to wandb
    if wandb_enabled:
        wandb.run.summary.update({
            'average/test/loss_viewpoint/psnr': avg_metrics.get("PSNR (Test)", 0),
            'average/test/loss_viewpoint/loss': avg_metrics.get("Loss (Test)", 0),
            'average/val/loss_viewpoint/psnr': avg_metrics["PSNR (Val)"],
            'average/val/loss_viewpoint/loss': avg_metrics["Loss (Val)"],
            'average/size': avg_metrics["Size (MB)"],
            'average/frame_time': avg_metrics["Frame time"],
            'average/elapsed_time': avg_metrics["Elapsed time"]
        })

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")


    # Create wandb logger
    if WANDB_FOUND and args.use_wandb:
        wandb_project = args.wandb_project
        wandb_run_name = args.wandb_run_name
        wandb_entity = args.wandb_entity
        wandb_mode = args.wandb_mode
        id = hashlib.md5(wandb_run_name.encode('utf-8')).hexdigest()
        name = os.path.basename(args.model_path) if wandb_run_name is None else wandb_run_name
        wandb.init(
            project=wandb_project,
            name=name,
            entity=wandb_entity,
            config=args,
            sync_tensorboard=False,
            dir=args.model_path,
            mode=wandb_mode,
            id=id,
            resume=True
        )

    return tb_writer

def training_report(tb_writer, wandb_enabled, model_args, frame_idx, iteration, Ll1, loss, l1_loss, size, 
                    elapsed, is_test, scene : Scene, renderFunc, renderArgs, prev_report=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('elapsed', elapsed, iteration)
        tb_writer.add_scalar('size', size, iteration)

    if wandb_enabled and frame_idx<=2:
        frame_str = f"{str(frame_idx).zfill(4)}"
        iter_metric = "iter_"+frame_str
        frame_str = "frame_"+frame_str

        # Log iterwise metrics only for the first few frames
        if iteration % model_args.wandb_log_interval == 0:

            wandb.log({frame_str+"/train_loss_patches/l1_loss": Ll1.item(), 
                       frame_str+"/train_loss_patches/total_loss": loss.item(), 
                       frame_str+"/num_points": scene.gaussians.get_xyz.shape[0],
                       frame_str+"/update_points": f"{torch.count_nonzero(scene.gaussians.mask_xyz)}",
                       frame_str+"/elapsed": elapsed,
                       frame_str+"/size": size,
                       iter_metric: iteration
                       })
        
    # Report test and samples of training set
    if is_test:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'val', 'cameras' : scene.getTrainCameras()[0:1]})  # hack: hardcoded val views indices

        report = {}
        for config in validation_configs:
            metrics = {}
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                if model_args.log_images: 
                    os.makedirs(os.path.join(model_args.model_path,config['name'],"gt"),exist_ok=True)
                    os.makedirs(os.path.join(model_args.model_path,config['name'],"renders"),exist_ok=True)
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image, 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), 
                                             image[None], global_step=iteration)
                        if prev_report is None: # First time logging
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), 
                                                 gt_image[None], global_step=iteration)
                    if wandb_enabled and model_args.wandb_log_images and frame_idx <= 2:
                        if prev_report is None: # First time logging
                            wandb.log({frame_str+"/"+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name): 
                                       wandb.Image(gt_image[None].detach().cpu().numpy(), caption="ground_truth"),
                                       iter_metric: iteration}
                                    )
                        wandb.log({frame_str+"/"+config['name'] + "_view_{}/render".format(viewpoint.image_name): 
                                    wandb.Image(image[None].detach().cpu().numpy(), caption="render"),
                                    iter_metric: iteration}
                                    )
                    if model_args.log_images:
                        # Not logging GTs
                        os.makedirs(os.path.join(model_args.model_path,config['name'],"renders", 
                                                 camName_from_Path(viewpoint.image_path)),exist_ok=True)
                        if prev_report is None:
                            os.makedirs(os.path.join(model_args.model_path,config['name'],"gt", 
                                                     camName_from_Path(viewpoint.image_path)),exist_ok=True)
                            if os.path.exists(os.path.join(
                                                model_args.model_path,config['name'],"gt", 
                                                camName_from_Path(viewpoint.image_path),str(model_args.start_idx+frame_idx).zfill(4)+".png"
                                                )
                                            ):
                                os.remove(os.path.join(model_args.model_path,config['name'],"gt", 
                                                       camName_from_Path(viewpoint.image_path),str(model_args.start_idx+frame_idx).zfill(4)+".png"))
                            os.symlink(viewpoint.image_path,os.path.join(model_args.model_path,config['name'],"gt", 
                                                                         camName_from_Path(viewpoint.image_path),str(model_args.start_idx+frame_idx).zfill(4)+".png"))

                        save_image(image,os.path.join(model_args.model_path,config['name'],"renders", 
                                                      camName_from_Path(viewpoint.image_path),str(model_args.start_idx+frame_idx).zfill(4)+".png"))


                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                metrics['l1'] = l1_test
                metrics['psnr'] = psnr_test
                report[config['name']] = metrics

                if wandb_enabled and frame_idx <= 2:
                    wandb.log({frame_str+"/"+config["name"]+"/loss_viewpoint/l1_loss": l1_test, 
                               frame_str+"/"+config["name"]+"/loss_viewpoint/psnr": psnr_test,
                               iter_metric: iteration})

        report['iteration'] = iteration
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
        return report
    else:
        return None

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
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_format", type=str, default='ply')
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--use_xyz_legacy', action='store_true', default=False, help='If set, use legacy xyz decoding in GaussianModel (_xyz_legacy) to reproduce paper numbers. To save compressed pkl\'s, leave unset or set to False. Default: False (use _xyz_fixed).')
    args = parser.parse_args(sys.argv[1:])
    # args.save_iterations.append(args.iterations)
    
    # Merge optimization args for initial and rest and change accordingly
    op = OptimizationParams(op_i.extract(args), op_r.extract(args))

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    lp_args = lp.extract(args)
    pp_args = pp.extract(args)
    qp_args = qp.extract(args)

    # Check for incompatible options
    if args.use_xyz_legacy and getattr(lp_args, 'log_compressed', False):
        print('Error: must use xyz_fixed with log_compressed (do not use --use-xyz-legacy with --log_compressed)')
        sys.exit(1)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp_args, op, pp_args, qp_args, args.test_iterations, args.save_iterations, 
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    # All done
    print("\nTraining complete.")
