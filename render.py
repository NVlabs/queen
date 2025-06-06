#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.


import torch
from scene import Scene
import os
import yaml
import socket
import sys
from collections import defaultdict
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.system_utils import searchForMaxIteration
from argparse import ArgumentParser
import torch.utils.benchmark as benchmark
from gaussian_renderer import GaussianModel
from utils.loader_utils import MultiViewVideoDataset, SequentialMultiviewSampler
from arguments import ModelParams, PipelineParams, OptimizationParams, QuantizeParams, OptimizationParamsInitial, OptimizationParamsRest, get_combined_args

def render_fn(views, gaussians, pipeline, background, use_amp):
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
        for view in views:
            render(view, gaussians, pipeline, background)

def measure_fps(scene, gaussians, pipeline, background, use_amp=False):
    with torch.no_grad():
        views = scene.getTrainCameras() + scene.getTestCameras()
        t0 = benchmark.Timer(stmt='render_fn(views, gaussians, pipeline, background, use_amp)',
                            setup='from __main__ import render_fn',
                            globals={'views': views, 'gaussians': gaussians, 'pipeline': pipeline, 
                                    'background': background, 'use_amp': use_amp},
                            )
        time = t0.timeit(50)
        fps = len(views)/time.median
        print("Rendering FPS: ", fps)
    return fps

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:04d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:04d}'.format(idx) + ".png"))

def render_sets(dataset: ModelParams, opt: OptimizationParams, pipeline: PipelineParams, qp:QuantizeParams, args,
             skip_train: bool, skip_test: bool):
    
    with torch.no_grad():

        if not skip_train:
            # Create dataset and loader for training and testing at each time instance
            train_image_dataset = MultiViewVideoDataset(dataset.source_path, split='train', test_indices=dataset.test_indices,
                                                        max_frames=dataset.max_frames, start_idx=0)
            train_sampler = SequentialMultiviewSampler(train_image_dataset)
            train_loader = iter(torch.utils.data.DataLoader(train_image_dataset, batch_size=train_image_dataset.n_cams, 
                                                            sampler=train_sampler, num_workers=4))
        
        if not skip_test:
            test_image_dataset = MultiViewVideoDataset(dataset.source_path, split='test', test_indices=dataset.test_indices, 
                                                    max_frames=dataset.max_frames, start_idx=0)
            test_sampler = SequentialMultiviewSampler(test_image_dataset)
            test_loader = iter(torch.utils.data.DataLoader(test_image_dataset, batch_size=test_image_dataset.n_cams, 
                                                            sampler=test_sampler, num_workers=4))
        


        start_frame_idx = dataset.start_idx + 1
        # Fast forward data loading
        for frame_ff in range(0, start_frame_idx):
            if not skip_train:
                train_data = next(train_loader)
                train_images, train_paths = train_data
            if not skip_test:
                try:
                    test_data = next(test_loader)
                    test_images, test_paths = test_data
                except StopIteration:
                    print('No test cameras found, disabling testing.')
                    test_images, test_paths = None, None

        if not skip_train:
            train_image_data = {'image':train_images.cuda(),'path':train_paths,'frame_idx':0}
        else:
            train_image_data = None
        if not skip_test:
            test_image_data = {'image':test_images.cuda(),'path':test_paths,'frame_idx':0}
        else:
            test_image_data = None

        # Create the gaussian model and scene, initialized with frame 1 images from dataset
        qp.seed = dataset.seed
        gaussians = GaussianModel(dataset.sh_degree, qp, dataset)
        scene = Scene(dataset, gaussians, 
                      train_image_data= train_image_data, test_image_data=test_image_data)
        opt.set_params(start_frame_idx)
        # Setup training arguments
        gaussians.training_setup(opt)
        
        gaussians.frame_idx = start_frame_idx
        scene.model_path = os.path.join(args.model_path,'frames',str(start_frame_idx).zfill(4))
        scene.updateCameraImages(args, train_image_data, test_image_data, start_frame_idx, resolution_scales=[1.0])
        scene.loaded_iter = searchForMaxIteration(os.path.join(scene.model_path, "point_cloud"))
        scene.gaussians.load_ply(os.path.join(scene.model_path,
                                                        "point_cloud",
                                                        "iteration_" + str(scene.loaded_iter),
                                                        "point_cloud.ply"))

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            print("Rendering training set frame {}".format(start_frame_idx))
            render_set(scene.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
            print("Rendering test set frame {}".format(start_frame_idx))
            render_set(scene.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

        fps = measure_fps(scene, gaussians, pipeline, background, use_amp=False)

if __name__ == "__main__":

    print('Running on ', socket.gethostname())
    # Config file is used for argument defaults. Command line arguments override config file.
    # testing
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
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    args = parser.parse_args(sys.argv[1:])
    
    # Merge optimization args for initial and rest and change accordingly
    op = OptimizationParams(op_i.extract(args), op_r.extract(args))

    print("Rendering " + args.model_path)
    safe_state(args.quiet)

    lp_args = lp.extract(args)
    pp_args = pp.extract(args)
    qp_args = qp.extract(args)

    render_sets(lp_args, op, pp_args, qp_args, args, args.skip_train, args.skip_test)
