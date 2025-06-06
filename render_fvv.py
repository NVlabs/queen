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
from utils.system_utils import searchForMaxIteration
from argparse import ArgumentParser
from gaussian_renderer import GaussianModel
from utils.loader_utils import MultiViewVideoDataset, SequentialMultiviewSampler
from scene.decoders import DecoderIdentity
from arguments import ModelParams, PipelineParams, OptimizationParams, QuantizeParams, OptimizationParamsInitial, OptimizationParamsRest, get_combined_args

def do_system(arg):
    print(f"==== running: {arg}")
    err=os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)

def render_fvv(dataset: ModelParams, opt: OptimizationParams, pipeline: PipelineParams, qp:QuantizeParams, args,
             skip_train: bool, skip_test: bool):
    
    with torch.no_grad():

        # Create the gaussian model and scene, initialized with frame 1 images from dataset
        qp.seed = dataset.seed
        gaussians = GaussianModel(dataset.sh_degree, qp, dataset)
        scene = Scene(dataset, gaussians, 
                      train_image_data= None, test_image_data=None, N_video_views=args.num_video_views)
        
        spiral_dir = os.path.join(dataset.model_path, 'spiral_v2')
        os.makedirs(spiral_dir, exist_ok=True)
        cameras = scene.getVideoCameras()
        for start_frame_idx in tqdm(range(1,dataset.max_frames+1), desc="Rendering spiral"):
            gaussians.frame_idx = start_frame_idx
            scene.model_path = os.path.join(args.model_path,'frames',str(start_frame_idx).zfill(4))
            scene.loaded_iter = searchForMaxIteration(os.path.join(scene.model_path, "point_cloud"))
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
                    torchvision.utils.save_image(img, os.path.join(spiral_dir, str(idx).zfill(3)+ ".png"))
                breakpoint()
            else:
                camera = cameras[start_frame_idx-1]
                img = render(camera, gaussians, pipeline, background)["render"]
                torchvision.utils.save_image(img, os.path.join(spiral_dir, str(start_frame_idx).zfill(3)+ ".png"))


        cmd = 'ffmpeg -y -thread_queue_size 1024 -framerate 30 -pattern_type glob -i "%s/*.png" -vb 20M -pix_fmt yuv420p %s/output.mp4' % (spiral_dir, spiral_dir)
        print(cmd)
        do_system(cmd)
            
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
    parser.add_argument("--num_spirals", type=int, default=2)
    parser.add_argument("--num_video_views", type=int, default=300)
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

    render_fvv(lp_args, op, pp_args, qp_args, args, args.skip_train, args.skip_test)


