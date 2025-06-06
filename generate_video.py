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

from pathlib import Path
import os
import torch
import sys
from argparse import ArgumentParser

def do_system(arg):
    print(f"==== running: {arg}")
    err=os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)

def generate(model_paths):
    print("")

    for scene_dir in model_paths:
        print("Scene:", scene_dir)
        test_dir = Path(scene_dir) / "test"

        gt_dir = test_dir / "gt"
        renders_dir = test_dir / "renders"

        for cam_name in os.listdir(gt_dir):
            print("Camera: ", cam_name)
            gt_cam_dir = gt_dir / cam_name
            renders_cam_dir = renders_dir / cam_name
            print

            cmd = 'ffmpeg -y -thread_queue_size 1024 -i %s/%%04d.png -i %s/%%04d.png -filter_complex hstack=inputs=2 -vb 20M -pix_fmt yuv420p %s/output.mp4' % (gt_cam_dir, renders_cam_dir, scene_dir)
            print(cmd)
            do_system(cmd)

        test_dir = Path(scene_dir) / "val"

        gt_dir = test_dir / "gt"
        renders_dir = test_dir / "renders"

        for cam_name in os.listdir(gt_dir):
            print("Camera: ", cam_name)
            gt_cam_dir = gt_dir / cam_name
            renders_cam_dir = renders_dir / cam_name
            print

            cmd = 'ffmpeg -y -thread_queue_size 1024 -i %s/%%04d.png -i %s/%%04d.png -filter_complex hstack=inputs=2 -vb 20M -pix_fmt yuv420p %s/output_val.mp4' % (gt_cam_dir, renders_cam_dir, scene_dir)
            print(cmd)
            do_system(cmd)

if __name__ == "__main__":
    # device = torch.device("cuda:0")
    # torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    generate(args.model_paths)
