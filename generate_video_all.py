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
# 

from pathlib import Path
import os
import torch
import sys
from argparse import ArgumentParser

def symlink(src, dest):
    if not os.path.exists(src):
        return
    if os.path.islink(dest):
        os.remove(dest)
    os.symlink(src,dest)

def do_system(arg):
    print(f"==== running: {arg}")
    err=os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)

def generate(model_paths, num_frames):
    print("")

    for scene_dir in model_paths:
        scene_dir = os.path.abspath(scene_dir)
        print("Scene:", scene_dir)
        os.makedirs(Path(scene_dir) / "render", exist_ok=True)
        os.makedirs(Path(scene_dir) / "gt", exist_ok=True)
        os.makedirs(Path(scene_dir) / "flow", exist_ok=True)
        os.makedirs(Path(scene_dir) / "err", exist_ok=True)
        os.makedirs(Path(scene_dir) / "mask", exist_ok=True)
        # os.makedirs(Path(scene_dir) / "warp", exist_ok=True)
        for frame_idx in range(1,num_frames+1):
            frame_dir = Path(scene_dir) / "frames" / str(frame_idx).zfill(4)
            # symlink(frame_dir / "warped.png", os.path.join(scene_dir, "warp", str(frame_idx).zfill(4)+'.png'))
            if os.path.exists(frame_dir / "mask.png"):
                os.makedirs(Path(scene_dir) / "mask", exist_ok=True)
                symlink(frame_dir / "mask.png", os.path.join(scene_dir, "mask", str(frame_idx).zfill(4)+'.png'))
            if os.path.exists(frame_dir / "orig_mask.png"):
                os.makedirs(Path(scene_dir) / "orig_mask", exist_ok=True)
                symlink(frame_dir / "orig_mask.png", os.path.join(scene_dir, "orig_mask", str(frame_idx).zfill(4)+'.png'))
            if os.path.exists(frame_dir / "gt.png"):
                os.makedirs(Path(scene_dir) / "gt", exist_ok=True)
                symlink(frame_dir / "gt.png", os.path.join(scene_dir, "gt", str(frame_idx).zfill(4)+'.png'))
            if os.path.exists(frame_dir / "flow.png"):
                os.makedirs(Path(scene_dir) / "flow", exist_ok=True)
                symlink(frame_dir / "flow.png", os.path.join(scene_dir, "flow", str(frame_idx).zfill(4)+'.png'))
            if os.path.exists(frame_dir / "err.png"):
                os.makedirs(Path(scene_dir) / "err", exist_ok=True)
                symlink(frame_dir / "err.png", os.path.join(scene_dir, "err", str(frame_idx).zfill(4)+'.png'))
            if os.path.exists(frame_dir / "rendered.png"):
                os.makedirs(Path(scene_dir) / "render", exist_ok=True)
                symlink(frame_dir / "rendered.png", os.path.join(scene_dir, "render", str(frame_idx).zfill(4)+'.png'))
            
        
        # if os.path.exists(os.path.join(scene_dir,"gt")) and os.path.exists(os.path.join(scene_dir,"render")):
        #     cmd = 'ffmpeg -y -thread_queue_size 1024 -pattern_type glob -i "%s/*.png" -pattern_type glob -i "%s/*.png" -filter_complex hstack=inputs=2 -vb 20M %s/output_render.mp4'\
        #         % (os.path.join(scene_dir,"gt"), os.path.join(scene_dir,"render"),  scene_dir)
        #     print(cmd)
        #     do_system(cmd)
        if os.path.exists(os.path.join(scene_dir,"orig_mask")) and os.path.exists(os.path.join(scene_dir,"mask")):
            cmd = 'ffmpeg -y -thread_queue_size 1024 -pattern_type glob -i "%s/*.png" -pattern_type glob -i "%s/*.png" -filter_complex hstack=inputs=2 -vb 20M %s/output_mask.mp4'\
                % (os.path.join(scene_dir,"orig_mask"), os.path.join(scene_dir,"mask"), scene_dir)
            print(cmd)
            do_system(cmd)
        if os.path.exists(os.path.join(scene_dir,"flow")):
            cmd = 'ffmpeg -y -thread_queue_size 1024 -pattern_type glob -i "%s/*.png" -vb 20M %s/flow.mp4' % (Path(scene_dir) / "flow", scene_dir)
            print(cmd)
            do_system(cmd)

        # cmd = 'ffmpeg -y -thread_queue_size 1024 -pattern_type glob -i "%s/*.png" -vb 20M %s/warp.mp4' % (Path(scene_dir) / "warp", scene_dir)
        # print(cmd)
        # do_system(cmd)


if __name__ == "__main__":
    # device = torch.device("cuda:0")
    # torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--n_frames', '-n',  type=int, default=300)
    args = parser.parse_args()
    generate(args.model_paths, args.n_frames)
