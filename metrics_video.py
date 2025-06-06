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
import hashlib
import wandb
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from collections import defaultdict


def readFrames(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in sorted(os.listdir(renders_dir)):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths, wandb_project):

    full_dict = {}
    per_view_dict = {}
    per_frame_dict = {}
    print("")

    for scene_dir in model_paths:
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        per_frame_dict[scene_dir] = {}

        test_dir = Path(scene_dir) / "test"

        gt_dir = test_dir / "gt"
        renders_dir = test_dir / "renders"

        for cam_name in os.listdir(gt_dir):
            print("Camera: ", cam_name)
            gt_cam_dir = gt_dir / cam_name
            renders_cam_dir = renders_dir / cam_name

            per_view_dict[scene_dir][cam_name] = {}
            per_frame_dict[scene_dir][cam_name] = {}

            renders, gts, image_names = readFrames(renders_cam_dir, gt_cam_dir)
            # renders, gts, image_names = renders[:100], gts[:100], image_names[:100]

            ssims = []
            psnrs = []
            lpipss = []
            progress_bar = tqdm(range(len(renders)), desc="Metric evaluation progress")
            for idx in range(len(renders)):
                metrics = {
                            'SSIM': ssim(renders[idx], gts[idx]), 
                            'PSNR': psnr(renders[idx], gts[idx]), 
                            'LPIPS': lpips(renders[idx], gts[idx], net_type='vgg')
                }
                ssims.append(metrics['SSIM'])
                psnrs.append(metrics['PSNR'])
                lpipss.append(metrics['LPIPS'])

                progress_bar.set_postfix({k:v.item() for k,v in metrics.items()})
                progress_bar.update(1)
            progress_bar.close()

            print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
            print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
            print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
            print("")

            cur_frame_dict = {  "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}
                                }
            avg_frame_dict = {"SSIM": torch.tensor(ssims).mean().item(),
                                "PSNR": torch.tensor(psnrs).mean().item(),
                                "LPIPS": torch.tensor(lpipss).mean().item()}
            per_frame_dict[scene_dir][cam_name].update(cur_frame_dict)
            per_view_dict[scene_dir][cam_name].update(avg_frame_dict)
            
        

        avg_view_dict = defaultdict(lambda:[])
        for cam_name in per_view_dict[scene_dir]:
            for metric in per_view_dict[scene_dir][cam_name]:
                avg_view_dict[metric] += [per_view_dict[scene_dir][cam_name][metric]]
        
        avg_view_dict = {k:sum(v)/len(v) for k,v in avg_view_dict.items()}

        full_dict[scene_dir] = avg_view_dict

        with open(scene_dir + "/results.json", 'w') as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
        with open(scene_dir + "/per_view.json", 'w') as fp:
            json.dump(per_view_dict[scene_dir], fp, indent=True)
        with open(scene_dir + "/per_frame.json", 'w') as fp:
            json.dump(per_frame_dict[scene_dir], fp, indent=True)

        if len(per_view_dict[scene_dir])>1:
            '\nAverage:'
            print("  SSIM : {:>12.7f}".format(avg_view_dict["SSIM"], ".5"))
            print("  PSNR : {:>12.7f}".format(avg_view_dict["PSNR"], ".5"))
            print("  LPIPS: {:>12.7f}".format(avg_view_dict["LPIPS"], ".5"))
            print("")
        else:
            if wandb_project:
                wandb_run_name = scene_dir.strip('/').split('/')[-1]
                wandb_entity = 'nvr-amri'
                id = hashlib.md5(wandb_run_name.encode('utf-8')).hexdigest()
                name = wandb_run_name

                api = wandb.Api()

                run = api.run(f"{wandb_entity}/{wandb_project}/{id}")
                run.summary['PSNR'] = avg_view_dict['PSNR']
                run.summary['SSIM'] = avg_view_dict['SSIM']
                run.summary['LPIPS'] = avg_view_dict['LPIPS']
                run.summary.update()

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--wandb_project', '-w', required=False, type=str, default="")
    args = parser.parse_args()
    evaluate(args.model_paths, args.wandb_project)
