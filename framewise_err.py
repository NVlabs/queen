# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from utils.loader_utils import MultiViewVideoDataset
import torch
import sys
import json
from argparse import ArgumentParser, Namespace
from utils.loader_utils import SequentialMultiviewSampler, MultiViewVideoDataset

parser = ArgumentParser(description="Training script parameters")
parser.add_argument('-s','--source_path', type=str, default=None)
parser.add_argument('-w','--workers', type=int, default=4)
args = parser.parse_args(sys.argv[1:])
# Create dataset and loader for training and testing at each time instance
max_frames = 300 if "truck" not in args.source_path else 150
train_image_dataset = MultiViewVideoDataset(args.source_path, split='train', test_indices=[0],
                                            max_frames=max_frames, start_idx=0)
test_image_dataset = MultiViewVideoDataset(args.source_path, split='test', test_indices=[0], 
                                            max_frames=max_frames, start_idx=0)

train_sampler = SequentialMultiviewSampler(train_image_dataset)
test_sampler = SequentialMultiviewSampler(test_image_dataset)

train_loader = iter(torch.utils.data.DataLoader(train_image_dataset, batch_size=train_image_dataset.n_cams, 
                                                sampler=train_sampler, num_workers=args.workers))
test_loader = iter(torch.utils.data.DataLoader(test_image_dataset, batch_size=test_image_dataset.n_cams, 
                                                sampler=test_sampler, num_workers=args.workers))


train_data = next(train_loader)
cur_train_images, cur_train_paths = train_data[0].cuda(), train_data[1]
test_data = next(test_loader)
test_images, test_paths = test_data[0].cuda(), test_data[1]
prev_train_images = cur_train_images
prev_test_images = test_images

metrics = {'train_l1_err': [], 'train_mse_err': [], 'train_psnr_err': [],
           'test_l1_err': [], 'test_mse_err': [], 'test_psnr_err': []}
for frame_idx in range(1, max_frames):
    train_data = next(train_loader)
    cur_train_images, cur_train_paths = train_data[0].cuda(), train_data[1]
    test_data = next(test_loader)
    test_images, test_paths = test_data[0].cuda(), test_data[1]

    # Compute framewise error
    train_l1_err = torch.mean(torch.abs(cur_train_images - prev_train_images))
    train_mse_err = torch.mean((cur_train_images - prev_train_images) ** 2)
    train_psnr_err = 20 * torch.log10(1.0 / torch.sqrt(train_mse_err))
    test_l1_err = torch.mean(torch.abs(test_images - prev_test_images))
    test_mse_err = torch.mean((test_images - prev_test_images) ** 2)
    test_psnr_err = 20 * torch.log10(1.0 / torch.sqrt(test_mse_err))

    metrics['train_l1_err'].append(train_l1_err.item())
    metrics['train_mse_err'].append(train_mse_err.item())
    metrics['train_psnr_err'].append(train_psnr_err.item())
    metrics['test_l1_err'].append(test_l1_err.item())
    metrics['test_mse_err'].append(test_mse_err.item())
    metrics['test_psnr_err'].append(test_psnr_err.item())

    prev_train_images = cur_train_images
    prev_test_images = test_images

    sys.stdout.write(f"\r Frame {frame_idx}/{max_frames}")
    sys.stdout.flush()

print("\n")
scene = args.source_path.split('/')[-1]
with open(f'output/plots/{scene}_framewise_metrics.json', 'w') as f:
    json.dump(metrics, f)