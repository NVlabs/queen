# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import glob
import wandb
import glob
import pandas as pd
import seaborn as sns
from pandas.api.types import is_numeric_dtype
import datetime
import json

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


def get_history(run, key):
    history = run.scan_history()
    metric = [row[key] for row in history]
    metric = [m for m in metric if m is not None]
    metric = np.array(metric)
    metric = metric[~np.isnan(metric)]
    return metric

api = wandb.Api()
wandb_entity = 'quantgsstream'
dynerf = False
if dynerf:
    wandb_project = 'dynerf'
    wandb_tag = 'dynerf_allrerun'
else:
    wandb_project = 'immersive'
    wandb_tag = 'immersiveall_final'
runs = api.runs(os.path.join(wandb_entity, wandb_project))

if dynerf:
    runs = [run for run in runs if (run.config["wandb_tags"]==wandb_tag and 
                                    run.config["adaptive_update_period"]==0.3)]
else:
    runs = [run for run in runs if (run.config["wandb_tags"]==wandb_tag)]
print(len(runs))
plt.rcParams.update({'font.size': 16})
results = {}
for run in runs:
    psnr = get_history(run, "frame/test/loss_viewpoint/psnr")
    size = get_history(run, "frame/size")
    time = get_history(run, "frame/iter_time_io")
    scene = os.path.basename(run.config["source_path"])
    print(scene)
    results[scene] = {"psnr": list(psnr), "size": list(size), "time": list(time)}

    with open(f"output/plots/{scene}_framewise_metrics.json", 'r') as f:
        gt_metrics = json.load(f)

    test_l1_err = gt_metrics["test_l1_err"]
    plt.clf()
    fig, ax1 = plt.subplots() 
    ax_twin = ax1.twinx()
    size_h = ax1.plot(size[1:], label="Model Size", color="red")
    l1_h = ax_twin.plot(test_l1_err, label="Frame Difference", color="blue")
    plt.xlabel("Frame Index")
    ax1.set_ylabel("Model Size (MB)")
    ax_twin.set_ylabel("Consecutive Frame Difference")
    ax1.legend(size_h, ("Model Size",), loc='upper left')
    ax_twin.legend(l1_h, ("Frame Difference",), loc='upper right')
    ax_twin.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.title(f"{' '.join(scene.split('_')[:2]).title()}")
    plt.savefig(f"output/plots/framewise/{scene}_size_vs_gt_l1.png", bbox_inches='tight')
    plt.savefig(f"output/plots/framewise/{scene}_size_vs_gt_l1.pdf", bbox_inches='tight')

    plt.clf()
    fig, ax1 = plt.subplots() 
    ax_twin = ax1.twinx()
    size_h = ax1.plot(time[1:], label="Iter Time", color="red")
    l1_h = ax_twin.plot(test_l1_err, label="Frame Difference", color="blue")
    plt.xlabel("Frame Index")
    ax1.set_ylabel("Iter Time (s)")
    ax_twin.set_ylabel("Consecutive Frame Difference")
    ax_twin.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax1.legend(size_h, ("Iter Time",), loc='upper left')
    ax_twin.legend(l1_h, ("Frame Difference",), loc='upper right')
    plt.title(f"{' '.join(scene.split('_')[:2]).title()}")
    plt.savefig(f"output/plots/framewise/{scene}_time_vs_gt_l1.png", bbox_inches='tight')
    plt.savefig(f"output/plots/framewise/{scene}_time_vs_gt_l1.pdf", bbox_inches='tight')

    plt.clf()
    fig, ax1 = plt.subplots() 
    ax_twin = ax1.twinx()
    size_h = ax1.plot(psnr[1:], label="PSNR", color="red")
    l1_h = ax_twin.plot(test_l1_err, label="Frame Difference", color="blue")
    plt.xlabel("Frame Index")
    ax1.set_ylabel("PSNR (dB)")
    ax_twin.set_ylabel("Consecutive Frame Difference")
    ax_twin.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax1.legend(size_h, ("PSNR",), loc='upper left')
    ax_twin.legend(l1_h, ("Frame Difference",), loc='upper right')
    plt.title(f"{' '.join(scene.split('_')[:2]).title()}")
    plt.savefig(f"output/plots/framewise/{scene}_psnr_vs_gt_l1.png", bbox_inches='tight')
    plt.savefig(f"output/plots/framewise/{scene}_psnr_vs_gt_l1.pdf", bbox_inches='tight')

    # plt.plot
if dynerf:
    with open("output/plots/framewise_dynerf.json","w") as f:
        json.dump(results, f)
else:
    with open("output/plots/framewise_immersive.json","w") as f:
        json.dump(results, f)