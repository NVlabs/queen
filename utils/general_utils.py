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
import math
import sys
from datetime import datetime
import numpy as np
import random

def kthvalue(input, k, dim):
    sorted, _ = torch.sort(input, dim=dim)
    return torch.take_along_dim(sorted, k, dim=dim)

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

class DecayScheduler(object):
    '''A simple class for decaying schedule of various hyperparameters.'''

    def __init__(self, total_steps, decay_name='fix', start=0, end=0, params=None):
        self.decay_name = decay_name
        self.start = start
        self.end = end
        self.total_steps = total_steps
        self.params = params

    def __call__(self, step):
        if self.decay_name == 'fix':
            return self.start
        elif self.decay_name == 'linear':
            if step>self.total_steps:
                return self.end
            return self.start + (self.end - self.start) * step / self.total_steps
        elif self.decay_name == 'exp':
            if step>self.total_steps:
                return self.end
            return max(self.end, self.start*(np.exp(-np.log(1/self.params['temperature'])*step/self.total_steps/self.params['decay_period'])))
            # return self.start * (self.end / self.start) ** (step / self.total_steps)
        elif self.decay_name == 'inv_sqrt':
            return self.start * (self.total_steps / (self.total_steps + step)) ** 0.5
        elif self.decay_name == 'cosine':
            if step>self.total_steps:
                return self.end
            return self.end + 0.5 * (self.start - self.end) * (1 + math.cos(step / self.total_steps * math.pi))
        else:
            raise ValueError('Unknown decay name: {}'.format(self.decay_name))
   
def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))



def warp_depth(pred_depth,gt_depth):
    from utils.warp import MonotonicNN

    model = MonotonicNN(1, 10).to(pred_depth)
    iterations = 20000
    lr = 1.0e-2
    loss = torch.nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0, nesterov=True, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations/100, eta_min=5.0e-3)

    from tqdm import tqdm
    import copy
    pbar = tqdm(range(1, iterations+1), desc="Training progress")
    best_loss = 100
    for iteration in range(iterations):
        with torch.no_grad():
            input, output = pred_depth.unsqueeze(-1).detach(), gt_depth.unsqueeze(-1).detach()
        warp_depth = model(input)
        l1loss = torch.abs(warp_depth-output).mean()
        l1loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if iteration%100 == 0:
            scheduler.step()

        if l1loss.item() < best_loss:
            best_dict = copy.deepcopy(model.state_dict())
            best_loss = l1loss.item()
        if iteration % 1000 == 0 or iteration == iterations-1:
            pbar.set_postfix({"Loss":l1loss.item(), "Best Loss": best_loss, "LR": optimizer.param_groups[0]["lr"]})
            pbar.update(1000)
    pbar.close()
    model.load_state_dict(best_dict)
    x = torch.linspace(0, 100, 100).to(pred_depth).unsqueeze(-1)
    y = model(x)
    import matplotlib.pyplot as plt
    plt.plot(x.flatten().detach().cpu().numpy(),y.flatten().detach().cpu().numpy())
    plt.show()
    plt.grid(True)
    plt.savefig('./output/viz_flow/warp.png')
    return model
