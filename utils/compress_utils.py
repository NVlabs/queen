# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
import torch
import numpy as np
from tqdm import tqdm
from scene.decoders import LatentAutoEncoder, StraightThrough
from utils.bitEstimator import BitEstimator
import os
import sys

class CompressedLatents(object):

    def size(self):
        def get_dict_size(d):
            size = sys.getsizeof(d)
            for key, value in d.items():
                # Add size of key (assuming it's a number)
                size += sys.getsizeof(key)
                # Add size of value (assuming it's a number)
                size += sys.getsizeof(value)
            return size
        
        # compute size of self.cdf
        cdf_size = self.cdf.nbytes
        byte_stream_size = len(self.byte_stream)
        mapping_size = get_dict_size(self.mapping)
        return cdf_size + byte_stream_size + mapping_size
    
    def compress(self, latent, scale=1.0):
        import torchac
        assert latent.dim() == 2, "Latent should be 2D"
        self.num_latents, self.latent_dim = latent.shape
        flattened = latent.flatten()

        # Scale the values to a larger range before rounding
        scaled = flattened * scale
        weight = torch.round(scaled).int()
        unique_vals, counts = torch.unique(weight, return_counts = True)
        probs = counts/torch.sum(counts)
        tail_idx = torch.where(probs <= 1.0e-4)[0]
        tail_vals = unique_vals[tail_idx]
        self.tail_locs = {}
        for val in tail_vals:
            self.tail_locs[val.item()] = torch.where(weight == val)[0].detach().cpu()
            weight[weight == val] = unique_vals[counts.argmax()]
        unique_vals, counts = torch.unique(weight, return_counts = True)
        probs = counts/torch.sum(counts)
        weight = weight.detach().cpu()

        cdf = torch.cumsum(probs,dim=0)
        cdf = torch.cat((torch.Tensor([0.0]).to(cdf),cdf))
        cdf = cdf/cdf[-1:] # Normalize the final cdf value just to keep torchac happy
        cdf = cdf.unsqueeze(0).repeat(flattened.size(0),1)
        
        mapping = {val.item():idx.item() for val,idx in zip(unique_vals,torch.arange(unique_vals.shape[0]))}
        self.mapping = mapping
        weight.apply_(mapping.get)
        byte_stream = torchac.encode_float_cdf(cdf.detach().cpu(), weight.to(torch.int16))
        
        self.byte_stream, self.mapping, self.cdf = byte_stream, mapping, cdf[0].detach().cpu().numpy()

    def uncompress(self, scale=1.0):
        import torchac
        cdf = torch.tensor(self.cdf).unsqueeze(0).repeat(self.num_latents*self.latent_dim,1)
        weight = torchac.decode_float_cdf(cdf, self.byte_stream)
        weight = weight.to(torch.float32)
        inverse_mapping = {v:k for k,v in self.mapping.items()}
        weight.apply_(inverse_mapping.get)
        for val, locs in self.tail_locs.items():
            weight[locs] = val
        # Scale back to original range
        weight = weight / scale
        weight = weight.view(self.num_latents, self.latent_dim)
        return weight

class EntropyLoss:
    def __init__(self, prob_models, lambdas, noise_freq=1):
        self.prob_models = prob_models
        self.lambdas = list(lambdas.values())
        self.noise_freq = noise_freq
        self.start_idx, self.end_idx = [0], []
        for prob_model in prob_models.values():
            self.start_idx += [self.start_idx[-1] + prob_model.num_channels]
            self.end_idx += [self.start_idx[-1]]
        self.net_channels = self.start_idx[-1]
        self.noise = None

    def loss(self, latents, iteration, is_val=False):
        latents = [l for param_name,l in latents.items() if param_name in self.prob_models]
        if len(latents) == 0:
            return 0.0, 0.0
        noise = self.noise
        if not is_val:
            if self.noise_freq == 1:
                noise = torch.rand(latents[0].shape[0],self.net_channels).to(latents[0])-0.5
            elif (iteration-1) % self.noise_freq == 0:
                self.noise = torch.rand(latents[0].shape[0],self.net_channels).to(latents[0])-0.5
                noise = self.noise
        total_bits, total_loss = 0.0, 0.0
        for i,prob_model in enumerate(self.prob_models.values()):
            weight = latents[i] + noise[:,self.start_idx[i]:self.end_idx[i]] if not is_val else torch.round(latents[i])
            weight_p, weight_n = weight + 0.5, weight - 0.5
            prob = prob_model(weight_p) - prob_model(weight_n)
            bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / np.log(2.0), 0, 50))
            total_bits += bits
            total_loss += self.lambdas[i] * bits
        return total_loss / latents[0].shape[0], total_bits / latents[0].shape[0], total_bits


def init_latents(
                latent_args,
                param: torch.Tensor,
                param_name: str,
                lambda_distortion: float = 0.02,
                lr: float = 1.0e-2,
                prob_lr: float = 1.0e-3,
                prob_nlayers: int = 1,
                iterations: int=50000,
                **kwargs
                ):
    

    decoder_args = {}
    for key, value in vars(latent_args).items():
        if key.startswith(param_name):
            decoder_args[key.split(param_name+"_")[-1]] = value
    decoder_args['ldec_std']=0.1
    decoder_args['ldecode_matrix']='sq'  
    assert param.dim() == 2
    feature_dim = param.shape[1]
    latent_dim = decoder_args['latent_dim']
    del decoder_args['latent_dim']
    latent_dim = feature_dim if latent_dim==0 else latent_dim

    if lambda_distortion>0.0:
        prob_model = BitEstimator(latent_dim, num_layers=prob_nlayers).cuda()
        prob_opt = torch.optim.Adam(prob_model.parameters(), lr=prob_lr)
        entloss = EntropyLoss({'feature': prob_model}, {'feature': lambda_distortion})
        
    autoencoder = LatentAutoEncoder(latent_dim, feature_dim, **decoder_args).cuda()
    opt = torch.optim.SGD(autoencoder.parameters(),lr=lr, nesterov=True, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iterations/1000, eta_min=lr/10)
    l2loss = torch.nn.MSELoss()
    best_recon = torch.tensor([1.0e6]).cuda()
    best_state_dict = None

    progress_bar = tqdm(range(1, iterations+1), desc=param_name+" initialization progress")
    for iteration in range(1,iterations+1):
        opt.zero_grad()
        if lambda_distortion>0.0:
            prob_opt.zero_grad()
        latents = autoencoder.encoder(param)
        latents = StraightThrough.apply(latents)
        recon = autoencoder.decoder(latents)
        rate_loss = l2loss(recon,param)
        if lambda_distortion>0.0:
            dist_loss, avg_bits, total_bits = entloss.loss({"feature":latents}, iteration)
        else:
            dist_loss, total_bits = torch.zeros(1).cuda(), torch.zeros(1).cuda()
        loss = rate_loss+dist_loss
        loss.backward()
        opt.step()
        if lambda_distortion>0.0:
            prob_opt.step()

        if rate_loss<best_recon:
            best_recon = rate_loss
            best_state_dict = copy.deepcopy(autoencoder.state_dict())
        if iteration%1000 == 0:
            log_dict = {
                "net_loss": loss.item(),
                "rate_loss": rate_loss.item(),
                "best loss": best_recon.item(),
                "dist_loss": dist_loss.item(),
                "size (MB)": total_bits.item()/8/10**6
            }
            progress_bar.set_postfix(log_dict)
            progress_bar.update(1000)
            scheduler.step()

    autoencoder.load_state_dict(best_state_dict)
    with torch.no_grad():
        return autoencoder.encoder(param), autoencoder.decoder.state_dict()


def search_for_max_iteration(folder):
    saved_iters = []
    for fname in os.listdir(folder):
        if fname.startswith("iteration_"):
            try:
                iter_num = int(fname.split("_")[-1])
                saved_iters.append(iter_num)
            except ValueError:
                continue
    return max(saved_iters) if saved_iters else 0