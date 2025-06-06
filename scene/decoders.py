# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Decoders for quantized latent representations in Gaussian Splatting.

This module provides various decoder architectures for converting quantized latent
codes back to Gaussian attributes (position, color, scaling, rotation, opacity, flow).
Includes support for identity, linear, and residual decoding.
Also includes a Gate for use in sparse gating of Gaussian attributes.
"""

import math
import copy
import torch
import scipy
import numpy as np
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module, Parameter, init
from torch.nn.modules.utils import _ntuple

# Small epsilon to prevent numerical instability
epsilon = 1e-6

def get_dft_matrix(conv_dim, channels):
    """Generate Discrete Fourier Transform matrix for frequency-domain decoding."""
    dft = torch.zeros(conv_dim,channels)
    for i in range(conv_dim):
        for j in range(channels):
            # Each row of dft is a bias vector
            dft[i,j] = math.cos(torch.pi/channels*(i+0.5)*j)/math.sqrt(channels) 
            dft[i,j] = dft[i,j]*(math.sqrt(2) if j>0 else 1)
    return dft

class StraightThrough(torch.autograd.Function):
    """Straight-through estimator for quantization - rounds in forward, passes gradients unchanged."""

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Atanh(Module):
    """Learnable inverse hyperbolic tangent activation with scaling parameter."""

    def __init__(self, dim):
        super().__init__()
        self.scale = Parameter(torch.ones(1,dim, requires_grad=True))

    def forward(self, x):
        clamped = torch.clamp(x,-1+epsilon,1-epsilon)
        return torch.atanh(clamped)*self.scale
    
    def reset_parameters(self, init_type='constant', param=1.0):
        """Initialize scale parameter with specified method."""
        if init_type == 'normal':
            init.normal_(self.scale, std=param)
        elif init_type == 'uniform':
            init.uniform_(self.scale, -param, param)
        elif init_type == 'constant':
            init.constant_(self.scale, val=param)

    def invert(self, out):
        """Compute inverse transformation."""
        clamped = torch.tanh(out/self.scale)
        return clamped
    
class StraightThroughEps(torch.autograd.Function):
    """Straight-through estimator with epsilon threshold for sparsity."""

    @staticmethod
    def forward(ctx, x, eps=0.001):
        diff = torch.where(torch.abs(x)<eps, x, torch.zeros_like(x))
        y = x - diff
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class StraightThroughFloor(torch.autograd.Function):
    """Straight-through estimator with floor operation in forward pass."""

    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    
class DecoderLayer(Module):
    """Linear decoder layer with optional DFT matrix and learnable scaling/shifting."""

    def __init__(self, in_features: int, out_features: int, ldecode_matrix: str, bias: bool = False) -> None:
        super(DecoderLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.freeze = 'frz' in ldecode_matrix
        
        # Initialize DFT matrix if specified
        if 'dft' in ldecode_matrix:
            self.dft = Parameter(get_dft_matrix(in_features, out_features), requires_grad=False)
        
        # Initialize scaling parameters
        if 'dft' in ldecode_matrix:
            self.scale = Parameter(torch.empty((1,out_features)), requires_grad=not self.freeze)
        else:
            self.scale = Parameter(torch.empty((in_features,out_features)), requires_grad=not self.freeze)
        
        # Initialize bias/shift parameters if requested
        if bias:
            self.shift = Parameter(torch.empty(1,out_features), requires_grad=not self.freeze)
        else:
            self.register_parameter('shift', None)

        self.ldecode_matrix = ldecode_matrix
        
        # Handle fixed DFT case
        if ldecode_matrix == 'dft_fixed':
            self.scale.requires_grad_(False)
            if not bias:
                self.shift.requires_grad_(False)

    def reset_parameters(self, param=1.0, init_type = 'normal') -> None:
        """Initialize layer parameters with specified method and scale."""
        if init_type == 'normal':
            init.normal_(self.scale, std=param)
        elif init_type == 'uniform':
            init.uniform_(self.scale, -param, param)
        elif init_type == 'constant':
            init.constant_(self.scale, val=param)
        if self.shift is not None:
            init.zeros_(self.shift)

    def clamp(self, val: float = 0.5) -> None:
        with torch.no_grad():
            self.scale.clamp_(-val, val)

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass through decoder layer."""
        if 'dft' in self.ldecode_matrix:
            w_out = torch.matmul(input,self.dft)*self.scale+(self.shift if self.shift is not None else 0)
        else:
            w_out = torch.matmul(input,self.scale)+(self.shift if self.shift is not None else 0)
        return w_out

    def invert(self, output: Tensor) -> Tensor:
        """Compute inverse transformation for initialization from target values."""
        shift = self.shift if self.shift is not None else 0
        if self.in_features == 1 and self.out_features == 1:
            # Simple scalar case
            input = (output-shift)/(self.scale+epsilon)
        else:
            batchsize = 500000
            batches = np.ceil(output.shape[0]/batchsize).astype(np.int32)
            input = None
            for batch_idx in range(batches):
                cur_output = output[batch_idx*batchsize:(batch_idx+1)*batchsize]
                if 'dft' in self.ldecode_matrix:
                    cur_input = torch.linalg.lstsq(self.dft.T,((cur_output-shift)/self.scale).T).solution.T
                else:
                    cur_input = torch.linalg.lstsq(self.scale.T,(cur_output-shift).T).solution.T
                if input is None:
                    input = cur_input
                else:
                    input = torch.cat((input,cur_input),dim=0)
        return input
    
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.shift is not None
        )

class LatentDecoder(Module):
    """Multi-layer neural decoder for converting quantized latents to Gaussian attributes."""

    def __init__(
        self,
        latent_dim: int,
        feature_dim: int,
        ldecode_matrix:str,
        use_shift: bool,
        latent_norm: str,
        num_layers_dec:int = 0,
        hidden_dim_dec:int = 0,
        activation:str = 'relu',
        final_activation:str = 'none',
        clamp_weights:float = 0.0,
        ldec_std:float = 1.0,
        **kwargs,
    ) -> None:
        super(LatentDecoder, self).__init__()
        latent_dim = feature_dim if latent_dim == 0 else latent_dim
        self.ldecode_matrix = ldecode_matrix
        self.channels = feature_dim
        self.latent_dim = latent_dim
        
        # Normalization divisor for input latents
        self.div = nn.Parameter(torch.ones(latent_dim),requires_grad=False)
        self.frz_div = None
        self.norm = latent_norm
        
        # Multi-layer decoder configuration
        self.num_layers_dec =  num_layers_dec
        if num_layers_dec>0:
            if hidden_dim_dec == 0:
                hidden_dim_dec = feature_dim
            self.hidden_dim_dec = _ntuple(num_layers_dec)(hidden_dim_dec)
        self.use_shift = use_shift
        
        # Activation function mapping
        act_dict = {
                    'none':torch.nn.Identity(), 'sigmoid':torch.nn.Sigmoid(), 'tanh':torch.nn.Tanh(),
                    'relu':torch.nn.ReLU(), 'atanh': Atanh
                    }
        self.activation = act_dict[activation]
        if final_activation == "atanh":
            self.final_activation = act_dict[final_activation](dim=feature_dim)
        else:
            self.final_activation = act_dict[final_activation]
        self.clamp_weights = clamp_weights
        
        # Build decoder layers
        layers = []
        for l in range(num_layers_dec):
            feature_dim = self.hidden_dim_dec[l]
            feature_dim = latent_dim if feature_dim == 0 else feature_dim
            layers.append(DecoderLayer(latent_dim, feature_dim, ldecode_matrix, bias=self.use_shift))
            if activation == "atanh":
                layers.append(self.activation(dim=feature_dim))
            else:
                layers.append(self.activation)
            latent_dim = feature_dim
        feature_dim = self.channels
        layers.append(DecoderLayer(latent_dim,feature_dim,ldecode_matrix,bias=self.use_shift))

        self.layers = nn.Sequential(*layers)
        self.frz_layers = None
        self.mask = None
        self.reset_parameters('normal', ldec_std)
        
    def normalize(self, input:Tensor):
        """Compute normalization factors based on input statistics."""
        if self.norm == "min_max":
            self.div.data = torch.max(torch.abs(input),dim=0)[0]
        elif self.norm == "mean_std":
            self.div.data = torch.std(input,dim=0)
        self.div.data = torch.max(self.div,torch.ones_like(self.div))
        
    def reset_parameters(self, init_type, param=0.5) -> None:
        """Initialize all decoder layer parameters."""
        for layer in list(self.layers.children()):
            if isinstance(layer, DecoderLayer):
                layer.reset_parameters(param,init_type)

    def get_scale(self):
        """Get scale parameter from first layer (single layer decoders only)."""
        assert self.num_layers_dec == 0, "Can only get scale for 0 hidden layers decoder!"
        return list(self.layers.children())[0].scale

    def get_shift(self):
        """Get shift parameter from first layer (single layer decoders only)."""
        assert self.num_layers_dec == 0, "Can only get scale for 0 hidden layers decoder!"
        return list(self.layers.children())[0].shift
    
    def clamp(self, val: float = 0.2) -> None:
        """Clamp all layer parameters to prevent extreme values."""
        for layer in list(self.layers.children()):
            if isinstance(layer, DecoderLayer):
                layer.clamp(val)

    def size(self, use_torchac=False):
        """Calculate decoder size in bits."""
        return sum([p.numel()*torch.finfo(p.dtype).bits for p in self.parameters()])

    def scale_norm(self):
        """Get norm of scale parameters from first layer."""
        return list(self.layers.children())[0].scale.norm()

    def scale_grad_norm(self):
        """Get gradient norm of scale parameters from first layer."""
        return list(self.layers.children())[0].scale.grad.norm()
    
    def forward(self, weight: Tensor) -> Tensor:
        """Forward pass through decoder with optional masking for partial freezing."""
        weight = StraightThrough.apply(weight)
        if self.mask is not None:
            # Handle partially frozen decoder
            w_out_nonfrz = self.layers(weight[self.mask]/self.div)
            w_out_frz = self.frz_layers(weight[~self.mask]/self.frz_div)
            w_out = torch.zeros((weight.shape[0], w_out_nonfrz.shape[1]),device='cuda',
                                dtype=w_out_nonfrz.dtype)
            w_out[self.mask] = w_out_nonfrz
            w_out[~self.mask] = w_out_frz
        else:
            w_out = self.layers(weight/self.div)
        w_out = self.final_activation(w_out)
        if self.clamp_weights>0.0:
            w_out = torch.clamp(w_out, min=-self.clamp_weights, max=self.clamp_weights)
        return w_out
    
    def freeze_partial(self, mask: torch.Tensor):
        """Freeze subset of parameters based on mask for selective training."""
        self.mask = mask
        self.frz_layers = copy.deepcopy(self.layers)
        self.frz_div = self.div.clone()
        for param in self.frz_layers.parameters():
            param.requires_grad_(False)
    
    def invert(self, output: Tensor) -> Tensor:
        with torch.no_grad():
            x = output
            prev_layer = None
            for idx,layers in enumerate(list(self.layers.children())[::-1]):
                if isinstance(layers, DecoderLayer):
                    x = layers.invert(x)
                elif isinstance(layers, torch.nn.Identity):
                    continue 
                elif isinstance(layers, torch.nn.ReLU):
                    if isinstance(prev_layer, DecoderLayer):
                        min_x = x.min(dim=0)[0]
                        shift_x = torch.min(min_x,torch.zeros_like(min_x)).unsqueeze(0)
                        if prev_layer.shift is not None:
                            prev_layer.shift.data -= torch.matmul(shift_x,prev_layer.scale)
                        else:
                            prev_layer.shift = Parameter(-torch.matmul(shift_x,prev_layer.scale),requires_grad=False)
                            prev_layer.shift.device = prev_layer.scale.device
                        x -= shift_x
                elif isinstance(layers, torch.nn.Sigmoid):
                    if isinstance(prev_layer, DecoderLayer):
                        max_x, min_x = x.max(dim=0)[0], x.min(dim=0)[0]
                        diff_x = max_x-min_x
                        diff_x = torch.max(diff_x,torch.ones_like(diff_x))
                        prev_layer.scale.data /= diff_x.unsqueeze(-1)
                        x /= diff_x.unsqueeze(0)

                        min_x = x.min(dim=0)[0]
                        shift_x = torch.min(min_x,torch.zeros_like(min_x)).unsqueeze(0)
                        if prev_layer.shift is not None:
                            prev_layer.shift.data -= torch.matmul(shift_x,prev_layer.scale)
                        else:
                            prev_layer.shift = Parameter(-torch.matmul(shift_x,prev_layer.scale),requires_grad=False)
                            prev_layer.shift.device = prev_layer.scale.device
                        x -= shift_x

                    x = torch.clamp(x, min=epsilon, max=1-epsilon)
                    x = torch.log(x/(1-x))
                elif isinstance(layers, torch.nn.Tanh):
                    if isinstance(prev_layer, DecoderLayer):
                        max_x, min_x = x.max(dim=0)[0], x.min(dim=0)[0]
                        diff_x = max_x-min_x
                        diff_x = torch.max(diff_x,torch.ones_like(diff_x)*2)
                        prev_layer.scale.data /= diff_x.unsqueeze(-1)
                        x /= diff_x.unsqueeze(0)

                        min_x = x.min(dim=0)[0]
                        shift_x = torch.min(min_x+1,torch.zeros_like(min_x)).unsqueeze(0)
                        if prev_layer.shift is not None:
                            prev_layer.shift.data -= torch.matmul(shift_x,prev_layer.scale)
                        else:
                            prev_layer.shift = Parameter(-torch.matmul(shift_x,prev_layer.scale),requires_grad=False)
                            prev_layer.shift.device = prev_layer.scale.device
                        x -= shift_x

                    x = torch.clamp(x, min=-1+epsilon, max=1-epsilon)
                    x = torch.atanh(x)
                elif isinstance(layers, Atanh):
                    x = layers.invert(x)
                prev_layer = layers
            return x*self.div
        
    
    def infer(self, weight: Tensor) -> Tensor:
        weight = StraightThrough.apply(weight)
        w_out = self.layers(weight/self.div)
        w_out = self.final_activation(w_out)
        if self.clamp_weights>0.0:
            w_out = torch.clamp(w_out, min=-self.clamp_weights, max=self.clamp_weights)
        return w_out


class LatentDecoderRes(LatentDecoder):
    """Residual decoder that adds decoded residuals to previous frame's attributes."""
    
    def __init__(self, **kwargs):
        super(LatentDecoderRes, self).__init__(**kwargs)
        self.frame_idx = 1
        self.identity = False  # Whether to use identity mode (no quantization)
        
    def forward(self, weight: Tensor) -> Tensor:
        """Forward pass with residual connection to previous frame."""
        if self.identity:
            # Identity mode: direct residual addition
            return weight+self.decoded_att
        if self.frame_idx == 1:
            # First frame: no residual connection
            return weight
            
        # Standard residual decoding
        weight = StraightThrough.apply(weight)
        if self.mask is not None:
            # Handle partially frozen decoder
            w_out_nonfrz = self.layers(weight[self.mask]/self.div)
            w_out = torch.zeros((weight.shape[0], w_out_nonfrz.shape[1]),device='cuda',
                                dtype=w_out_nonfrz.dtype)
            w_out[self.mask] = w_out_nonfrz
        else:
            w_out = self.layers(weight/self.div)
        w_out = self.final_activation(w_out)
        if self.clamp_weights>0.0:
            w_out = torch.clamp(w_out, min=-self.clamp_weights, max=self.clamp_weights)
        return w_out+self.decoded_att
    
    def invert(self, output: Tensor): 
        """Invert decoder - output must be residual for non-identity mode."""
        if self.identity:
            return output
        else:
            return super().invert(output)
        
    def init_decoded(self, decoded_att: torch.Tensor):
        """Initialize with previous frame's decoded attributes."""
        self.decoded_att = decoded_att.clone()
        self.decoded_att.requires_grad_(False)

    def freeze_partial(self, mask: torch.Tensor):
        """Freeze subset of parameters based on mask."""
        self.mask = mask

    def state_dict(self):
        state = super().state_dict()
        return state
    
    def load_state_dict(self, state):
        super().load_state_dict(state)

class DecoderIdentity(Module):
    """Identity decoder that passes input through unchanged (no compression)."""

    def __init__(self) -> None:
        super(DecoderIdentity, self).__init__()
        # Compatibility attributes for interface consistency
        self.latent_dim = 1
        self.num_layers_dec = 0
        self.shift = False
        self.norm = 'none'
        self.div = 1.0
        
    # For compatibility with Decoder
    def reset_parameters(self, init_type, param=1.0) -> None:
        return

    def forward(self, input: Tensor) -> Tensor:
        return input

    def scale_norm(self):
        return 1
    
    def scale_grad_norm(self):
        return 1
    
    def size(self, use_torchac=False) -> int:
        return 0
    
    def freeze_partial(self, mask: torch.Tensor):
        return None
    
    def invert(self, output: Tensor) -> Tensor:
        return output



class LatentEncoder(Module):
    """Neural encoder for converting Gaussian attributes to quantized latent codes."""
    
    def __init__(
        self,
        latent_dim: int,
        feature_dim: int,
        ldecode_matrix:str,
        use_shift: bool,
        latent_norm: str,
        num_layers_dec:int = 0,
        hidden_dim_dec:int = 0,
        activation:str = 'relu',
        final_activation:str = 'none',
        clamp_weights:float = 0.0,
        ldec_std:float = 1.0,
        **kwargs,
    ) -> None:
        
        super(LatentEncoder, self).__init__()
        latent_dim = feature_dim if latent_dim == 0 else latent_dim
        self.ldecode_matrix = ldecode_matrix
        self.channels = feature_dim
        self.latent_dim = latent_dim
        
        # Multi-layer encoder configuration
        self.num_layers_dec =  num_layers_dec
        if num_layers_dec>0:
            if hidden_dim_dec == 0:
                hidden_dim_dec = feature_dim
            self.hidden_dim_dec = _ntuple(num_layers_dec)(hidden_dim_dec)
        self.use_shift = use_shift
        
        # Activation function mapping
        act_dict = {
                    'none':torch.nn.Identity(), 'sigmoid':torch.nn.Sigmoid(), 'tanh':torch.nn.Tanh(),
                    'relu':torch.nn.ReLU(),
                    }
        self.activation = act_dict[activation]
        self.final_activation = act_dict[final_activation]
        self.clamp_weights = clamp_weights
        
        # Build encoder layers (reverse of decoder)
        layers = []
        for l in range(num_layers_dec):
            latent_dim = self.hidden_dim_dec[l]
            latent_dim = feature_dim if latent_dim == 0 else feature_dim
            layers.append(DecoderLayer(feature_dim, latent_dim, ldecode_matrix, bias=self.use_shift))
            layers.append(self.activation)
            feature_dim = latent_dim
        layers.append(DecoderLayer(feature_dim,latent_dim,ldecode_matrix,bias=self.use_shift))

        self.layers = nn.Sequential(*layers)
        self.reset_parameters('normal', ldec_std)

    def reset_parameters(self, init_type, param=0.5) -> None:
        """Initialize all encoder layer parameters."""
        for layer in list(self.layers.children()):
            if isinstance(layer, DecoderLayer):
                layer.reset_parameters(param,init_type)

    def forward(self, weight: Tensor) -> Tensor:
        """Forward pass through encoder."""
        w_out = self.layers(weight)
        w_out = self.final_activation(w_out)
        if self.clamp_weights>0.0:
            w_out = torch.clamp(w_out, min=-self.clamp_weights, max=self.clamp_weights)
        return w_out
    

class LatentAutoEncoder(Module):
    """Autoencoder combining encoder and decoder for end-to-end training."""
    
    def __init__(self, latent_dim, feature_dim, **kwargs) -> None:
        super().__init__()
        self.encoder = LatentEncoder(latent_dim, feature_dim, **kwargs)
        self.decoder = LatentDecoder(latent_dim, feature_dim, **kwargs)

    def forward(self, weight: Tensor) -> Tensor:
        """Encode then decode input for reconstruction."""
        latent = self.encoder(weight)
        reconstructed = self.decoder(latent)
        return reconstructed

class Gate(Module):
    """Learnable gating mechanism for temporal consistency in Gaussian attributes.
    
    Uses concrete relaxation of Bernoulli variables to learn which Gaussians
    should be updated between frames. Supports L0 and L2 regularization.
    """
    
    def __init__(self, num_gates, gamma=-0.1, eta=1.1, lr = 1.0e-4, temp=0.5,
                 lambda_l2=0.0, lambda_l0=1.0, iter_noise=20, device='cuda', 
                 init_probs=None):
        super(Gate, self).__init__()
        
        # Learnable gate parameters (log odds)
        self.log_alphas = nn.Parameter(torch.zeros((num_gates),dtype=torch.float32,
                                                   device=device).requires_grad_(True))
        self.num_gates = num_gates                                   
        
        # Concrete distribution parameters
        self.gamma = gamma  # Lower bound of concrete distribution
        self.eta = eta      # Upper bound of concrete distribution
        self.temp = temp    # Temperature for concrete relaxation
        
        # Training parameters
        self.lr = lr
        self.iter_noise = iter_noise  # How often to resample noise
        self.device = device
        self.iter_counter = 0
        
        # Regularization weights
        self.lambda_l2 = lambda_l2  # L2 regularization on residuals
        self.lambda_l0 = lambda_l0  # L0 regularization (sparsity)
        
        self.resample_noise()
        if init_probs is not None:
            self.reset_params(init_probs)
        self.optimizer = torch.optim.Adam([self.log_alphas], lr= self.lr, eps=1.0e-15)

    def reset_params(self, init_probs=None):
        """Reset gate parameters, optionally with initialization probabilities."""
        if init_probs is not None:
            init_probs = init_probs.flatten()
            assert init_probs.shape[0] == self.log_alphas.shape[0]
            init_probs = torch.clamp(init_probs,epsilon,1-epsilon)
            # Convert probabilities to log odds with temperature scaling
            self.log_alphas.data = torch.log(init_probs/(1-init_probs))+\
                                    self.temp*math.log(-self.gamma/self.eta)
            self.clamp_params()
            gate = self.sample_gate(stochastic=False)
            self.gate = gate # used outside of module
        else:
            self.log_alphas.data *= 0

    def prune_params(self, keep_mask):
        """Remove gates corresponding to pruned Gaussians."""
        self.noise = self.noise[keep_mask]

        # Update optimizer state
        assert len(self.optimizer.param_groups) == 1
        group = self.optimizer.param_groups[0]
        stored_state = self.optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][keep_mask]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][keep_mask]

            del self.optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter((group["params"][0][keep_mask].requires_grad_(True)))
            self.optimizer.state[group['params'][0]] = stored_state

            log_alphas = group["params"][0]
        else:
            group["params"][0] = nn.Parameter(group["params"][0][keep_mask].requires_grad_(True))
            log_alphas = group["params"][0]
        self.log_alphas = log_alphas
        self.num_gates = self.log_alphas.shape[0]
        self.gate = self.gate[keep_mask]


    def add_params(self, clone_mask=None, split_mask=None, N=2):
        """Add gates for new Gaussians created during densification."""
        assert (clone_mask is not None) ^ (split_mask is not None), \
                "Only one of clone mask or split mask should be enabled!"

        if clone_mask is not None:
            # Clone existing gates
            new_noise = torch.rand(clone_mask.sum().item(), self.iter_noise).to(self.device)
            self.noise = torch.cat((self.noise,new_noise*(1-2*epsilon)+epsilon),dim=0)
            
            extension_tensor = self.log_alphas[clone_mask]
            self.gate = torch.cat((self.gate,self.gate[clone_mask]),dim=0)
        else:
            # Split existing gates
            new_noise = torch.rand(split_mask.sum().item()*N, self.iter_noise).to(self.device)
            self.noise = torch.cat((self.noise,new_noise*(1-2*epsilon)+epsilon),dim=0)
            extension_tensor = self.log_alphas[split_mask].repeat(N)
            self.gate = torch.cat((self.gate,self.gate[split_mask].repeat(N)),dim=0)

        # Update optimizer with new parameters
        group = self.optimizer.param_groups[0]
        stored_state = self.optimizer.state.get(group['params'][0], None)
        if stored_state is not None:

            stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
            stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

            del self.optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
            self.optimizer.state[group['params'][0]] = stored_state

            log_alphas = group["params"][0]
        else:
            group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
            log_alphas= group["params"][0]

        self.log_alphas = log_alphas
        self.num_gates = self.log_alphas.shape[0]

    @torch.no_grad()
    def size(self):
        """Calculate compressed size of gate pattern using entropy."""
        gate = self.sample_gate(stochastic=False)
        num_inactive = (gate==0).sum().item()
        num_active = gate.numel()-num_inactive
        num_gates = gate.numel()
        net_size = 0
        if num_active>0:
            net_size += -num_active*np.log2(num_active/num_gates)
        if num_inactive>0:
            net_size += -num_inactive*np.log2(num_inactive/num_gates)
        return net_size

    def clamp_params(self):
        """Clamp log_alphas to prevent extreme values."""
        self.log_alphas.data.clamp_(min=math.log(1e-3), max=math.log(1e3))

    def cdf(self, x):
        """Cumulative distribution function for concrete distribution."""
        xstretch = (x - self.gamma) / (self.eta - self.gamma)
        logits = math.log(xstretch) - math.log(1 - xstretch) # log(-gamma/eta) for x=0
        probs = F.sigmoid(logits * self.temp - self.log_alphas)
        return probs.clamp(min=epsilon, max=1-epsilon)

    def reg_loss(self, x):
        """Compute L0 + L2 regularization loss."""
        assert x.shape[0] == self.num_gates
        lambda_net = torch.mean((0.5*self.lambda_l2 * x.pow(2)) + self.lambda_l0, dim=1)
        net_reg_loss = torch.mean((1 - self.cdf(0)) * lambda_net)
        return net_reg_loss

    def resample_noise(self):
        """Resample noise for stochastic gate sampling."""
        noise = torch.rand(self.num_gates, self.iter_noise).to(self.device)
        self.noise = noise*(1-2*epsilon)+epsilon

    def hconcrete_dist(self, x):
        """Hard concrete distribution transformation."""
        y = F.sigmoid((torch.log(x) - torch.log(1 - x) + self.log_alphas) / self.temp)
        return y * (self.eta - self.gamma) + self.gamma

    def sample_gate(self, stochastic=True):
        """Sample gate values (stochastic during training, deterministic for eval)."""
        if stochastic:
            if self.iter_counter == self.iter_noise:
                self.resample_noise()
                self.iter_counter = 0
            noise = self.noise[:,self.iter_counter]
            gate = self.hconcrete_dist(noise)
        else:
            # Deterministic sampling for evaluation
            gate = F.sigmoid(self.log_alphas/self.temp)*(self.eta-self.gamma)+self.gamma
        
        clamped_gate = F.hardtanh(gate, min_val=0.0, max_val=1.0)
        return clamped_gate

    def forward(self, x, stochastic=False):
        """Apply gating to input tensor x."""
        gate = self.sample_gate(stochastic=stochastic)
        self.gate = gate # used outside of module
        if x.dim() == 2:
            out = x*gate.unsqueeze(-1)
        elif x.dim() == 3:
            out = x*gate.unsqueeze(-1).unsqueeze(-1)
        else:
            raise Exception("Dimension should be 2 or 3 for gate input!")
        return out

    def forward_eval(self, x):
        """Apply deterministic gating for evaluation."""
        gate = self.sample_gate(stochastic=False)
        out = x*gate.unsqueeze(-1)
        return out
    
    def step(self):
        """Perform optimizer step and update iteration counter."""
        self.optimizer.step()
        self.iter_counter += 1
        self.optimizer.zero_grad()

    def get_gating_pattern(self):
        """Get current gating pattern for compression."""
        return {'active_mask': (self.gate!=0.0).detach().cpu().numpy(), 'active_values': self.gate.detach().cpu().numpy()}

    def copy(self):
        """Create a deep copy of the Gate instance."""
        # Create new instance with same initialization parameters
        new_gate = Gate(
            num_gates=self.num_gates,
            gamma=self.gamma,
            eta=self.eta,
            lr=self.lr,
            temp=self.temp,
            lambda_l2=self.lambda_l2,
            lambda_l0=self.lambda_l0,
            iter_noise=self.iter_noise,
            device=self.device
        )
        
        # Copy the log_alphas parameter
        new_gate.log_alphas.data = self.log_alphas.data.clone()
        
        # Copy the current gate state if it exists
        if hasattr(self, 'gate'):
            new_gate.gate = self.gate.clone()
        
        # Copy the noise state
        new_gate.noise = self.noise.clone()
        
        # Copy the iteration counter
        new_gate.iter_counter = self.iter_counter
        
        # Create new optimizer with copied parameters
        new_gate.optimizer = torch.optim.Adam([new_gate.log_alphas], lr=self.lr, eps=1.0e-15)
        
        return new_gate


