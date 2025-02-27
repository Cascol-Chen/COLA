"""
Copyright to DeYO Authors, ICLR 2024 Spotlight (top-5% of the submissions)
built upon on Tent code.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import torchvision
import math
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
from dataset.ImageNetMask import imagenet_r_mask

class DeYO(nn.Module):
    """SAR online adapts a model by Sharpness-Aware and Reliable entropy minimization during testing.
    Once SARed, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, deyo_margin=0.5*math.log(1000), margin_e0=0.4*math.log(1000)):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic

        self.deyo_margin = deyo_margin
        self.margin_e0 = margin_e0
        self.imagenet_mask = None

        # self.model_state, self.optimizer_state = \
        #     copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, targets=None, flag=True, group=None):
        if self.episodic:
            self.reset()
        
        if targets is None:
            for _ in range(self.steps):
                if flag:
                    outputs, backward, final_backward = forward_and_adapt_sar(x, self.model,
                                                                              self.optimizer, self.deyo_margin,
                                                                              self.margin_e0, targets, flag, group, self.imagenet_mask)
                else:
                    outputs = forward_and_adapt_sar(x, self.model,
                                                    self.optimizer, self.deyo_margin,
                                                    self.margin_e0, targets, flag, group)
        else:
            for _ in range(self.steps):
                if flag:
                    outputs, backward, final_backward, corr_pl_1, corr_pl_2 = forward_and_adapt_sar(x, self.model, 
                                                                                                    self.optimizer, 
                                                                                                    self.deyo_margin,
                                                                                                    self.margin_e0,
                                                                                                    targets, flag, group)
                else:
                    outputs = forward_and_adapt_sar(x, self.model, self.optimizer, 
                                                    self.deyo_margin,
                                                    self.margin_e0,
                                                    targets, flag, group, self)
        
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.ema = None


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    #temprature = 1.1 #0.9 #1.2
    #x = x ** temprature #torch.unsqueeze(temprature, dim=-1)
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_sar(x, model, optimizer, deyo_margin, margin, targets=None, flag=True, group=None, imagenet_mask=None):
    """Forward and adapt model input data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    outputs = model(x)
    if imagenet_mask is not None:
        outputs = outputs[:, imagenet_mask]
    if not flag:
        return outputs
    
    optimizer.zero_grad()
    entropys = softmax_entropy(outputs)
    filter_ids_1 = torch.where((entropys < deyo_margin))
    entropys = entropys[filter_ids_1]
    backward = len(entropys)
    if backward==0:
        if targets is not None:
            return outputs, 0, 0, 0, 0
        return outputs, 0, 0

    x_prime = x[filter_ids_1]
    x_prime = x_prime.detach()

    resize_t = torchvision.transforms.Resize(((x.shape[-1]//4)*4,(x.shape[-1]//4)*4))
    resize_o = torchvision.transforms.Resize((x.shape[-1],x.shape[-1]))
    x_prime = resize_t(x_prime)
    x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=4, ps2=4)
    perm_idx = torch.argsort(torch.rand(x_prime.shape[0],x_prime.shape[1]), dim=-1)
    x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1),perm_idx]
    x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=4, ps2=4)
    x_prime = resize_o(x_prime)
    
    with torch.no_grad():
        model.eval()
        outputs_prime = model(x_prime)
        if imagenet_mask is not None:
            outputs_prime = outputs_prime[:, imagenet_mask]
        model.train()

    prob_outputs = outputs[filter_ids_1].softmax(1)
    prob_outputs_prime = outputs_prime.softmax(1)

    cls1 = prob_outputs.argmax(dim=1)

    plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1,1)) - torch.gather(prob_outputs_prime, dim=1, index=cls1.reshape(-1,1))
    plpd = plpd.reshape(-1)
    
    
    filter_ids_2 = torch.where(plpd > 0.2)
    entropys = entropys[filter_ids_2]
    final_backward = len(entropys)
    plpd = plpd[filter_ids_2]
    
    if targets is not None:
        corr_pl_1 = (targets[filter_ids_1] == prob_outputs.argmax(dim=1)).sum().item()
        corr_pl_2 = (targets[filter_ids_1][filter_ids_2] == prob_outputs[filter_ids_2].argmax(dim=1)).sum().item()

    coeff = ((1 / (torch.exp(((entropys.clone().detach()) - margin)))) +
             (1 / (torch.exp(-1. * plpd.clone().detach()))))            
    entropys = entropys.mul(coeff)
    loss = entropys.mean(0)

    if final_backward != 0:
        loss.backward()
        optimizer.step()
    optimizer.zero_grad()

    del x_prime
    del plpd
    
    if targets is not None:
        return outputs, backward, final_backward, corr_pl_1, corr_pl_2
    return outputs, backward, final_backward

def collect_params(model):
    """Collect the affine scale + shift parameters from norm layers.
    Walk the model's modules and collect all normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
        if 'layer4' in nm:
            continue
        if 'blocks.9' in nm:
            continue
        if 'blocks.10' in nm:
            continue
        if 'blocks.11' in nm:
            continue
        if 'norm.' in nm:
            continue
        if nm in ['norm']:
            continue

        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")

    return params, names


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with DeYO."""
    # train mode, because SAR optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what SAR updates
    model.requires_grad_(False)
    # configure norm for SAR updates: enable grad + force batch statisics (this only for BN models)
    for nm, m in model.named_modules():
        # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
        if 'layer4' in nm:
            continue
        if 'blocks.9' in nm:
            continue
        if 'blocks.10' in nm:
            continue
        if 'blocks.11' in nm:
            continue
        if 'norm.' in nm:
            continue
        if nm in ['norm']:
            continue

        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state