from collections import defaultdict
import os.path as path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer
import math
import copy

class CoLAViT(nn.Module):
    '''
    An efficient wrapper to implement CoLA.
    This is however targeted for ViT only.
    '''
    def __init__(self,
                vit:VisionTransformer,
                weight_margin=0.01, # w_m in Eqn. (8)
                logger=None,
                domain_detect_on=True,
                shift_detection_threshold=0.1,
                save_per_iteration=100,
                auto_remove_on=False, # for single-domain TTA
                max_num_vectors=32,
                save_vectors_to_file_on=False,
                save_root=None,
                fp_agent_mode_on=False,
                fp_temperature=5):
        super().__init__()
        self.model = vit

        self.weight_margin = weight_margin
        self.logger = logger

        self.domain_detect_on = domain_detect_on
        self.shift_detection_threshold = shift_detection_threshold

        self.save_per_iteration_on = not domain_detect_on
        self.save_per_iteration = save_per_iteration
        self.current_iteration = 0

        self.auto_remove_on = auto_remove_on
        self.max_num_vectors = max_num_vectors

        self.save_vectors_to_file_on = save_vectors_to_file_on
        self.save_root = save_root

        self.fp_agent_mode_on = fp_agent_mode_on
        self.fp_temperature = fp_temperature

        ### For shift detection
        self.domain_var = None
        self.domain_mean = None
        self.domain_infos = []

        self.cola_optimizer = None
        self.share_idx = 1

        ### For debug and file saving purpose ###
        self.corruption = 'original'
        self.save_root = None
        self.agent_id = 0
        self.adaptation_round = 0
        self.count = 1
        self.weight_names = []
        ##########################################

    @torch.no_grad()
    def count_iteration(self):
        if self.save_per_iteration_on:
            self.current_iteration += 1
            if self.current_iteration == self.save_per_iteration:
                self.save_current_weight()
                self.current_iteration = 0

    def forward(self, x):
        if not self.fp_agent_mode_on:
            if self.training:
                embed_features = self.forward_embedding(x)
                if self.domain_detect_on and self._check_should_save(embed_features):
                    # self.logger.info('===> saving models')
                    self.save_current_weight()
                outputs = self.model(x)
                self._update_domain_info(embed_features)
            else:
                # when multiple passes for the same inputs are needed, e.g., DeYO
                # this ensure the domain info is not updated twice or more
                outputs = self.model(x)
            self.count_iteration()
        else:
            outputs = self.model(x)
            embed_features = self.forward_embedding(x)
            self._update_domain_info(embed_features)
            self._update_alpha()
        return outputs
    
    @torch.no_grad()
    def _update_alpha(self):
        if self.domain_mean is None:
            return
        self.alpha.data = torch.Tensor([self._calculate_edge(
            self.domain_var, self.domain_mean, self.weights_domain_vars[i], self.weights_domain_means[i]
        ) for i in range(len(self.weights_domain_means))]).cuda()
        self.alpha.data /= self.fp_temperature
        # self.alpha.data = torch.zeros_like(self.alpha.data)

    @torch.no_grad()
    def _calculate_edge(self, var1, mean1, var2, mean2):
        dist = self._calculate_domain_shift(var1, mean1, var2, mean2)
        return 1 / (dist + 1e-4)

    @torch.no_grad()
    def save_file(self):
        current_weights = [torch.stack(m.get_fused_weights_and_biases(), dim=0) for m in self.fuse_modules]
        current_weights = torch.stack(current_weights, dim=0)
        self.state_dict
        torch.save({
            'model': current_weights,
            'domain_mean': self.domain_mean, # For Eqn. (4)
            'domain_var': self.domain_var # For Eqn. (4)
        }, self.save_root + f'agent{self.agent_id+1}_{self.corruption}_{self.count}.pth')
        self.count += 1

    def estimate_scale_factor(self):
        max_weight = max([_.abs().mean().item() for _ in self.params['epsilon_weight']])
        max_bias = max([_.abs().mean().item() for _ in self.params['epsilon_bias']])
        max_value = max(max_weight, max_bias)

        scale_factor = max_value / self.weight_margin
        return scale_factor
    
    @torch.no_grad()
    def save_current_weight(self):
        if self.auto_remove_on and self.weight_nums > self.max_num_vectors:
            idx = torch.argmin(self.alpha[1:] * self.alpha_scale) # would not remove the original weight
            self._remove(idx + 1)

        self.weight_nums += 1
        scale_factor = self.estimate_scale_factor()
        # self.logger.info(f'===> saving weight !!!, scale factor: {scale_factor}')
        new_alpha_value = -np.inf
        if scale_factor > 1:
            new_alpha_value = torch.log(torch.exp(self.alpha * self.alpha_scale).sum() * (scale_factor-1)) # Eqn. (8)
        new_alpha_value = torch.max(torch.max(self.alpha * self.alpha_scale), torch.Tensor([new_alpha_value]).cuda()) / self.alpha_scale
        new_alpha_value = torch.clip(new_alpha_value, -10, 10)

        self.alpha = nn.Parameter(torch.cat([self.alpha, new_alpha_value]))

        for m in self.model.modules():
            if isinstance(m, FuseLayerNorm):
                m.save_current_weight(self.alpha)

        if self.save_vectors_to_file_on: 
            self.save_file()

        self.domain_infos.append((self.domain_var, self.domain_mean))
        self.reset_domain_info()

        self.collect_params()
        self.alpha_optimizer = torch.optim.AdamW([
            {'params': self.params['alpha'], 'lr': 0.1, 'weight_decay': 0.1},
            {'params': self.params['alpha_scale'], 'lr': 0.1}
        ])

        if self.cola_optimizer is not None: # update optimizer
            self.cola_optimizer.alpha_optimizer = self.alpha_optimizer
            self.cola_optimizer.weight_nums = self.weight_nums
            

    def load_weights_from_files(self, root, files):
        self.weight_nums = len(files)
        self.state_dicts = [torch.load(path.join(root, file)) for file in files]
        self._construct_fuse_layer()
        del self.state_dicts

        self.collect_params() # to initialize self.params
        self.cuda()
        self.configure_model()
    
    def _construct_fuse_layer(self):
        self.alpha = nn.Parameter(torch.randn(self.weight_nums) / self.weight_nums)
        self.alpha_scale = nn.Parameter(torch.ones(1))
        ln_modules = [(name, module) for name, module in self.model.named_modules() if isinstance(module, nn.LayerNorm) and module.weight.requires_grad == True]
        self.fuse_modules = []

        if 'domain_var' in self.state_dicts[0].keys():
            self.weights_domain_vars = [sd['domain_var'] for sd in self.state_dicts]
            self.weights_domain_means = [sd['domain_mean'] for sd in self.state_dicts]

        for layer_idx, (name, module) in enumerate(ln_modules):
            if isinstance(self.state_dicts[0]['model'], dict):
                ln_weight_list = [sd['model'][f'{name}.weight'].cuda() for sd in self.state_dicts]
                ln_bias_list = [sd['model'][f'{name}.bias'].cuda() for sd in self.state_dicts]
            else: # for loading domain vectors
                ln_weight_list = [sd['model'][layer_idx][0].cuda() for sd in self.state_dicts]
                ln_bias_list = [sd['model'][layer_idx][1].cuda() for sd in self.state_dicts]
            
            fuse_ln = FuseLayerNorm(
                module.normalized_shape, module.eps,
                ln_weight_list, ln_bias_list, self.alpha, self.alpha_scale
            )
            self.fuse_modules.append(fuse_ln)
            
            modules_dict = dict(self.model.named_modules())
            if '.' in name:
                parent_name, child_name = name.rsplit('.', 1)
                parent_module = modules_dict[parent_name]
                setattr(parent_module, child_name, fuse_ln)
            else:
                setattr(self.model, name, fuse_ln)

    def configure_model(self):
        self.model.train()
        self.model.requires_grad_(False)
        self.alpha.requires_grad_(True)
        self.alpha_scale.requires_grad_(True)
        for m in self.model.modules():
            if isinstance(m, FuseLayerNorm):
                m.epsilon_weight.requires_grad_(True)
                m.epsilon_bias.requires_grad_(True)
        
    def collect_params(self):
        update_names = ['epsilon_weight', 'epsilon_bias']
        params = {update_name:[] for update_name in update_names}
        params['alpha'] = [self.alpha]
        params['alpha_scale'] = [self.alpha_scale]
        for nm, m in self.model.named_modules():
            if isinstance(m, FuseLayerNorm):
                for np, p in m.named_parameters():
                    if np in update_names:  # weight is scale, bias is shift
                        params[np].append(p)
        self.params = params
        return params
    
    @torch.no_grad()
    def forward_embedding(self, x):
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        return self.model.norm_pre(x)[:,1:]
        
    def plot_alpha(self): # debug purpose
        with torch.no_grad():
            info = f"=====> All alpha: {self.alpha.data}, scale: {self.alpha_scale.item():.4f}, dw: {self.params['epsilon_weight'][0].abs().mean(0).item():.4f}, db: {self.params['epsilon_bias'][0].abs().mean(0).item():.4f}"
        return info

    def set_optimizers(self, alpha_optimizer, eps_optimizer):
        self.alpha_optimizer = alpha_optimizer
        self.eps_optimizer = eps_optimizer

    def reset_domain_info(self):
        self.domain_var = None
        self.domain_mean = None
    
    @torch.no_grad()
    def _check_should_save(self, embed_features):
        if embed_features.shape[0] < 64: # small batch size leads to disturbulance
            return False
        
        if self.domain_var is None or self.domain_mean is None:
            return False
        
        emb_var, emb_mean = embed_features.var(dim=(0,1)), embed_features.mean(dim=(0,1))
        return self._calculate_domain_shift(
            self.domain_var,
            self.domain_mean,
            emb_var,
            emb_mean
        ) > self.shift_detection_threshold # there is a shift
    
    @torch.no_grad()
    def _calculate_domain_shift(self, domain_var, domain_mean, cur_var, cur_mean):
        d1 = (domain_var + (domain_mean - cur_mean) ** 2) / 2. / cur_var - 0.5
        d2 = (cur_var + (domain_mean - cur_mean) ** 2) / 2. / domain_var - 0.5
        return torch.mean((d1+d2))

    @torch.no_grad()
    def _update_domain_info(self, embed_features):
        if embed_features.shape[0] < 64: # small batch size leads to disturbulance
            return False
        
        emb_var, emb_mean = embed_features.var(dim=(0,1)), embed_features.mean(dim=(0,1))
        if self.domain_var is None:
            self.domain_var, self.domain_mean = emb_var, emb_mean
        else:
            self.domain_var = 0.8 * self.domain_var + 0.2 * emb_var
            self.domain_mean = 0.8 * self.domain_mean + 0.2 * emb_mean

    @torch.no_grad()
    def get_weights_for_share(self):
        weights = []
        for i in range(self.share_idx, self.weight_nums):
            weights.append( [(m.origin_weight+m.weights[i], m.origin_bias+m.biases[i]) for m in self.fuse_modules] )

        weights.append( [m.get_fused_weights_and_biases() for m in self.fuse_modules] ) # current weight would be saved later
        self.share_idx = self.weight_nums + 2

        self.logger.info(f'==> sharing {len(weights)} weights.')
        return weights
    
    @torch.no_grad()
    def add_weights(self, weights):
        if len(weights) == 0: return
        # 在等价操作下，先删除、后增加、再保存
        weights = copy.deepcopy(weights)

        self.weight_nums += len(weights)
        self.share_idx += len(weights)
        new_alpha = torch.randn(len(weights)).cuda()
        self.alpha = nn.Parameter(torch.cat([self.alpha, new_alpha]))

        for i, m in enumerate(self.fuse_modules):
            layer_weights =  [weight[i] for weight in weights]
            new_weights, new_biases = [_[0] for _ in layer_weights], [_[1] for _ in layer_weights]
            m.add_weights(self.alpha, new_weights, new_biases)

        self.save_current_weight()



    @torch.no_grad()
    def _remove(self, idx):
        self.weight_nums -= 1
        self.alpha = nn.Parameter(torch.cat([self.alpha.data[:idx], self.alpha.data[idx+1:]]))
        print(f'removing {idx}')
        for i, m in enumerate(self.fuse_modules):
            m.remove_idx(self.alpha, idx)
    
    def use_idx(self, idx):
        for nm, m in self.model.named_modules():
            if isinstance(m, FuseLayerNorm):
                m.use_idx(idx)

class FuseLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps, weights, biases, alpha, alpha_scale):
        super(FuseLayerNorm, self).__init__()
        # default LayerNorm parameter
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # fuse parameter
        self.weight_nums = len(weights)
        self.origin_weight = weights[0]
        self.origin_bias = biases[0]

        self.weights = torch.stack(weights, dim=0) - self.origin_weight
        self.biases = torch.stack(biases, dim=0) - self.origin_bias

        self.alpha = alpha # a reference of alpha shared across layers
        self.alpha_scale = alpha_scale # a reference of temperature shared across layers
        self.epsilon_weight = nn.Parameter(torch.zeros(self.weights[0].shape))
        self.epsilon_bias = nn.Parameter(torch.zeros(self.biases[0].shape))

    def forward(self, x):
        fuse_weight, fuse_bias = self.get_fused_weights_and_biases()
        return F.layer_norm(x, self.normalized_shape, fuse_weight, fuse_bias, self.eps)

    def get_fused_weights_and_biases(self):
        alphas_normalized = F.softmax(self.alpha *  self.alpha_scale, dim=0).view(-1, 1)
        fuse_weight = self.origin_weight + (self.weights * alphas_normalized).sum(0) + self.epsilon_weight
        fuse_bias = self.origin_bias + (self.biases * alphas_normalized).sum(0) + self.epsilon_bias
        return fuse_weight, fuse_bias
    
    @torch.no_grad()
    def save_current_weight(self, cur_alpha):
        self.weight_nums += 1
        cur_weight, cur_bias = self.get_fused_weights_and_biases()
        self.weights = torch.cat([self.weights, (cur_weight - self.origin_weight).unsqueeze(0)], dim=0) # update weights
        self.biases = torch.cat([self.biases, (cur_bias - self.origin_bias).unsqueeze(0)], dim=0) # update biases
        self.alpha = cur_alpha # update alpha

        # we aim to keep fused theta the same after saving new weights
        # which is achieved by adding an offset to epsilon_weight/bias
        new_fuse_weight, new_fuse_bias = self.get_fused_weights_and_biases()
        self.epsilon_weight += cur_weight - new_fuse_weight
        self.epsilon_bias += cur_bias - new_fuse_bias

    @torch.no_grad()
    def add_weights(self, cur_alpha, new_weights, new_biases):
        assert(len(new_weights) == len(new_biases))
        self.weight_nums += len(new_weights)
        new_weights = torch.stack(new_weights, dim=0) - self.origin_weight
        new_biases = torch.stack(new_biases, dim=0) - self.origin_bias

        cur_weight, cur_bias = self.get_fused_weights_and_biases()
        self.weights = torch.cat([self.weights, new_weights], dim=0)
        self.biases = torch.cat([self.biases, new_biases], dim=0)
        self.alpha = cur_alpha

        new_fuse_weight, new_fuse_bias = self.get_fused_weights_and_biases()
        self.epsilon_weight += cur_weight - new_fuse_weight
        self.epsilon_bias += cur_bias - new_fuse_bias

    @torch.no_grad()
    def remove_idx(self, cur_alpha, remove_idx):
        # used when auto remove is on for single-domain TTA
        self.weight_nums -= 1
        cur_weight, cur_bias = self.get_fused_weights_and_biases()
        self.weights = torch.cat([self.weights[:remove_idx], self.weights[remove_idx+1:]], dim=0)
        self.biases = torch.cat([self.biases[:remove_idx], self.biases[remove_idx+1:]], dim=0)
        self.alpha = cur_alpha

        new_fuse_weight, new_fuse_bias = self.get_fused_weights_and_biases()
        self.epsilon_weight += cur_weight - new_fuse_weight
        self.epsilon_bias += cur_bias - new_fuse_bias

    @torch.no_grad()
    def use_idx(self, idx):
        # set the fuse weight equivalant to the idx-th weight
        cur_weight, cur_bias = self.origin_weight + self.weights[idx], self.origin_bias + self.biases[idx]
        new_fuse_weight, new_fuse_bias = self.get_fused_weights_and_biases()
        self.epsilon_weight += cur_weight - new_fuse_weight
        self.epsilon_bias += cur_bias - new_fuse_bias
    
class CoLAOptimizer(torch.optim.Optimizer):
    def __init__(self, alpha_num, alpha_optimizer, eps_optimizer):
        self.weight_nums = alpha_num
        self.alpha_optimizer = alpha_optimizer
        self.eps_optimizer = eps_optimizer

    def zero_grad(self):
        self.alpha_optimizer.zero_grad()
        self.eps_optimizer.zero_grad()
        self.alpha_optimizer.state = defaultdict(dict)

    def step(self):
        if self.weight_nums > 1:
            self.alpha_optimizer.step()
        self.eps_optimizer.step()

    def state_dict(self):
        return {
            'weight_nums': self.weight_nums,
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'eps_optimizer': self.eps_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.weight_nums = state_dict['weight_nums']
        self.alpha_optimizer.load_state_dict(state_dict['alpha_optimizer'])
        self.eps_optimizer.load_state_dict(state_dict['eps_optimizer'])