import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.cli_utils import *
import time

class Community(nn.Module):
    def __init__(self,
                agents,
                loaders,
                algorithm,
                domain_names,
                save_root,
                logger,
                save_vectors_to_file_on=True):
        super().__init__()
        self.logger = logger
        self.loaders = loaders
        
        self.num_agents = len(loaders)
        self.num_domains = len(loaders[0])
        self.domain_names = domain_names
        self.algorithm = algorithm
        self.save_root = save_root

        self.agents = agents
        for agent_idx, agent in enumerate(self.agents):
            agent.model.agent_id = agent_idx
            agent.model.save_root = save_root
            agent.model.save_vectors_to_file_on = save_vectors_to_file_on

        self.target_agent = 2
        self.target_corruption = 'gaussian_noise'

        self.loader_iters = [[iter(loader) for loader in agent_loaders] for agent_loaders in self.loaders]
        self.results = [{
            'acc1': [],
            # 'acc5': [],
        } for _ in range(self.num_agents)]

    @torch.no_grad()
    def run_exp(self):
        for type_idx in range(self.num_domains):
            for agent_idx, agent_loaders in enumerate(self.loaders):
                val_loaders = agent_loaders[type_idx]
                model = self.agents[agent_idx]
                accs, eces = [], []
                for corruption_idx, val_loader in enumerate(val_loaders):
                    self.logger.info(f'--- agent {agent_idx+1}, {self.domain_names[agent_idx][type_idx][corruption_idx]} begin ---')
                    batch_time = AverageMeter('Time', ':6.3f')
                    top1 = AverageMeter('Acc@1', ':6.2f')
                    top5 = AverageMeter('Acc@5', ':6.2f')
                    progress = ProgressMeter(
                        len(val_loader),
                        [batch_time, top1, top5],
                        prefix='Test: ')
                    for i, dl in enumerate(val_loader):
                        end = time.time()
                        images, target = dl[0].cuda(), dl[1].cuda()
                        output = model(images)
                        acc1, acc5 = accuracy(output, target, topk=(1, 5))
                        top1.update(acc1[0], images.size(0))
                        top5.update(acc5[0], images.size(0))
                        batch_time.update(time.time() - end)
                        del output

                        if i % 20 == 0:
                            self.logger.info(progress.display(i))
                            if 'fuse' in self.algorithm:
                                self.logger.info(model.model.plot_alpha())
                        
                        if i == 0:
                            model.model.corruption = self.domain_names[agent_idx][type_idx][corruption_idx]
                            model.model.count = 1
                    
                    self.logger.info(f'--- agent {agent_idx+1}, {self.domain_names[agent_idx][type_idx][corruption_idx]} end ---')
                    accs.append(top1.avg.item())

                self.results[agent_idx]['acc1'].append(accs)
            
            if 'cola' in self.algorithm:
                self._exchange_knowledge()

    def _exchange_knowledge(self):
        self.logger.info('=== exchanging knowledge ===')
        current_knowledge = [agent.model.get_weights_for_share() for agent in self.agents]
        for i in range(self.num_agents):
            other_knowledge = []
            for j in range(self. num_agents):
                if i == j: continue
                other_knowledge += current_knowledge[j]
            self.agents[i].model.add_weights(other_knowledge)