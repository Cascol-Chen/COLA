import torch
from timm.models.vision_transformer import VisionTransformer
import torch.nn as nn
from dataset.ImageNetMask import imagenet_r_mask, imagenet_a_mask

def get_vit_featurer(network:VisionTransformer):
    network.head = nn.Identity()
    if hasattr(network, 'head_dist'):
        network.head_dist = None
    return network


class T3A(torch.nn.Module):
    """
    Test Time Template Adjustments (T3A)
    """
    def __init__(self, model:VisionTransformer, num_classes, filter_K):
        super().__init__()
        self.classifier = model.head
        self.featurizer = get_vit_featurer(model)
        # self.classifier = Classifier(self.featurizer.norm.normalized_shape[0], num_classes)

        warmup_supports = self.classifier.weight.data
        self.warmup_supports = warmup_supports
        warmup_prob = self.classifier(self.warmup_supports)

        # warmup_prob = warmup_prob[:, imagenet_a_mask]
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = torch.nn.functional.one_hot(warmup_prob.argmax(1), num_classes=num_classes).float()

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

        self.filter_K = filter_K
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, x, adapt=True):
        z = self.featurizer(x)
        if adapt:
            # online adaptation
            p = self.classifier(z)
            if self.imagenet_mask is not None:
                p = p[:, self.imagenet_mask]
            yhat = torch.nn.functional.one_hot(p.argmax(1), num_classes=self.num_classes).float()
            ent = softmax_entropy(p)

            # prediction
            self.supports = self.supports.to(z.device)
            self.labels = self.labels.to(z.device)
            self.ent = self.ent.to(z.device)
            self.supports = torch.cat([self.supports, z])
            self.labels = torch.cat([self.labels, yhat])
            self.ent = torch.cat([self.ent, ent])
        
        supports, labels = self.select_supports()
        supports = torch.nn.functional.normalize(supports, dim=1)
        weights = (supports.T @ (labels))
        return z @ torch.nn.functional.normalize(weights, dim=0)

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s)))).cuda()

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s)))).cuda()
        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat==i][indices2][:filter_K])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]
        
        return self.supports, self.labels

    def predict(self, x, adapt=False):
        return self(x, adapt)

    def reset(self):
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)