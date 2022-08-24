import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam

NOISE_SCALE = 0.0001


class KMeans(nn.Module):
    def __init__(
            self,
            input_dim=10,
            num_steps=406,
            num_classes=19,
            num_prototypes=19,
            empty_cluster_threshold=0.02,
            learn_weights=False,
    ):
        super(KMeans, self).__init__()
        self.empty_cluster_threshold = empty_cluster_threshold / num_prototypes
        self.input_dim = input_dim
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.prototypes = torch.nn.Parameter(torch.randn(self.num_prototypes, self.num_steps, self.input_dim),
                                             requires_grad=True)

        self.criterion = lambda x, y, m: (m * (x - y) ** 2).sum((2, 3), keepdim=True)

    def forward(self, input_seq, label, mask):
        batch_size = input_seq.size(0)
        label = label[:, None]
        mask = mask[:, None, :, None].expand(-1, self.num_prototypes, -1, -1)
        input_seq = input_seq[:, None, ...].expand(-1, self.num_prototypes, -1, -1)
        prototypes = self.prototypes[None].expand(batch_size, -1, -1, -1)
        distances = self.criterion(input_seq, prototypes, mask)
        distances = distances.reshape(batch_size, self.num_prototypes)
        _, indices = torch.min(distances, 1)  # N x K
        output_seq = torch.gather(prototypes, 1, indices[..., None, None, None].expand(-1, 1,
                                                                                       self.num_steps,
                                                                                       self.input_dim)
                                  ).squeeze(1)
        input_seq = input_seq[:, 0, ...]  # N x T x C
        label = label.view(batch_size)  # N
        mask = mask[:, 0, ...].view(batch_size, self.num_steps)  # N x T
        return output_seq, input_seq, distances, indices, label, mask

    def reassign_empty_clusters(self, proportions):
        idx = np.argmax(proportions)
        reassigned = []
        for i in range(self.num_prototypes):
            if proportions[i] < self.empty_cluster_threshold:
                self.restart_branch_from(i, idx)
                reassigned.append(i)
        if len(reassigned) > 0:
            print("      Reassigned clusters {} from cluster {}".format(reassigned, idx))
            self.restart_branch_from(idx, idx)
        return reassigned, idx

    def restart_branch_from(self, i, j, ):
        self.prototypes[i].data.copy_(self.prototypes[j])
        with torch.no_grad():
            self.prototypes[i].add_(torch.randn(self.prototypes[i].size(), device=self.prototypes[i].device) * NOISE_SCALE)

        if hasattr(self, 'optimizer'):
            opt = self.optimizer
            if isinstance(opt, (Adam,)):
                param = self.prototypes
                opt.state[param]['exp_avg'][i] = opt.state[param]['exp_avg'][j]
                opt.state[param]['exp_avg_sq'][i] = opt.state[param]['exp_avg_sq'][j]
            else:
                raise NotImplementedError('unknown optimizer: you should define how to reinstanciate statistics if any')
