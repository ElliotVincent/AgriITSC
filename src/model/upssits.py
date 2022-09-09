"""
U-TAE Implementation
Author: Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""
import os

import torch
import torch.nn as nn
import numpy as np

from torch.optim import Adam
from src.utils.tps import TPSGrid
from src.model.fcn_ts import FCNBaseline
from src.utils.path import RESULTS_PATH
from src.utils.supervised_utils import gaussian

NOISE_SCALE = 0.0001


class AgriSits(nn.Module):
    def __init__(
            self,
            input_dim=10,
            num_steps=406,
            num_classes=19,
            num_prototypes=128,
            empty_cluster_threshold=0.02,
            dataset_name='pastis',
            feature_size=128,
            learn_weights=False,
            init_proto=True,
            n_heads=16,
            mlp=[256, 128],
            d_k=4,
            dropout=0.2,
            d_model=256,
            supervised=False,
    ):
        super(AgriSits, self).__init__()
        self.empty_cluster_threshold = empty_cluster_threshold / num_prototypes
        self.input_dim = input_dim
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.feature_size = feature_size
        self.num_prototypes = num_prototypes
        if dataset_name[:-1] == 'pastis':
            self.fold = dataset_name[-1]
            self.dataset_name = dataset_name[:-1]
        else:
            self.dataset_name = dataset_name
        self.num_control_points = self.num_steps // 30 - 1
        self.num_attention = self.num_control_points + 2
        self.supervised = supervised

        proto_init = torch.randn(self.num_prototypes, self.num_steps, self.input_dim)
        if init_proto:
            if supervised:
                if dataset_name == 'pastis':
                    model_path = os.path.join(RESULTS_PATH, self.dataset_name, 'pastis_means_supervised_gaussian', f'Fold_{self.fold}', 'model.pth.tar')
                elif dataset_name == 'ts2c':
                    model_path = os.path.join(RESULTS_PATH, self.dataset_name, 'supervised_0', 'model.pth.tar')
                else:
                    model_path = os.path.join(RESULTS_PATH, dataset_name, 'supervised_0', 'model.pth.tar')
                proto_init = torch.load(model_path)["state_dict"]["module.prototypes"]
            else:
                model_path = os.path.join(RESULTS_PATH, dataset_name, f'kmeans{num_prototypes}_1', 'model_last.pth.tar')
                proto_init = torch.load(model_path)["state_dict"]["module.prototypes"]
        self.prototypes = torch.nn.Parameter(proto_init, requires_grad=True)

        self.criterion = lambda x, y, m: (m * (x - y) ** 2).sum((2, 3), keepdim=True)

        self.encoder = FCNBaseline(self.input_dim, self.feature_size)

        self.trans_predictor = torch.nn.ModuleList([create_mlp(self.feature_size, self.num_prototypes * 2, 0, 0
                                                               ) for _ in range(self.num_control_points)])
        a, b = torch.meshgrid([torch.linspace(-1, 1, self.num_control_points + 2), torch.linspace(-1, 1, 2)])
        target_control_points = torch.stack([a.flatten(), b.flatten()], dim=1)
        self.register_buffer('target_control_points', target_control_points)
        self.tps_grid = TPSGrid([self.num_steps, 1], self.target_control_points)

        self.offset_predictor = create_mlp(self.feature_size, self.num_prototypes * self.input_dim, 0, 0)
        self.scale_predictor = create_mlp(self.feature_size, self.num_prototypes * self.input_dim, 0, 0)

        if self.dataset_name in ['ts2c', 'pastis']:
            weights = torch.tensor([[gaussian(date, step, 7) for date in range(self.num_steps)]
                                    for step in range(self.num_steps)], dtype=torch.float)
            self.register_buffer('weights', weights)

    def forward(self, input_seq, label, mask, trans_activ=True, offset_activ=True, scale_activ=True):
        batch_size = input_seq.size(0)
        feature = self.encoder(input_seq)[:, None].expand(-1, self.num_attention, -1)
        feature_trans, feature_offset, feature_scale = torch.split(feature, [self.num_control_points, 1, 1], dim=1)
        if self.dataset_name in ['ts2c', 'pastis']:
            input_seq = torch.einsum('ti,bic->btc', self.weights, input_seq)
            mask = torch.einsum('ti,bi->bt', self.weights, mask.float())
            mask_nonzero = torch.where(mask[..., None] == 0,
                                             torch.ones_like(mask[..., None]),
                                             mask[..., None])
            input_seq = input_seq / mask_nonzero
        mask = mask[:, None, :, None].expand(-1, self.num_prototypes, -1, -1)
        input_seq = input_seq[:, None, ...].expand(-1, self.num_prototypes, -1, -1)
        prototypes = self.prototypes[None].expand(batch_size, -1, -1, -1)
        if trans_activ:
            prototypes = self.translate_prototypes(prototypes, feature_trans, batch_size)
        if offset_activ:
            prototypes = self.offset_prototypes(prototypes, feature_offset, batch_size)
        if scale_activ:
            prototypes = self.scale_prototypes(prototypes, feature_scale, batch_size)
        distances = self.criterion(input_seq, prototypes, mask)
        distances = distances.reshape(batch_size, self.num_prototypes)
        if self.supervised:
            indices = label
        else:
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

    def translate_prototypes(self, prototypes, feature, batch_size):
        time_map_trans = torch.stack([torch.tanh(self.trans_predictor[k](feature[:, k])) * 5 for k in range(self.num_control_points)], dim=1)
        time_map_trans = time_map_trans.reshape(batch_size, self.num_control_points * 2, self.num_prototypes)  # B x K x CP x 2
        time_map_trans = time_map_trans.permute(0, 2, 1).reshape(batch_size * self.num_prototypes, -1) # (B*K) x (CP*2)
        time_map_trans = torch.nn.functional.pad(time_map_trans, [2, 2], mode='constant')  # B*K x 2*(CP+2)
        time_map_trans = torch.stack([torch.zeros_like(time_map_trans), time_map_trans], dim=2)  # (B*K) x 2*(CP+2) x 2
        source_control_points = time_map_trans + self.target_control_points
        grid = self.tps_grid(source_control_points).view(batch_size * self.num_prototypes, self.num_steps, 1, 2)
        prototypes = prototypes.permute(0, 1, 3, 2).reshape(batch_size * self.num_prototypes, self.input_dim, self.num_steps, 1)
        prototypes = torch.nn.functional.grid_sample(prototypes, grid, align_corners=True, padding_mode='border')
        prototypes = prototypes.reshape(batch_size, self.num_prototypes, self.input_dim, self.num_steps).permute(0, 1, 3, 2)  # B x K x T x C
        return prototypes

    def offset_prototypes(self, prototypes, feature, batch_size):
        offset = torch.tanh(self.offset_predictor(feature.squeeze(1)).view(batch_size, self.num_prototypes, 1, self.input_dim))
        prototypes = prototypes + offset
        return prototypes

    def scale_prototypes(self, prototypes, feature, batch_size):
        scale = torch.tanh(self.scale_predictor(feature.squeeze(1)).view(batch_size, self.num_prototypes, 1, self.input_dim)) * 0.75 + 1.25
        prototypes = prototypes * scale
        return prototypes


def create_mlp(in_ch, out_ch, n_hidden_units, n_layers):
    if n_layers > 0:
        seq = [nn.Linear(in_ch, n_hidden_units), nn.ReLU(True)]
        for _ in range(n_layers - 1):
            seq += [nn.Linear(n_hidden_units, n_hidden_units), nn.ReLU(True)]
        seq += [nn.Linear(n_hidden_units, out_ch)]
    else:
        seq = [nn.Linear(in_ch, out_ch)]
    return nn.Sequential(*seq)
