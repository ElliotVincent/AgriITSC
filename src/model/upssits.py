"""
U-TAE Implementation
Author: Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from src.model.ltae_seq import LTAE1d
from torch.optim import Adam
from src.utils.tps import TPSGrid
from src.utils.model_utils import get_proto_init


NOISE_SCALE = 0.0001


class UPSSITS(nn.Module):
    def __init__(
            self,
            input_dim=10,
            num_steps=406,
            num_classes=17,
            num_proto_per_class=4,
            loss="recons",
            learn_weights=False,
            feature_size=128,
            num_control_points=28,
            init_type='constant',
    ):
        super(UPSSITS, self).__init__()
        self.input_dim = input_dim
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.num_proto_per_class = num_proto_per_class
        self.num_prototypes = self.num_proto_per_class * self.num_classes
        self.loss = loss
        self.learn_weights = learn_weights
        self.feature_size = feature_size
        self.num_control_points = num_control_points
        self.init_type = init_type
        #
        init_weights = get_proto_init(self.num_prototypes, self.num_steps, self.input_dim, init_type=self.init_type)
        # model_res = './StepSize-1-Sigma-7-NDVI-False-v-18-00-37'
        # state_dict = torch.load(
        #     os.path.join(
        #         model_res, "Fold_{}".format(1), "model.pth.tar"
        #     )
        # )["state_dict"]
        # init_weights = state_dict["module.prototypes"][2:-1]
        # init_weights = torch.repeat_interleave(init_weights, self.num_proto_per_class, dim=0)

        self.encoder = LTAE1d(
            in_channels=self.input_dim,
            n_head=16,
            d_k=4,
            mlp=[256, 128],
            dropout=0.2,
            d_model=256,
            return_att=False
        )
        self.weights = torch.nn.Parameter(torch.ones((self.num_steps,
                                                      self.input_dim)) * (1./(self.num_steps*self.input_dim)),
                                          requires_grad=False)
        self.prototypes = torch.nn.Parameter(torch.randn(self.num_classes, self.num_steps, self.input_dim), requires_grad=True)

        self.trans_predictor = create_mlp(self.feature_size, self.num_prototypes * self.num_control_points * 2, 0, 0)
        self.app_predictor = create_mlp(self.feature_size, self.num_prototypes * self.input_dim, 0, 0)
        a, b = torch.meshgrid([torch.linspace(-1, 1, self.num_control_points + 2), torch.linspace(-1, 1, 2)])
        target_control_points = torch.stack([a.flatten(), b.flatten()], dim=1)
        self.register_buffer('target_control_points', target_control_points)
        self.tps_grid = TPSGrid([self.num_steps, 1], self.target_control_points)

        self.proto_count = torch.nn.Parameter(torch.zeros(self.num_prototypes), requires_grad=False)

        # if self.learn_weights:
        #     self.weights = torch.nn.Parameter(torch.randn((self.num_steps, self.input_dim)), requires_grad=True)
        #     self.prototypes = torch.nn.Parameter(init_weights, requires_grad=False)
        # else:
        #     self.weights = torch.nn.Parameter(torch.ones((self.num_steps, self.input_dim)), requires_grad=False)
        #     self.prototypes = torch.nn.Parameter(init_weights, requires_grad=True)

        if self.loss == 'recons':
            self.beta = torch.nn.Parameter(torch.tensor(-10.), requires_grad=False)
            self.criterion = lambda x, y, m, w, b:((m * (x-y)**2) * (F.softplus(
                w[None, None].expand(x.size(0), x.size(1), -1, -1))/F.softplus(
                w[None, None].expand(x.size(0), x.size(1), -1, -1)).sum((2, 3), keepdim=True))
                                                  ).sum((2, 3), keepdim=True)

        elif self.loss == 'ce':
            self.beta = torch.nn.Parameter(torch.tensor(-10.), requires_grad=True)
            self.criterion = lambda x, y, w, b: ((((x-y)**2) * (F.softplus(
                w[None, None].expand(x.size(0), x.size(1), -1, -1))/F.softplus(
                w[None, None].expand(x.size(0), x.size(1), -1, -1)).sum((2, 3), keepdim=True))
                                                  ).sum((2,3), keepdim=True))

        elif self.loss == 'logreg':
            self.beta = torch.nn.Parameter(torch.tensor(-10.), requires_grad=True)
            self.criterion = lambda x, y, w, b: -1. * (x*y).sum((2, 3), keepdim=True) / (
                    torch.sqrt((x ** 2).sum((2, 3), keepdim=True))
                    * torch.sqrt((y ** 2).sum((2, 3), keepdim=True)))

        elif self.loss == 'mixed':
            self.beta = torch.nn.Parameter(torch.tensor(-10.), requires_grad=True)
            self.criterion_logreg = lambda x, y, w, b: - F.softplus(b) * (-1. * (x*y).sum((2, 3), keepdim=True))
            self.criterion_logreg_prob = lambda x, y, w, b: - torch.log(F.softmax(self.criterion_logreg(x, y, w, b), dim=1))
            self.criterion_recons = lambda x, y, w: ((x-y)**2).mean(3, keepdim=True).mean(2, keepdim=True)
            self.criterion = lambda x, y, w, b: self.criterion_logreg_prob(x, y, w, b) + 0.1 * self.criterion_recons(x, y, w)

    def forward(self, input_seq, label, mask):
        batch_size = input_seq.size(0)
        # feature = self.encoder(input_seq)
        label = label[:, None]
        mask = mask[:, None, :, None].expand(-1, self.num_prototypes, -1, -1)
        input_seq = input_seq[:, None, ...].expand(-1, self.num_prototypes, -1, -1)
        prototypes = self.prototypes[None].expand(batch_size, -1, -1, -1)
        # prototypes = self.change_appearance(prototypes, feature, batch_size)
        # prototypes = self.translate_prototypes(prototypes, feature, batch_size)
        distances = self.criterion(input_seq, prototypes, mask, self.weights, self.beta)
        distances = distances.reshape(batch_size, self.num_classes, self.num_proto_per_class)
        loss, predictions = torch.min(distances, 2)  # N x K x 1 x 1 x 1
        indices = torch.gather(predictions, 1, label)
        indices = label * self.num_proto_per_class + indices
        indices = indices[..., None, None]
        elem, elem_counts = torch.unique(indices, return_counts=True)
        self.proto_count[elem] += elem_counts
        output_seq = torch.gather(prototypes, 1, indices.expand(-1, -1, self.num_steps, self.input_dim)).squeeze(1)
        input_seq = input_seq[:, 0, ...]  # N x T x C
        loss = loss.view(batch_size, self.num_classes)  # N x K
        indices = torch.floor_divide(indices.view(batch_size), self.num_proto_per_class)  # N
        label = label.view(batch_size)  # N
        mask = mask[:, 0, ...].view(batch_size, self.num_steps)  # N
        if self.loss == 'mixed':
            loss = self.criterion_logreg(input_seq[:, None, ...].expand(-1, self.num_classes, -1, -1),
                                         self.prototypes[None].expand(batch_size, -1, -1, -1),
                                         self.weights, self.beta).view(batch_size, self.num_classes)
        return output_seq, input_seq, loss, indices, label, mask

    def translate_prototypes(self, prototypes, feature, batch_size):
        time_map_trans = torch.clamp(self.trans_predictor(feature), -2, 2)
        time_map_trans = time_map_trans.reshape(batch_size, self.num_classes * self.num_control_points * 2)  # B x (K*CP*2)
        time_map_trans = time_map_trans.reshape(batch_size, self.num_classes, self.num_control_points, 2)  # B x K x CP x 2
        time_map_trans = time_map_trans.reshape(batch_size * self.num_classes, self.num_control_points * 2)  # (B*K) x (CP*2)
        time_map_trans = torch.nn.functional.pad(time_map_trans, [2, 2], mode='constant')  # B*K x 2*(CP+2)
        time_map_trans = torch.stack([torch.zeros_like(time_map_trans), time_map_trans], dim=2)  # (B*K) x 2*(CP+2) x 2
        source_control_points = time_map_trans + self.target_control_points
        grid = self.tps_grid(source_control_points).view(batch_size * self.num_prototypes, self.num_steps, 1, 2)
        prototypes = prototypes.permute(0, 1, 3, 2).reshape(batch_size * self.num_prototypes, self.input_dim, self.num_steps, 1)
        prototypes = torch.nn.functional.grid_sample(prototypes, grid, align_corners=True, padding_mode='border')
        prototypes = prototypes.reshape(batch_size, self.num_prototypes, self.input_dim, self.num_steps).permute(0, 1, 3, 2)  # B x K x T x C
        return prototypes

    def change_appearance(self, prototypes, feature, batch_size):
        appearances = torch.relu(self.app_predictor(feature)).reshape(batch_size, self.num_prototypes, 1, self.input_dim)
        appearances = appearances.expand(-1, -1, self.num_steps, -1)
        return prototypes * appearances

    def reassign_empty_clusters(self, proportions):
        all_reassigned = []
        idxs = []
        for i in range(self.num_classes):
            reassigned = []
            idx = np.argmax(proportions[i])
            for j in range(self.num_proto_per_class):
                if proportions[i][j] < self.empty_cluster_threshold:
                    self.restart_branch_from(i * self.num_proto_per_class + j, i * self.num_proto_per_class + idx)
                    reassigned.append(i * self.num_proto_per_class + j)
            if len(reassigned) > 0:
                self.restart_branch_from(i * self.num_proto_per_class + idx, i * self.num_proto_per_class + idx)
            all_reassigned.append(reassigned)
            idxs.append(idx)
        return all_reassigned, idxs

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


def create_mlp(in_ch, out_ch, n_hidden_units, n_layers):
    if n_layers > 0:
        seq = [nn.Linear(in_ch, n_hidden_units), nn.ReLU(True)]
        for _ in range(n_layers - 1):
            seq += [nn.Linear(n_hidden_units, n_hidden_units), nn.ReLU(True)]
        seq += [nn.Linear(n_hidden_units, out_ch)]
    else:
        seq = [nn.Linear(in_ch, out_ch)]
    return nn.Sequential(*seq)