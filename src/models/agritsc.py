import torch
import torch.nn as nn
import numpy as np

from torch.optim import Adam
from src.utils.tps import TPSGrid
from src.models.fcn_ts import FCNBaseline

NOISE_SCALE = 0.0001


class AgriSits(nn.Module):
    def __init__(
            self,
            input_dim=4,
            num_steps=365,
            num_classes=9,
            num_prototypes=32,
            empty_cluster_threshold=0.02,
            feature_size=128,
            supervised=False,
            amplitude=0.5,
            init_proto='random',
            sample=None,
            missing_dates=False,
            model_path=None,
    ):
        super(AgriSits, self).__init__()
        self.input_dim = input_dim
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.empty_cluster_threshold = empty_cluster_threshold / num_prototypes
        self.feature_size = feature_size
        self.supervised = supervised
        self.amplitude = amplitude
        self.init_proto = init_proto
        self.sample = sample
        self.missing_dates = missing_dates
        self.model_path = model_path

        self.num_control_points = self.num_steps // 30 - 1
        self.num_attention = self.num_control_points + 2

        # Init Gaussian Filter if missing dates
        if self.missing_dates:
            weights = torch.tensor([[gaussian(date, step, 7) for date in range(self.num_steps)]
                                    for step in range(self.num_steps)], dtype=torch.float, device='cuda')
            self.register_buffer('weights', weights)

        # Init Prototypes
        proto_init = self.initialize_prototypes()
        self.prototypes = torch.nn.Parameter(proto_init, requires_grad=True)

        # Masked distance to minimize
        self.criterion = lambda x, y, m: (m * (x - y) ** 2).sum((2, 3), keepdim=True)

        # Init FCN encoder
        self.encoder = FCNBaseline(self.input_dim, self.feature_size)

        # Init Time Warping Module
        self.trans_predictor = torch.nn.ModuleList([nn.Linear(self.feature_size, self.num_prototypes)
                                                    for _ in range(self.num_control_points)])
        for module in self.trans_predictor:
            torch.nn.init.zeros_(module.weight)
            torch.nn.init.zeros_(module.bias)
        a, b = torch.meshgrid([torch.linspace(-1, 1, 2), torch.linspace(-1, 1, self.num_control_points + 2)])
        target_control_points = torch.stack([a.flatten(), b.flatten()], dim=1)
        self.register_buffer('target_control_points', target_control_points)
        self.tps_grid = TPSGrid([self.num_steps, 1], self.target_control_points)

        # Init Offset Module
        self.offset_predictor = nn.Linear(self.feature_size, self.num_prototypes * self.input_dim)
        torch.nn.init.zeros_(self.offset_predictor.weight)
        torch.nn.init.zeros_(self.offset_predictor.bias)

        # Init Scaling Module
        self.scale_predictor = nn.Linear(self.feature_size, self.num_prototypes * self.input_dim)
        torch.nn.init.zeros_(self.scale_predictor.weight)
        torch.nn.init.zeros_(self.scale_predictor.bias)

        self.decoders_n_params = {'trans': 1,
                                  'offset': self.input_dim}

    def forward(self, input_seq, label, mask, trans_activ=True, offset_activ=True, return_all=False):
        batch_size = input_seq.size(0)
        feature = self.encoder(input_seq)[:, None].expand(-1, self.num_attention, -1)
        feature_trans, feature_offset, feature_scale = torch.split(feature, [self.num_control_points, 1, 1], dim=1)
        if self.missing_dates:
            input_seq, mask = self.fill_missing_dates(input_seq, mask)
        mask = mask[:, None, :, None].expand(-1, self.num_prototypes, -1, -1)
        input_seq = input_seq[:, None, ...].expand(-1, self.num_prototypes, -1, -1)
        prototypes = self.prototypes[None].expand(batch_size, -1, -1, -1)
        if trans_activ:
            prototypes = self.translate_prototypes(prototypes, feature_trans, batch_size)
        if offset_activ:
            prototypes = self.offset_prototypes(prototypes, feature_offset, batch_size)
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
        if return_all:
            return output_seq, input_seq, distances, indices, label, mask, prototypes
        return output_seq, input_seq, distances, indices, label, mask


    def initialize_prototypes(self):
        if self.init_proto == 'sample':
            sample, sample_mask = self.sample
            if self.missing_dates:
                sample, sample_mask = self.fill_missing_dates(sample, sample_mask)
            return sample
        elif self.init_proto == 'random':
            proto_init = torch.randn(self.num_prototypes, self.num_steps, self.input_dim)
            return proto_init
        elif self.init_proto == 'load':
            proto_init = torch.load(self.model_path)["state_dict"]["module.prototypes"]
            return proto_init
        else:
            raise NameError(self.init_proto)

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

        for predictor_name, predictor in zip(['trans' for _ in range(self.num_control_points)] + ['offset'],
                                             [self.trans_predictor[k] for k in range(self.num_control_points)] + [self.offset_predictor, self.scale_predictor]):
            fs = [lambda l: (self.decoders_n_params[predictor_name] * l + n) for n in range(self.decoders_n_params[predictor_name])]
            for param in ["weight", "bias"]:
                for f in fs:
                    getattr(predictor, param)[f(i)].data.copy_(torch.clone(getattr(predictor, param)[j]))
                    with torch.no_grad():
                        getattr(predictor, param)[f(i)].add_(
                            torch.randn(getattr(predictor, param)[f(i)].size(), device=self.prototypes[i].device) * NOISE_SCALE)
                    if hasattr(self, 'optimizer'):
                        for exp in ["exp_avg", "exp_avg_sq"]:
                            if hasattr(self.optimizers().state[getattr(predictor, param)], exp):
                                self.optimizers().state[getattr(predictor, param)][exp][f(i)].data.copy_(torch.clone(self.optimizers().state[getattr(predictor, param)][exp][f(j)]))

    def translate_prototypes(self, prototypes, feature, batch_size):
        amplitude = self.amplitude * 0.5 * 2. / (self.num_control_points + 1)
        time_map_trans = torch.cat([torch.tanh(self.trans_predictor[k](feature[:, k])).reshape(batch_size, self.num_prototypes, 1) * amplitude for k in range(self.num_control_points)], dim=2)  # B x K x CP
        time_map_trans = time_map_trans.reshape(batch_size * self.num_prototypes, self.num_control_points)  # (B*K) x CP
        time_map_trans = torch.nn.functional.pad(time_map_trans, [1, 1], mode='constant')  # (B*K) x (CP+2)
        time_map_trans = torch.stack([torch.zeros_like(time_map_trans), time_map_trans], dim=2)  # (B*K) x (CP+2) x 2
        time_map_trans = torch.cat([time_map_trans, time_map_trans], dim=1)  # (B*K) x 2*(CP+2) x 2
        source_control_points = time_map_trans + self.target_control_points
        grid = self.tps_grid(source_control_points).view(batch_size * self.num_prototypes, self.num_steps, 1, 2)
        prototypes = prototypes.permute(0, 1, 3, 2).reshape(batch_size * self.num_prototypes, self.input_dim, self.num_steps, 1)
        prototypes = torch.nn.functional.grid_sample(prototypes, grid, align_corners=True, padding_mode='border', mode='bicubic')
        prototypes = prototypes.reshape(batch_size, self.num_prototypes, self.input_dim, self.num_steps).permute(0, 1, 3, 2)  # B x K x T x C
        return prototypes

    def offset_prototypes(self, prototypes, feature, batch_size):
        offset = torch.tanh(self.offset_predictor(feature.squeeze(1)).view(batch_size, self.num_prototypes, 1, self.input_dim))
        prototypes = prototypes + offset
        return prototypes

    def fill_missing_dates(self, input, mask):
        input = torch.einsum('ti,bic->btc', self.weights, input)
        mask = torch.einsum('ti,bi->bt', self.weights, mask.float())
        mask_nonzero = torch.where(mask[..., None] == 0,
                                   torch.ones_like(mask[..., None]),
                                   mask[..., None])
        input = input / mask_nonzero
        return input, mask


def gaussian(x, mu, sig=1):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) / (sig * np.sqrt(2 * np.pi))
