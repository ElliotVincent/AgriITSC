import torch
import torch.nn as nn
import numpy as np


class Ncc(nn.Module):
    def __init__(
            self,
            input_dim=4,
            num_steps=365,
            num_classes=9,
            num_prototypes=32,
            init_proto='random',
            sample=None,
            missing_dates=False,
            model_path=None,
    ):
        super(Ncc, self).__init__()
        self.input_dim = input_dim
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.init_proto = init_proto
        self.sample = sample
        self.missing_dates = missing_dates
        self.model_path = model_path

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

    def forward(self, input_seq, label, mask, return_all=False):
        batch_size = input_seq.size(0)
        if self.missing_dates:
            input_seq, mask = self.fill_missing_dates(input_seq, mask)
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
        if return_all:
            return output_seq, input_seq, distances, indices, label, mask, prototypes
        return output_seq, input_seq, distances, indices, label, mask

    def initialize_prototypes(self):
        if self.init_proto in ['sample']:
            sample, sample_mask = self.sample
            if self.missing_dates:
                sample, sample_mask = self.fill_missing_dates(sample, sample_mask)
            return sample
        elif self.init_proto == 'random':
            proto_init = torch.randn(self.num_prototypes, self.num_steps, self.input_dim)
            return proto_init
        elif self.init_proto in ['kmeans', 'means', 'means_gaussian', 'means_previous', 'means_next', 'means_avg']:
            return self.sample
        else:
            raise NameError(self.init_proto)

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
