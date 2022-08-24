import torch
from torch import nn


class TPSGrid(nn.Module):
    """Original implem: https://github.com/WarBean/tps_stn_pytorch"""

    def __init__(self, img_size, target_control_points):
        super().__init__()
        img_height, img_width = img_size
        N = target_control_points.size(0)
        self.num_points = N
        target_control_points = target_control_points.float()

        # create padded kernel matrix
        forward_kernel = torch.zeros(N + 3, N + 3)
        target_control_partial_repr = self.compute_partial_repr(target_control_points, target_control_points)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, -3].fill_(1)
        forward_kernel[-3, :N].fill_(1)
        forward_kernel[:N, -2:].copy_(target_control_points)
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))
        inverse_kernel = torch.inverse(forward_kernel)

        # create target cordinate matrix
        HW = img_height * img_width
        y, x = torch.meshgrid(torch.linspace(-1, 1, img_height), torch.linspace(-1, 1, img_width))
        target_coordinate = torch.stack([x.flatten(), y.flatten()], 1)
        target_coordinate_partial_repr = self.compute_partial_repr(target_coordinate, target_control_points)
        target_coordinate_repr = torch.cat([target_coordinate_partial_repr, torch.ones(HW, 1), target_coordinate], 1)

        # register precomputed matrices
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding_matrix', torch.zeros(3, 2))
        self.register_buffer('target_coordinate_repr', target_coordinate_repr)

    @staticmethod
    def compute_partial_repr(input_points, control_points):
        """Compute radial basis kernel phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2"""
        N = input_points.size(0)
        M = control_points.size(0)
        pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
        pairwise_diff_square = pairwise_diff * pairwise_diff
        pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
        repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
        repr_matrix.masked_fill_(repr_matrix != repr_matrix, 0)
        return repr_matrix

    def forward(self, source_control_points):
        Y = torch.cat([source_control_points, self.padding_matrix.expand(source_control_points.size(0), 3, 2)], 1)
        mapping_matrix = torch.matmul(self.inverse_kernel, Y)
        source_coordinate = torch.matmul(self.target_coordinate_repr, mapping_matrix)
        return source_coordinate

