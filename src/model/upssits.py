"""
U-TAE Implementation
Author: Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
from abc import ABCMeta, abstractmethod
from src.model.utae import UTAE
from src.model.blocks import ConvBlock, DownConvBlock


N_HIDDEN_UNITS = 128
N_LAYERS = 2
N_CLASSES = 20


class UPSSITS(nn.Module):
    def __init__(
        self,
        input_dim,
        input_size=(128, 128),
        encoder_widths=[64, 64, 64, 128],
        str_conv_k=4,
        str_conv_s=2,
        str_conv_p=1,
        encoder_norm="group",
        n_head=16,
        d_model=256,
        d_k=4,
        pad_value=0,
        padding_mode="reflect",
        value=0.5,
    ):
        """
        U-TAE architecture for spatio-temporal encoding of satellite image time series.
        Args:
            input_dim (int): Number of channels in the input images.
            encoder_widths (List[int]): List giving the number of channels of the successive encoder_widths of the convolutional encoder.
            This argument also defines the number of encoder_widths (i.e. the number of downsampling steps +1)
            in the architecture.
            The number of channels are given from top to bottom, i.e. from the highest to the lowest resolution.
            decoder_widths (List[int], optional): Same as encoder_widths but for the decoder. The order in which the number of
            channels should be given is also from top to bottom. If this argument is not specified the decoder
            will have the same configuration as the encoder.
            out_conv (List[int]): Number of channels of the successive convolutions for the
            str_conv_k (int): Kernel size of the strided up and down convolutions.
            str_conv_s (int): Stride of the strided up and down convolutions.
            str_conv_p (int): Padding of the strided up and down convolutions.
            agg_mode (str): Aggregation mode for the skip connections. Can either be:
                - att_group (default) : Attention weighted temporal average, using the same
                channel grouping strategy as in the LTAE. The attention masks are bilinearly
                resampled to the resolution of the skipped feature maps.
                - att_mean : Attention weighted temporal average,
                 using the average attention scores across heads for each date.
                - mean : Temporal average excluding padded dates.
            encoder_norm (str): Type of normalisation layer to use in the encoding branch. Can either be:
                - group : GroupNorm (default)
                - batch : BatchNorm
                - instance : InstanceNorm
            n_head (int): Number of heads in LTAE.
            d_model (int): Parameter of LTAE
            d_k (int): Key-Query space dimension
            encoder (bool): If true, the feature maps instead of the class scores are returned (default False)
            return_maps (bool): If true, the feature maps instead of the class scores are returned (default False)
            pad_value (float): Value used by the dataloader for temporal padding.
            padding_mode (str): Spatial padding strategy for convolutional layers (passed to nn.Conv2d).
        """
        super(UPSSITS, self).__init__()
        self.input_dim = input_dim
        self.input_size = input_size
        self.n_stages = len(encoder_widths)
        self.encoder_widths = encoder_widths
        self.pad_value = pad_value

        self.temporal_encoder = UTAE(
            input_dim=self.input_dim,
            encoder_widths=self.encoder_widths,
            decoder_widths=[32, 32, 64, 128],
            out_conv=[32, 1],
            str_conv_k=str_conv_k,
            str_conv_s=str_conv_s,
            str_conv_p=str_conv_p,
            agg_mode="att_group",
            encoder_norm=encoder_norm,
            n_head=n_head,
            d_model=d_model,
            d_k=d_k,
            encoder=False,
            return_maps=False,
            pad_value=self.pad_value,
            padding_mode=padding_mode,
        )

        self.value = value or 0.5
        self.feature_size = n_head * (self.input_size[0] // (2**(self.n_stages-1))) ** 2 + 1  # F
        self.regressor_col_list = [create_mlp(self.feature_size, self.input_dim * 2, N_HIDDEN_UNITS, N_LAYERS).cuda()
                                   for _ in range(N_CLASSES)]
        self.register_buffer('identity_col', torch.eye(self.input_dim, self.input_dim))
        [regressor[-1].weight.data.zero_() for regressor in self.regressor_col_list]
        [regressor[-1].bias.data.zero_() for regressor in self.regressor_col_list]
        self.regressor_col_list = nn.ModuleList(self.regressor_col_list)

        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, input, batch_positions=None, return_att=True):
        intensity_map, att = self.temporal_encoder(input, batch_positions, return_att)
        device = input.device
        identity_col = self.identity_col.to(device)
        num_dates = batch_positions.shape[1]
        batch_size = batch_positions.shape[0]
        batch_positions = batch_positions.float()
        att = att.permute(1, 2, 0, 3, 4).reshape(batch_size, num_dates, -1)
        out = intensity_map.unsqueeze(1).expand(-1, num_dates, self.input_dim, -1, -1)
        feature_dates = torch.cat([att, batch_positions.unsqueeze(2)], dim=-1)  # .view(batch_size * num_dates, -1)  # B x T x F
        beta_list_col = [regressor_col(feature_dates) for regressor_col in self.regressor_col_list]
        weight_bias_list = [torch.split(beta_col.view(batch_size, num_dates, self.input_dim, 2), [1, 1], dim=3) for beta_col
                            in beta_list_col]
        weight_bias_list = [[w.expand(-1, -1, -1, self.input_dim) * identity_col + identity_col,
                             b.unsqueeze(-1).expand(-1, -1, -1, out.size(3),
                                                    out.size(4))] for w, b in weight_bias_list]
        recons = torch.stack([torch.einsum('btij, btjkl -> btikl', w, out) + b
                                    for w, b in weight_bias_list], dim=1)  # B x K x T x C x H x W
        temporal_sum_loss = self.criterion(input.unsqueeze(1), recons).mean(2, keepdim=True)
        _, indices = torch.min(temporal_sum_loss, 1, keepdim=True)
        output = torch.gather(recons, 1, indices.expand(-1, -1, num_dates, -1, -1, -1)).squeeze(1)
        return output, intensity_map


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


def create_mlp(in_ch, out_ch, n_hidden_units, n_layers, norm_layer=None):
    if norm_layer is None or norm_layer in ['id', 'identity']:
        norm_layer = Identity
    elif norm_layer in ['batch_norm', 'bn']:
        norm_layer = nn.BatchNorm1d
    elif not norm_layer == nn.BatchNorm1d:
        raise NotImplementedError

    if n_layers > 0:
        seq = [nn.Linear(in_ch, n_hidden_units), norm_layer(n_hidden_units), nn.ReLU(True)]
        for _ in range(n_layers - 1):
            seq += [nn.Linear(n_hidden_units, n_hidden_units), nn.ReLU(True)]
        seq += [nn.Linear(n_hidden_units, out_ch)]
    else:
        seq = [nn.Linear(in_ch, out_ch)]
    return nn.Sequential(*seq)
