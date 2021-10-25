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
from src.model.ltae import LTAE2d
from src.model.blocks import ConvBlock, DownConvBlock


N_HIDDEN_UNITS = 128
N_LAYERS = 2
N_CLASSES = 4


class UPSSITS(nn.Module):
    def __init__(
        self,
        input_dim,
        input_size=(128, 128),
        encoder_widths=[64, 64, 64, 128],
        num_dates=61,
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

        self.in_conv = ConvBlock(
            nkernels=[input_dim] + [encoder_widths[0], encoder_widths[0]],
            pad_value=pad_value,
            norm=encoder_norm,
            padding_mode=padding_mode,
        )
        self.down_blocks = nn.ModuleList(
            DownConvBlock(
                d_in=encoder_widths[i],
                d_out=encoder_widths[i + 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                pad_value=pad_value,
                norm=encoder_norm,
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1)
        )

        self.temporal_encoder = LTAE2d(
            in_channels=encoder_widths[-1],
            d_model=d_model,
            n_head=n_head,
            mlp=[d_model, encoder_widths[-1]],
            return_att=True,
            d_k=d_k,
        )
        self.value = value or 0.5
        self.proto = nn.Parameter(torch.full((1, 1, *self.input_size), self.value, dtype=torch.float))
        self.feature_size = self.encoder_widths[-1] * (self.input_size[0] // (2**(self.n_stages-1))) ** 2
        self.regressor_aff_list = [create_mlp(self.feature_size, 6, N_HIDDEN_UNITS, N_LAYERS).cuda()
                                   for _ in range(N_CLASSES)]

        self.regressor_col_list = [create_mlp(self.feature_size + 1, self.input_dim * 2, N_HIDDEN_UNITS, N_LAYERS).cuda()
                                   for _ in range(N_CLASSES)]
        self.register_buffer('identity_aff', torch.cat([torch.eye(2, 2), torch.zeros(2, 1)], dim=1))
        [regressor[-1].weight.data.zero_() for regressor in self.regressor_aff_list]
        [regressor[-1].bias.data.zero_() for regressor in self.regressor_aff_list]
        self.regressor_aff_list = nn.ModuleList(self.regressor_aff_list)

        self.register_buffer('identity_col', torch.eye(self.input_dim, self.input_dim))
        [regressor[-1].weight.data.zero_() for regressor in self.regressor_col_list]
        [regressor[-1].bias.data.zero_() for regressor in self.regressor_col_list]
        self.regressor_col_list = nn.ModuleList(self.regressor_col_list)

        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, input, segm_maps, batch_positions=None, return_att=False):
        device = input.device
        identity_aff = self.identity_aff.to(device)
        identity_col = self.identity_col.to(device)
        zero = torch.zeros((N_CLASSES, batch_positions.shape[1], self.input_dim, *self.input_size)).to(device)
        batch_positions = batch_positions.float()
        pad_mask = ((input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1))  # BxT pad mask
        out = self.in_conv.smart_forward(input)
        feature_maps = [out]
        # SPATIAL ENCODER
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i].smart_forward(feature_maps[-1])
            feature_maps.append(out)
        # TEMPORAL ENCODER
        out, att = self.temporal_encoder(feature_maps[-1], batch_positions=batch_positions, pad_mask=pad_mask)
        batch_size = input.shape[0]
        features = out.reshape(batch_size, -1)
        n_segm = [int(torch.max(segm_maps[k]).cpu()) + 1 for k in range(batch_size)]
        output = []
        for input_id in range(batch_size):
            curr_input = input[input_id]
            feature = features[input_id]
            segm_map = segm_maps[input_id]
            feature_dates = torch.cat([feature.expand(batch_positions.shape[1], -1),
                                       batch_positions[input_id].unsqueeze(1)], dim=-1) # T x F
            segm_list = [zero[0].unsqueeze(0)]
            for segm_id in range(1, n_segm[input_id]):
                beta_list_aff = [regressor_aff(feature).view(-1, 2, 3) + identity_aff for regressor_aff
                                in self.regressor_aff_list]
                grid_list = [F.affine_grid(beta, [self.proto.size(0), self.proto.size(1), self.input_size[0], self.input_size[1]],
                                           align_corners=False) for beta in beta_list_aff]
                transf_proto_list = torch.stack([F.grid_sample(self.proto, grid, mode='bilinear',
                                                               padding_mode='border', align_corners=False)
                                                 for grid in grid_list], dim=0).expand(-1, batch_positions.shape[1], -1, -1, -1)  # K x T x 1 x H x W
                transf_proto_list = transf_proto_list.expand(-1, -1, self.input_dim, -1, -1)  # K x T x C x H x W
                beta_list_col = [regressor_col(feature_dates) for regressor_col in self.regressor_col_list]
                weight_bias_list = [torch.split(beta_col.view(-1, self.input_dim, 2), [1, 1], dim=2) for beta_col
                                   in beta_list_col]
                weight_bias_list = [[w.expand(-1, -1, self.input_dim) * identity_col + identity_col,
                                    b.unsqueeze(-1).expand(-1, -1, transf_proto_list.size(3),
                                                           transf_proto_list.size(4))] for w, b in weight_bias_list]
                transf_proto = torch.stack([torch.einsum('bij, bjkl -> bikl', w, transf_proto_list[k]) + b
                                      for k, (w, b) in enumerate(weight_bias_list)], dim=0)  # K x T x C x H x W
                masked_transf_proto = torch.where(segm_map == segm_id, transf_proto, zero)
                masked_input = torch.where(segm_map == segm_id, curr_input.unsqueeze(0).repeat((N_CLASSES, 1, 1, 1, 1)), zero)
                temporal_sum_loss = self.criterion(masked_input, masked_transf_proto).flatten(2).mean(2).sum(1)
                selected_crop = torch.index_select(masked_transf_proto, 0, torch.argmin(temporal_sum_loss))  # T x C x H x W
                segm_list.append(selected_crop)
            output.append(torch.sum(torch.cat(segm_list, dim=0), dim=0, keepdim=True))
        output = torch.cat(output, dim=0) # B x T x C x H x W
        return output


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
