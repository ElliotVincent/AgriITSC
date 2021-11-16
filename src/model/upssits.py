"""
U-TAE Implementation
Author: Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""
import torch
import torch.nn as nn
import numpy as np

from torch.optim import Adam

from src.model.utae import UTAE

N_HIDDEN_UNITS = 128
N_LAYERS = 2
N_CLASSES = 20
NOISE_SCALE = 0.0001


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
        empty_cluster_threshold=0.1,
        constant_map=False,
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
        self.constant_map = constant_map
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
            return_maps=True,
            pad_value=self.pad_value,
            padding_mode=padding_mode,
        )
        self.sequence_embedding_size = 256
        self.value = value or 0.5
        self.threshold = 0.99
        self.feature_size = self.sequence_embedding_size + 2  # F
        self.regressor_col_list = [create_mlp(self.feature_size, self.input_dim * 2, N_HIDDEN_UNITS, N_LAYERS).cuda()
                                   for _ in range(N_CLASSES)]
        self.register_buffer('identity_col', torch.eye(self.input_dim, self.input_dim))
        [regressor[-1].weight.data.zero_() for regressor in self.regressor_col_list]
        [regressor[-1].bias.data.zero_() for regressor in self.regressor_col_list]
        self.regressor_col_list = nn.ModuleList(self.regressor_col_list)
        self.criterion = nn.MSELoss(reduction='none')
        self.empty_cluster_threshold = empty_cluster_threshold / N_CLASSES
        self.encoder_regressor = create_mlp(self.encoder_widths[-1] * (self.input_size[0] // (2**(self.n_stages-1)))**2,
                                            self.sequence_embedding_size,
                                            self.sequence_embedding_size * 4, 1).cuda()

    def forward(self, input_seq, gt_map, batch_positions, return_recons=False, use_gt=True, return_map_att=False):
        intensity_map, att, feature_map = self.temporal_encoder(input_seq, batch_positions, return_att=True)
        if self.constant_map:
            intensity_map = torch.ones_like(intensity_map, device=intensity_map.device)
        device = input_seq.device
        identity_col = self.identity_col.to(device)
        num_dates = batch_positions.shape[1]
        batch_size = batch_positions.shape[0]
        dates_mask = (batch_positions > 0).view(batch_size, num_dates, 1, 1, 1)
        feature = self.encoder_regressor(feature_map.flatten(1)).unsqueeze(1).expand(-1, num_dates, -1)  # B x T x F
        att = torch.mean(att.permute(1, 2, 0, 3, 4), (2, 3, 4)).unsqueeze(2)  # B x T x 1
        out = intensity_map.unsqueeze(1).expand(-1, num_dates, self.input_dim, -1, -1)  # B x T x C x H x W
        feature_dates = torch.cat([feature, att, batch_positions.unsqueeze(2)], dim=-1)  # B x T x (F + 2)
        beta_list_col = [regressor_col(feature_dates) for regressor_col in self.regressor_col_list]
        weight_bias_list = [torch.split(beta_col.view(batch_size, num_dates, self.input_dim, 2),
                                        [1, 1], dim=3) for beta_col in beta_list_col]
        weight_bias_list = [[w.expand(-1, -1, -1, self.input_dim) * identity_col + identity_col,
                             b.unsqueeze(-1).expand(-1, -1, -1, out.size(3),
                                                    out.size(4))] for w, b in weight_bias_list]
        recons = torch.stack([torch.einsum('btij, btjkl -> btikl', w, out) + b
                              for w, b in weight_bias_list], dim=1)  # B x K x T x C x H x W

        weights = torch.mean(input_seq[:, :, :3, :, :] * dates_mask, 2, keepdim=True)
        mean, std = torch.mean(weights.flatten(1), dim=1), torch.std(weights.flatten(1), dim=1)
        mask_sup, mask_inf = weights < mean + std, weights > mean - std
        mask = mask_sup == mask_inf
        mask = torch.sum(mask, (2, 3, 4), keepdim=True) / (self.input_size[0]*self.input_size[1]*1.) > self.threshold
        mask = mask * dates_mask
        if use_gt:
            indices = gt_map.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        else:
            temporal_sum_loss = self.criterion(mask * input_seq.unsqueeze(1),
                                               mask * recons).mean(3, keepdim=True).mean(2, keepdim=True)
            _, indices = torch.min(temporal_sum_loss, 1, keepdim=True)
        output = torch.gather(recons, 1, indices.expand(-1, -1, num_dates, self.input_dim, -1, -1)).squeeze(1)
        indices = indices.squeeze().unsqueeze(0)
        if return_recons:
            return output, intensity_map, indices, recons
        elif return_map_att:
            return output, intensity_map, indices, feature_map, att
        else:
            return output, intensity_map, indices, mask

    def reassign_empty_clusters(self, proportions):
        idx = np.argmax(proportions)
        reassigned = []
        for i in range(N_CLASSES):
            if proportions[i] < self.empty_cluster_threshold:
                self.restart_branch_from(i, idx)
                reassigned.append(i)
        if len(reassigned) > 0:
            self.restart_branch_from(idx, idx)
        return reassigned, idx

    def restart_branch_from(self, i, j, ):
        self.regressor_col_list[i].load_state_dict(self.regressor_col_list[j].state_dict())
        with torch.no_grad():
            for param in self.regressor_col_list[i].parameters():
                param.add_(torch.randn(param.size(), device=param.device) * NOISE_SCALE)

        if hasattr(self, 'optimizer'):
            opt = self.optimizer
            if isinstance(opt, (Adam,)):
                param = self.regressor_col_list
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
