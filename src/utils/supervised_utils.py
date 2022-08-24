import json
import os
import numpy as np
import torch
import visdom
import scipy.interpolate as interp
import matplotlib.pyplot as plt

FIRST_DATE = 16
LAST_DATE = 421
CLOUD_THRESHOLD = 0.999
IMG_SIZE = 128
FIRST_VAL = torch.tensor([[-0.2521, -0.2088, -0.1168, -0.1525, -0.3105, -0.3568, -0.3413, -0.3428, 0.1995,  0.2402]])
LAST_VAL = torch.tensor([[-0.2943, -0.2658, -0.2373, -0.2527, -0.3119, -0.3381, -0.3675, -0.3503, -0.1666, -0.1314]])
NORMS = json.load(open('/home/vincente/upssits/datasets/PASTIS/NORM_S2_patch.json', 'r'))

def cloud_filtering(input_seq, dates, sigma=1., threshold=CLOUD_THRESHOLD):
    dates_mask = (dates > 0).view(input_seq.size(0), input_seq.size(1), 1, 1, 1).float()
    weights = input_seq[:, :, 0, :, :].unsqueeze(2)
    mean, std = torch.mean(weights, dim=1, keepdim=True), torch.std(weights, dim=1, keepdim=True)
    mask_sup, mask_inf = weights < (mean + sigma*std), weights > (mean - sigma*std)
    mask = mask_sup == mask_inf
    mask_mean = torch.sum(mask, (2, 3, 4), keepdim=True) / (IMG_SIZE * IMG_SIZE * 1.) > threshold
    dates_mask = dates_mask * mask_mean  # B x T x 1 x 1 x 1
    mask = dates_mask.expand(-1, -1, -1, IMG_SIZE, IMG_SIZE)
    return mask


class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name):
        self.viz = visdom.Visdom(port=8889, env=env_name)
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name],
                          name=split_name, update='append')

    def histogram(self, var_name, split_name, title_name, x):
        self.viz.image(np.tile(np.array(x), (3, 1, 1))/np.max(np.array(x)), env=self.env, win=var_name, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                store_history=True,
            ))

    def bar(self, var_name, split_name, title_name, x):
        self.viz.bar(x, env=self.env, win=var_name, opts=dict(
            title=title_name,
            xlabel='Epochs',
            store_history=True,
        ))


def prepare_inputs(input_seq, mask, dates, label, device, fold=1, mode='none',
                   FIRST_VAL=FIRST_VAL, LAST_VAL=LAST_VAL):
    """
    Available modes:
        - previous: previous value interpolation
        - next:     next value interpolation
        - nearest:  nearest value interpolation
        - linear:   linear interpolation
        - mov_avg:  moving average
        - gaussian: weighted moving average with gaussian weights

    """
    FIRST_VAL = FIRST_VAL.to(device)
    LAST_VAL = LAST_VAL.to(device)
    batch_size = input_seq.size(0)
    num_dates = input_seq.size(1)
    num_pixel_seq = batch_size * IMG_SIZE ** 2
    red_mean, red_std = NORMS[f'Fold_{fold}']['mean'][2], NORMS[f'Fold_{fold}']['std'][2]
    nir_mean, nir_std = NORMS[f'Fold_{fold}']['mean'][6], NORMS[f'Fold_{fold}']['std'][6]
    ndvi = torch.where(((input_seq[:, :, 2, ...] * red_std + red_mean) == (input_seq[:, :, 6, ...] * nir_std + nir_mean) * ((input_seq[:, :, 2, ...] * red_std + red_mean) == 0)),
                       torch.zeros_like(input_seq[:, :, 0, ...]),
                       ((input_seq[:, :, 6, ...] * nir_std + nir_mean) -
                       (input_seq[:, :, 2, ...] * red_std + red_mean)) /
                       ((input_seq[:, :, 6, ...] * nir_std + nir_mean) +
                        (input_seq[:, :, 2, ...] * red_std + red_mean))
                       )[:, :, None, ...]
    input_seq = torch.cat([input_seq, ndvi], dim=2)
    dates = dates.permute(1, 0)[..., None].expand(-1, -1, IMG_SIZE ** 2).flatten(1).permute(1, 0).long().flatten()
    dates = dates - FIRST_DATE
    label = label.flatten()
    seq = input_seq.permute(2, 1, 0, 3, 4).flatten(2).permute(2, 1, 0).permute(2, 0, 1).flatten(1).permute(1, 0)
    mask = mask.permute(1, 0, 3, 4, 2).flatten(1).permute(1, 0).flatten()
    seq_indices = torch.tensor(range(num_pixel_seq), device=device, dtype=torch.long)[:, None].expand(-1,
                                                                                                      num_dates).flatten()
    new_dates = dates[mask > 0].reshape(num_pixel_seq, -1)
    if new_dates.size(1) == 0:
        return None, None, None, None

    if mode in ['mov_avg', 'gaussian', 'none']:
        input_seq = torch.zeros((num_pixel_seq, 406, 11), device=device)
        input_seq[seq_indices, dates] += mask[..., None].expand(-1, 11) * seq
    else:
        new_dates = dates[mask > 0].reshape(num_pixel_seq, -1)
        if new_dates.size(1) == 0:
            input_seq = torch.zeros((num_pixel_seq, model.module.num_dates, model.module.input_dim), device=device)
            input_seq[seq_indices, dates] += mask[..., None].expand(-1, model.module.input_dim) * seq
        else:
            new_seq = seq[mask > 0].reshape(num_pixel_seq, -1, model.module.input_dim)
            if new_dates[0][0] != 0:
                new_dates = torch.cat([torch.zeros_like(new_dates)[:, 0][:, None], new_dates], dim=1)
                new_seq = torch.cat([FIRST_VAL[None].expand(num_pixel_seq, -1, -1), new_seq], dim=1)
            if new_dates[0][-1] != 405:
                new_dates = torch.cat([new_dates, 405 * torch.ones_like(new_dates)[:, 0][:, None]], dim=1)
                new_seq = torch.cat([new_seq, LAST_VAL[None].expand(num_pixel_seq, -1, -1)], dim=1)
            rep = new_dates[:, 1:] - new_dates[:, :-1]
            rep_prev = torch.cat([rep, torch.zeros_like(rep)[:, 0][:, None]], dim=1)[0]
            rep_next = torch.cat([torch.zeros_like(rep)[:, 0][:, None], rep], dim=1)[0]
            rep_prev[-2] += 1
            rep_next[-1] += 1
            ref_dates = torch.tensor(range(model.module.num_dates), device=device)[None].expand(num_pixel_seq, -1).float()[
                ..., None]
            prev_dates = torch.repeat_interleave(new_dates, repeats=rep_prev, dim=1).float()[..., None]
            prev_vals = torch.repeat_interleave(new_seq, repeats=rep_prev, dim=1)
            next_dates = torch.repeat_interleave(new_dates, repeats=rep_next, dim=1).float()[..., None]
            next_vals = torch.repeat_interleave(new_seq, repeats=rep_next, dim=1)
            if mode == 'previous':
                input_seq = prev_vals
            elif mode == 'next':
                input_seq = next_vals
            elif mode == 'nearest':
                input_seq = torch.where(torch.abs(ref_dates-prev_dates) <= torch.abs(ref_dates-next_dates), prev_vals, next_vals)
            elif mode == 'linear':
                input_seq = prev_vals + ((ref_dates - prev_dates) / (next_dates - prev_dates)) * (next_vals - prev_vals)
    input_mask = torch.zeros((num_pixel_seq, 406), device=device)
    input_mask[seq_indices, dates] += mask
    input_seq_scatter = torch.zeros((num_pixel_seq, 406, 11), device=device)
    input_seq_scatter[seq_indices, dates] += mask[..., None].expand(-1, 11) * seq
    return input_seq, input_mask, label, input_seq_scatter


def gaussian(x, mu, sig=1):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) / (sig * np.sqrt(2 * np.pi))


def single(x, mu):
    return int(x == mu)


def moving(x, mu, sig=7):
    if mu - sig < x <= mu + sig:
        return 1/(2*sig)
    else:
        return 0


def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    else:
        return [recursive_todevice(c, device) for c in x]


def prepare_output(config):
    os.makedirs(config.res_dir, exist_ok=True)
    for fold in range(1, 6):
        os.makedirs(os.path.join(config.res_dir, "Fold_{}".format(fold)), exist_ok=True)


def checkpoint(log, config, fold=0):
    with open(
        os.path.join(config.res_dir, "Fold_{}".format(fold + 1), "trainlog.json"), "w"
    ) as outfile:
        json.dump(log, outfile, indent=4)


def get_metrics(config, curr_metrics, conf_matrix):
    assert config.selected_class in [-3, -2, -1], \
        'Not a valid value for selected_class. Cannot compute metrics for single class.'
    reduced_size = 19
    if config.selected_class == -1:
        conf_matrix = conf_matrix[:19]
        reduced_size = 19
    elif config.selected_class == -2:
        conf_matrix = conf_matrix[1:19]
        reduced_size = 18
    elif config.selected_class == -3:
        conf_matrix = conf_matrix[2:19]
        reduced_size = 17

    conf_matrix = conf_matrix[:, :reduced_size]
    conf_matrix_to_save = conf_matrix.int().cpu().numpy().tolist()
    conf_matrix = conf_matrix.cpu().numpy()
    true_positive = np.diag(conf_matrix)
    false_positive = np.sum(conf_matrix, 0) - true_positive
    false_negative = np.sum(conf_matrix, 1) - true_positive

    # Just in case we get a division by 0, ignore/hide the error
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = true_positive / (true_positive + false_positive + false_negative)
    m_iou = float(np.nanmean(iou) * 100)
    m_acc = {class_id: conf_matrix[i, i] / np.sum(conf_matrix[i]) * 100
             for i, class_id in enumerate(range(-config.selected_class-1, 19))}
    acc = float(np.diag(conf_matrix).sum() / conf_matrix.sum() * 100)
    metrics = {'loss': curr_metrics["loss"], 'epoch_time': curr_metrics["epoch_time"], 'iou': float(m_iou),
               'acc': float(acc), 'm_acc': np.mean(list(m_acc.values())), 'class_loss': curr_metrics['class_loss'],
               'class_acc': m_acc, 'conf_matrix': conf_matrix_to_save}
    return metrics
