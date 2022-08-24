import argparse
import json
import os
import pprint
import time
import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchnet as tnt
from collections import defaultdict
from src.utils import utils, model_utils
from src.dataset.pastis_raw import PASTIS_Dataset
from src.model.weight_init import weight_init
from src.utils.metrics import IoU


FIRST_DATE = 16
LAST_DATE = 421

parser = argparse.ArgumentParser()

# Model parameters
parser.add_argument("--model", default="upssitsv2", type=str, help="Type of architecture to use.")

# UPSSITS Hyperparameters
parser.add_argument("--num_classes", default=17, type=int)
parser.add_argument("--input_size", default="[128,128]", type=str)
parser.add_argument("--selected_class", default=-3, type=int)
parser.add_argument("--median_filtering", default=False, type=bool, help="Whether to learn the intensity map or not")
parser.add_argument("--app_map", default=False, type=bool, help="Whether to learn the intensity map or not")
parser.add_argument("--temp_int", default=False, type=bool, help="Whether to learn the intensity map or not")
parser.add_argument("--time_trans", default=False, type=bool, help="Whether to learn the intensity map or not")
parser.add_argument("--timestep", default=1, type=int, help="Whether to learn the intensity map or not")
parser.add_argument("--sigma", default=0.5, type=int, help="Whether to learn the intensity map or not")

# Set-up parameters
parser.add_argument("--dataset_folder", default="", type=str, help="Path to the folder where the results are saved.")
parser.add_argument("--target", default="semantic_only", type=str, help="Ground truth type. Either instance_only or semantic_only.")
parser.add_argument("--res_dir", default="./results", help="Path to the folder where the results should be stored")
parser.add_argument("--num_workers", default=8, type=int, help="Number of data loading workers")
parser.add_argument("--rdm_seed", default=5, type=int, help="Random seed")
parser.add_argument("--device", default="cuda", type=str, help="Name of device to use for tensor computation (cuda/cpu)")
parser.add_argument("--cache", dest="cache", action="store_true", help="If specified, the whole dataset is kept in RAM")

# Training parameters
parser.add_argument("--epochs", default=30, type=int, help="Number of epochs per fold")
parser.add_argument("--validation", default=False, type=bool, help="If to validate every 25 iterations")
parser.add_argument("--supervised", default=True, type=bool, help="Whether learn with supervision or not")
parser.add_argument("--supervised_double", default=True, type=bool, help="Whether learn with supervision or not")
parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
parser.add_argument("--lr", default=0.0001, type=float, help="Learning rate")
parser.add_argument("--alpha", default=0, type=float, help="Learning rate")
parser.add_argument("--beta", default=1, type=float, help="Learning rate")
parser.add_argument("--gamma", default=0, type=float, help="Learning rate")
parser.add_argument("--check_cluster_step", default=100, type=int, help="Interval in batches between reassignment of empty clusters")
parser.add_argument("--mono_date", default=None, type=str)
parser.add_argument("--ref_date", default="2018-09-01", type=str)
parser.add_argument("--ignore_index", default=-1, type=int)
parser.add_argument("--pad_value", default=0, type=float)
parser.add_argument("--padding_mode", default="reflect", type=str)

list_args = ["input_size", "encoder_widths", "decoder_widths", "out_conv"]
parser.set_defaults(cache=False)
#
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

def iterate(model, data_loader, criterion, epoch, config, val_metrics=None, val_loader=None, optimizer=None, mode="train", device=None):
    loss_meter = tnt.meter.AverageValueMeter()

    if mode != "train":
        iou_meter = IoU(
            num_classes=max(20, config.num_classes),
            ignore_index=None,
            cm_device=config.device,
        )

    ce_loss_meter = tnt.meter.AverageValueMeter()
    log_reg_loss_meter = tnt.meter.AverageValueMeter()

    # Store the class proportions in a dictionary
    proportions_meters = {f'prop_clus{class_id}': tnt.meter.AverageValueMeter()
                              for class_id in range(config.num_classes)}
    t_start = time.time()
    per_class_loss = torch.zeros(17, device=device)
    N = [0 for _ in range(17)]
    for i, batch in enumerate(data_loader):


        if device is not None:
            batch = recursive_todevice(batch, device)
        (x, dates), y = batch
        y = y.long()
        if config.target == "instance_only":
            y, y_instance = y[:, 0, :, :], y[:, 1, :, :]

        with torch.no_grad():
            dates_mask = (dates > 0).view(x.shape[0], x.shape[1], 1, 1, 1).float()
            weights = x[:, :, 0, :, :].unsqueeze(2)
            mean, std = torch.mean(weights, dim=1, keepdim=True), torch.std(weights, dim=1, keepdim=True)
            mask_sup, mask_inf = weights < (mean + std), weights > (mean - std)
            mask = mask_sup == mask_inf
            mask_mean = torch.sum(mask, (2, 3, 4), keepdim=True) / (128 * 128 * 1.) > 0.999
            dates_mask = dates_mask * mask_mean  # B x T x 1 x 1 x 1

            if config.selected_class == -1:
                mask = torch.where((19 > y[:, None, None, ...]) * (y[:, None, None, ...] >= 0), mask, torch.zeros_like(mask))
                mask = dates_mask * mask  # B x T x 1 x H x W

            elif config.selected_class == -2:
                mask = torch.where((19 > y[:, None, None, ...]) * (y[:, None, None, ...] > 0), mask, torch.zeros_like(mask))
                mask = dates_mask * mask  # B x T x 1 x H x W

            elif config.selected_class == -3:
                mask = torch.where((19 > y[:, None, None, ...]) * (y[:, None, None, ...] > 1), mask, torch.zeros_like(mask))
                mask = dates_mask * mask  # B x T x 1 x H x W

            elif config.selected_class >= 0:
                mask = torch.where(y[:, None, None, ...] == config.selected_class, mask, torch.zeros_like(mask))
                mask = dates_mask * mask  # B x T x 1 x H x W

            else:
                assert config.selected_class == -1, \
                    'Not a valid value for selected_class. Should be an int from -3 to 18'
                mask = dates_mask * mask_mean  # B x T x 1 x 1 x 1

        if mode != "train":
            with torch.no_grad():
                input, out, intensity_map, pred_seg, mask, temp_loss_MSE, temp_loss_LOG = model(x, dates, mask, mode='test', labels=y)
        else:
            optimizer.zero_grad()
            if epoch < 5:
                model.module.centroids.requires_grad = False
            else:
                model.module.centroids.requires_grad = True
            input, out, intensity_map, pred_seg, mask, y, temp_loss_MSE, temp_loss_LOG = model(x, dates, mask, labels=y, supervised=config.supervised)
        if config.selected_class == -1:
            loss = criterion(mask * out, mask * input).flatten(2).mean(2).mean(1).mean()
        else:
            loss = (criterion(mask * out, mask * input).mean(2).sum(1) / (mask.mean(2).sum(1)+1.)).mean()
            for class_id in range(2, 19):
                class_mask = torch.where(y[:, None, None, ...] == class_id, mask, torch.zeros_like(mask))
                class_loss = ((criterion(class_mask * out, class_mask * input).mean(2).sum(1) / (class_mask.mean(2).sum(1)+1.)).sum((1, 2)) / ((y == class_id).sum((1,2)) + 1.)).mean()
                per_class_loss[class_id-2] += class_loss
                if class_loss > 0.:
                    N[class_id - 2] += 1
        loss_to_back = loss
        if not config.supervised:
            ce_loss = -torch.min(torch.log(torch.softmax(-10.*temp_loss_MSE, dim=1)), dim=1)[0].flatten().mean()
            loss_to_back = config.alpha_r * loss + config.alpha_ce * ce_loss
        if config.supervised:
            criterion2 = nn.CrossEntropyLoss()
            if mode == 'train':
                inp = (-1.*temp_loss_MSE).permute(0, 4, 1, 2, 3).reshape(input.size(0) * input.size(3) * input.size(4), 20 + config.selected_class)
                inp_reg = (-torch.nn.functional.softplus(model.module.lambd)*temp_loss_LOG).permute(0, 4, 1, 2, 3).reshape(input.size(0) * input.size(3) * input.size(4), 20 + config.selected_class)
                tar = y.reshape(-1) + config.selected_class + 1
            else:
                inp = (-1.*temp_loss_MSE).permute(0, 4, 1, 2, 3).reshape(input.size(0) * input.size(3) * input.size(4), 20 + config.selected_class)
                inp_reg = (-torch.nn.functional.softplus(model.module.lambd)*temp_loss_LOG).permute(0, 4, 1, 2, 3).reshape(input.size(0) * input.size(3) * input.size(4), 20 + config.selected_class)
                tar = y.flatten(1).reshape(-1)
                mask = ((tar > 1) * (tar < 19)) == 1
                inp = inp[mask, :]
                inp_reg = inp_reg[mask, :]
                tar = tar[mask] + config.selected_class + 1

            ce_loss = criterion2(inp, tar)
            log_reg_loss = criterion2(inp_reg, tar)
            loss_to_back = loss

        ce_loss_meter.add(ce_loss.item())
        log_reg_loss_meter.add(log_reg_loss.item())
        if mode == "train":
            loss_to_back.backward()
            optimizer.step()
        loss_meter.add(loss.item())
        if mode != "train":
            iou_meter.add(pred_seg, y)

        if mode == "train":
            if not config.supervised:
                with torch.no_grad():
                    elem, elem_counts = torch.unique(pred_seg, return_counts=True)
                    curr_proportions = {elem[j].cpu().int().item(): (elem_counts[j].item()/torch.sum(elem_counts).item())
                                        for j in range(elem.shape[0]) if elem[j].cpu().int().item() != -1}
                    proportions = {f'prop_clus{class_id}': curr_proportions[class_id] if class_id in curr_proportions.keys() else 0.
                                   for class_id in range(config.num_classes)}
                    for class_id in range(config.num_classes):
                        proportions_meters[f'prop_clus{class_id}'].add(proportions[f'prop_clus{class_id}'])

                    if (i+1) % config.check_cluster_step == 0 and mode == "train" and not config.supervised:
                        proportions = [proportions_meters[f'prop_clus{class_id}'].value()[0]
                                       for class_id in range(config.num_classes)]
                        reassigned, idx = model.module.reassign_empty_clusters(proportions)
                        if len(reassigned) > 0:
                            print("Epoch [{}/{}], Iter {}, Loss : {:.3f}: Reassigned clusters {} from cluster {}".format(epoch,
                                                                                                          config.epochs, i+1, loss,
                                                                                                          reassigned, idx))
                        else:
                            print("Epoch [{}/{}], Iter {}, Loss : {:.3f}: No Reassigned cluster".format(epoch,
                                                                                                        config.epochs, i+1,
                                                                                                        loss))
                        for class_id in range(config.num_classes):
                            proportions_meters[f'prop_clus{class_id}'].reset()
            if (i+1) % 100 == 0:
                if config.supervised_double:
                    num_gt_classes = 20 + model.module.selected_class
                    num_cluster_per_class = model.module.num_classes // num_gt_classes
                    proto_count = model.module.proto_count.detach().cpu().numpy()
                    proportions = [[proto_count[j] / np.sum([proto_count[l] for l in range(
                        (j // num_cluster_per_class) * num_cluster_per_class,
                        (j // num_cluster_per_class + 1) * num_cluster_per_class)]) for j in range(
                        (k // num_cluster_per_class) * num_cluster_per_class,
                        (k // num_cluster_per_class + 1) * num_cluster_per_class)] for k in range(num_gt_classes)]
                    reassigned, idx = model.module.reassign_empty_clusters(proportions)
                    nn.init.zeros_(model.module.proto_count)
                    print("Epoch [{}/{}], Iter {}, LogReg-Loss: {:.3f}, CE-Loss: {:.3f}, R-Loss: {:.3f}, Reassigned prototypes: {}".format(epoch, config.epochs, i+1,
                                                                                      log_reg_loss_meter.value()[0],
                                                                                      ce_loss_meter.value()[0],
                                                                                      loss_meter.value()[0], reassigned))
                else:
                    print("Epoch [{}/{}], Iter {}, LogReg-Loss: {:.3f}, CE-Loss: {:.3f}, R-Loss: {:.3f}, Lambda: {:.3f}".format(epoch, config.epochs, i+1,
                                                                                      log_reg_loss_meter.value()[0],
                                                                                      ce_loss_meter.value()[0],
                                                                                      loss_meter.value()[0],
                                                                                      torch.nn.functional.softplus(model.module.lambd).item()))
                if config.validation:
                    with torch.no_grad():
                        val_metrics, _, conf_matrix = get_metrics(config, device, val_loader, "test", model, criterion,
                                                                  val_metrics)
                        print(val_metrics['acc']['test'][-1])
                        val_metrics['conf_matrix'].append(conf_matrix)
                        val_metrics['loss']['train'].append(loss_meter.value()[0])
                        val_metrics['ce_loss']['train'].append(ce_loss_meter.value()[0])
                        val_metrics['log_reg_loss']['train'].append(log_reg_loss_meter.value()[0])
    for class_id in range(2, 19):
        per_class_loss[class_id-2] = per_class_loss[class_id-2] / N[class_id-2]
    print(per_class_loss)
    t_end = time.time()
    total_time = t_end - t_start
    print("Epoch time : {:.1f}s".format(total_time))
    metrics = {"loss": loss_meter.value()[0], "ce_loss": ce_loss_meter.value()[0], "log_reg_loss": log_reg_loss_meter.value()[0],
               "epoch_time": total_time, "lambda": torch.nn.functional.softplus(model.module.lambd).item()}

    if mode == "train":
        metrics.update({f'prop_clus{class_id}': proportions_meters[f'prop_clus{class_id}'].value()[0] for class_id in range(config.num_classes)})
        return metrics, val_metrics
    else:
        return metrics, iou_meter.conf_metric.conf

def gaussian(x, mu=0, sig=1):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) / (sig * np.sqrt(2 * np.pi))

def compute_mean(model, data_loader, criterion, epoch, config, val_metrics=None, val_loader=None, optimizer=None, mode="train", device=None):
    loss_meter = tnt.meter.AverageValueMeter()

    if mode != "train":
        iou_meter = IoU(
            num_classes=max(20, config.num_classes),
            ignore_index=None,
            cm_device=config.device,
        )

    ce_loss_meter = tnt.meter.AverageValueMeter()
    log_reg_loss_meter = tnt.meter.AverageValueMeter()

    # Store the class proportions in a dictionary
    proportions_meters = {f'prop_clus{class_id}': tnt.meter.AverageValueMeter()
                              for class_id in range(config.num_classes)}
    t_start = time.time()
    time_size = config.timestep * model.module.timelen
    means = torch.zeros((20, model.module.timelen, model.module.input_dim), device=device)
    counts = torch.zeros(20, model.module.timelen, device=device)
    gaussian_weights = torch.Tensor([[gaussian(k, i*config.timestep, config.sigma) for k in range(time_size)]
                                     for i in range(model.module.timelen)]).to(device)
    for i, batch in enumerate(data_loader):
        if device is not None:
            batch = recursive_todevice(batch, device)
        (x, dates), y = batch
        y = y.long()
        if config.target == "instance_only":
            y, y_instance = y[:, 0, :, :], y[:, 1, :, :]

        with torch.no_grad():
            dates_mask = (dates > 0).view(x.shape[0], x.shape[1], 1, 1, 1).float()
            weights = x[:, :, 0, :, :].unsqueeze(2)
            mean, std = torch.mean(weights, dim=1, keepdim=True), torch.std(weights, dim=1, keepdim=True)
            mask_sup, mask_inf = weights < (mean + std), weights > (mean - std)
            mask = mask_sup == mask_inf
            mask_mean = torch.sum(mask, (2, 3, 4), keepdim=True) / (128 * 128 * 1.) > 0.999
            dates_mask = dates_mask * mask_mean  # B x T x 1 x 1 x 1

            # if configs.selected_class == -1:
            #     mask = torch.where((19 > y[:, None, None, ...]) * (y[:, None, None, ...] >= 0), mask, torch.zeros_like(mask))
            #     mask = dates_mask * mask  # B x T x 1 x H x W
            #
            # elif configs.selected_class == -2:
            #     mask = torch.where((19 > y[:, None, None, ...]) * (y[:, None, None, ...] > 0), mask, torch.zeros_like(mask))
            #     mask = dates_mask * mask  # B x T x 1 x H x W
            #
            # elif configs.selected_class == -3:
            #     mask = torch.where((19 > y[:, None, None, ...]) * (y[:, None, None, ...] > 1), mask, torch.zeros_like(mask))
            #     mask = dates_mask * mask # B x T x 1 x H x W
            # elif configs.selected_class >= 0:
            #     mask = torch.where(y[:, None, None, ...] == configs.selected_class, mask, torch.zeros_like(mask))
            #     mask = dates_mask * mask # B x T x 1 x H x W
            mask = dates_mask.expand(-1, -1, -1, 128, 128)
            # else:
            #     assert configs.selected_class == -1, \
            #         'Not a valid value for selected_class. Should be an int from -3 to 18'
            #     mask = dates_mask * mask_mean  # B x T x 1 x 1 x 1

        # x = (((x[:, :, 6, ...] * 1784.860595703125 + 3205.90185546875) - (x[:, :, 2, ...] * 1429.2191162109375 + 1429.2191162109375)) / ((x[:, :, 6, ...] * 1784.860595703125 + 3205.90185546875) + (x[:, :, 2, ...] * 1429.2191162109375 + 1429.2191162109375)))[:, :, None, ...]
        if mode == "train":
            with torch.no_grad():
                batch_size = x.size(0)
                num_dates = x.size(1)
                img_size = model.module.input_size
                num_pixel_seq = batch_size*img_size[0]*img_size[1]
                dates = dates.permute(1, 0)[..., None].expand(-1, -1, img_size[0]*img_size[1]).flatten(1).permute(1, 0).long().flatten()
                y = y.flatten(0).long()
                mask = mask.permute(1, 0, 3, 4, 2).flatten(1).permute(1, 0).flatten()
                curr_dates = dates - config.timestep * (FIRST_DATE // config.timestep)
                x = x.permute(2, 1, 0, 3, 4).flatten(2).permute(2, 1, 0)
                curr_seq = torch.Tensor(range(num_pixel_seq)).long()[:, None].expand(-1, num_dates).flatten().to(device)
                curr_means = torch.zeros((num_pixel_seq, time_size, model.module.input_dim), device=device)
                curr_counts = torch.zeros((num_pixel_seq, time_size), device=device)
                x = x.permute(2, 0, 1).flatten(1).permute(1, 0)
                curr_means[curr_seq, curr_dates] += mask[..., None].expand(-1, model.module.input_dim) * x  # (BHW) x T x C
                curr_counts[curr_seq, curr_dates] += mask  # (BHW) x T
                curr_means = torch.einsum('ti,pic->ptc', gaussian_weights, curr_means)
                curr_counts = torch.einsum('ti,pi->pt', gaussian_weights, curr_counts)
                curr_means = curr_means / torch.where(curr_counts[..., None] == 0, torch.ones_like(curr_counts[..., None]), curr_counts[..., None])
                saved_mask = torch.where(curr_counts <= 1., curr_counts, torch.ones_like(curr_counts))
                means.index_put_((y,), saved_mask[...,None] * curr_means, accumulate=True)
                counts.index_put_((y,), saved_mask, accumulate=True)
        else:
            with torch.no_grad():
                batch_size = x.size(0)
                num_dates = x.size(1)
                img_size = model.module.input_size
                num_pixel_seq = batch_size*img_size[0]*img_size[1]
                centroids = model.module.centroids.unsqueeze(0).expand(num_pixel_seq, -1, -1, -1, -1)
                dates = dates.permute(1, 0)[..., None].expand(-1, -1, img_size[0]*img_size[1]).flatten(1).permute(1, 0).long().flatten()
                mask_tmp = mask.permute(1, 0, 3, 4, 2).flatten(1).permute(1, 0).flatten()
                curr_dates = dates - config.timestep * (FIRST_DATE // config.timestep)
                curr_counts = torch.zeros((num_pixel_seq, time_size), device=device)
                x_tmp = x.permute(2, 1, 0, 3, 4).flatten(2).permute(2, 1, 0)
                curr_seq = torch.Tensor(range(num_pixel_seq)).long()[:, None].expand(-1, num_dates).flatten().to(device)
                curr_means = torch.zeros((num_pixel_seq, time_size, model.module.input_dim), device=device)
                x_tmp = x_tmp.permute(2, 0, 1).flatten(1).permute(1, 0)
                curr_means[curr_seq, curr_dates] += mask_tmp[..., None].expand(-1, model.module.input_dim) * x_tmp  # (BHW) x T x C
                curr_counts[curr_seq, curr_dates] += mask_tmp  # (BHW) x T

                pred_seg = None
                for k in range(16):
                    curr_means_tmp = curr_means[1024*k:1024*(k+1)]
                    curr_counts_tmp = curr_counts[1024*k:1024*(k+1)]
                    centroids_tmp = centroids[1024*k:1024*(k+1)]
                    x_tmp = torch.einsum('ti,pic->ptc', gaussian_weights, curr_means_tmp)
                    mask_tmp = torch.einsum('ti,pi->pt', gaussian_weights, curr_counts_tmp)
                    x_tmp = x_tmp / torch.where(mask_tmp[..., None] == 0, torch.ones_like(mask_tmp[..., None]),
                                                mask_tmp[..., None])
                    x_tmp_2 = x_tmp[:, None, ...].expand(-1, 17, -1, -1)
                    mask_tmp_2 = mask_tmp[:, None, ..., None].expand(-1, model.module.num_classes, -1, model.module.input_dim)
                    centroids_tmp = centroids_tmp.squeeze(-1)
                    mask_tmp_2 = torch.where(mask_tmp_2 <= 1., mask_tmp_2, torch.ones_like(mask_tmp_2))
                    curr_masked_inp = mask_tmp_2 * x_tmp_2
                    curr_masked_out = mask_tmp_2 * centroids_tmp
                    temp_loss_MSE = (model.module.mseloss(curr_masked_inp, curr_masked_out).mean(3, keepdim=True).sum(2, keepdim=True)) / (mask_tmp_2.mean(3, keepdim=True).sum(2, keepdim=True) + 1.)
                    _, pred_seg_tmp = torch.min(temp_loss_MSE, 1, keepdim=True)  # (BHW) x 1 x 1 x
                    if pred_seg is None:
                        pred_seg = pred_seg_tmp
                    else:
                        pred_seg = torch.cat([pred_seg, pred_seg_tmp], dim=0)
                out = torch.gather(centroids.squeeze(-1), 1, pred_seg.expand(-1, -1, model.module.timelen, model.module.input_dim)).squeeze(1)
                out = out.reshape(batch_size, 128, 128, model.module.timelen, model.module.input_dim).permute(0, 3, 4, 1, 2)
                pred_seg = pred_seg.squeeze(1).squeeze(1).squeeze(1).reshape(batch_size, 128, 128)
                # input = x_tmp.reshape(batch_size, 128, 128, model.module.timelen, model.module.input_dim).permute(0, 3, 4, 1, 2)
                # mask = mask_tmp.reshape(batch_size, 128, 128, model.module.timelen, 1).permute(0, 3, 4, 1, 2)
            if config.selected_class == -1:
                loss = criterion(mask * out, mask * input).flatten(2).mean(2).mean(1).mean()
            else:
                # loss = (criterion(mask * out, mask * input).mean(2).sum(1) / (mask.mean(2).sum(1)+1.)).mean()
                loss = criterion(model.module.centroids, model.module.centroids).mean()
                loss_to_back = loss
            if config.supervised:
                ce_loss = 0
                log_reg_loss = 0
                loss_to_back = loss
            ce_loss_meter.add(ce_loss)
            log_reg_loss_meter.add(log_reg_loss)
            if mode == "train":
                loss_to_back.backward()
                optimizer.step()
            loss_meter.add(loss.item())
            if mode != "train":
                iou_meter.add(pred_seg, y)

    t_end = time.time()
    total_time = t_end - t_start
    print("Epoch time : {:.1f}s".format(total_time))
    if mode == "train":
        means = means / counts[..., None]
        state_dict = model.module.state_dict()
        state_dict['centroids'] = means[2:-1][..., None]
        model.module.load_state_dict(state_dict)
        return None
    else:
        metrics = {"loss": loss_meter.value()[0], "ce_loss": ce_loss_meter.value()[0], "log_reg_loss": log_reg_loss_meter.value()[0],
                   "epoch_time": total_time}

        if mode == "train":
            metrics.update({f'prop_clus{class_id}': proportions_meters[f'prop_clus{class_id}'].value()[0] for class_id in range(config.num_classes)})
            return metrics, val_metrics
        else:
            return metrics, iou_meter.conf_metric.conf


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


def norm_quantile(I, min_alpha=0.05, max_alpha=0.95, given_min=None, given_max=None):
    if (given_min is not None) and (given_max is not None):
        return np.clip((I - given_min)/(given_max - given_min), 0, 1)**0.5
    else:
        used_min = np.quantile(I, min_alpha)
        used_max = np.quantile(I, max_alpha)
        return np.clip((I - used_min)/(used_max - used_min), 0, 1)**0.5, used_min, used_max


def get_metrics(config, device, data_loader, mode, model, criterion, metrics, matching=None):
    curr_metrics, conf_matrix = compute_mean(
        model,
        data_loader=data_loader,
        criterion=criterion,
        epoch=1,
        config=config,
        mode="test",
        device=device,
    )

    assert config.selected_class in [-3, -2, -1], \
        'Not a valid value for selected_class. Cannot compute metrics for single class.'
    if config.selected_class == -1:
        conf_matrix = conf_matrix[:19]
        reduced_size = 19
    elif config.selected_class == -2:
        conf_matrix = conf_matrix[1:19]
        reduced_size = 18
    elif config.selected_class == -3:
        conf_matrix = conf_matrix[2:19]
        reduced_size = 17


    if config.supervised:
        conf_matrix = conf_matrix[:, :reduced_size]
        conf_matrix_to_save = conf_matrix.int().cpu().numpy().tolist()
        conf_matrix = conf_matrix.cpu().numpy()
        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, 0) - true_positive
        false_negative = np.sum(conf_matrix, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)
        miou = float(np.nanmean(iou) * 100)
        acc = float(np.diag(conf_matrix).sum() / conf_matrix.sum() * 100)

    else:
        conf_matrix_to_save = conf_matrix.int().cpu().numpy().tolist()
        if matching is None:
            matching = torch.argmax(conf_matrix, dim=0)
        reduced_conf_matrix = torch.zeros((reduced_size, reduced_size), device=conf_matrix.device)
        for i, col in enumerate(matching):
            reduced_conf_matrix[:, col] = reduced_conf_matrix[:, col] + conf_matrix[:, i]
        reduced_conf_matrix = reduced_conf_matrix.cpu().numpy()
        true_positive = np.diag(reduced_conf_matrix)
        false_positive = np.sum(reduced_conf_matrix, 0) - true_positive
        false_negative = np.sum(reduced_conf_matrix, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)
        miou = float(np.nanmean(iou) * 100)
        acc = float(np.diag(reduced_conf_matrix).sum() / reduced_conf_matrix.sum() * 100)
    metrics['loss'][mode].append(float(curr_metrics["loss".format(mode)]))
    metrics['ce_loss'][mode].append(float(curr_metrics["ce_loss".format(mode)]))
    metrics['log_reg_loss'][mode].append(float(curr_metrics["log_reg_loss".format(mode)]))
    metrics['epoch_time'][mode].append(float(curr_metrics["epoch_time".format(mode)]))
    metrics['iou'][mode].append(float(miou))
    metrics['acc'][mode].append(float(acc))
    return metrics, matching, conf_matrix_to_save


def init_metrics():
    return {'acc': {'train': [], 'test': []},
            'iou': {'train': [], 'test': []},
            'loss': {'train': [], 'test': []},
            'ce_loss': {'train': [], 'test': []},
            'log_reg_loss': {'train': [], 'test': []},
            'epoch_time': {'train': [], 'test': []},
            'conf_matrix': [],
            'matching': []}


def main(config):
    if config.supervised:
        main_supervised(config)
    else:
        main_supervised(config)


def main_supervised(config):
    # exp_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # exp_id = "{}-{}-{}-{}-{}-{}-{}".format(exp_time,
    #                                        configs.num_classes,
    #                                        configs.median_filtering,
    #                                        configs.app_map, configs.temp_int,
    #                                        configs.time_trans,
    #                                        configs.selected_class)
    # configs.res_dir = './results-{}'.format(exp_id)
    config.res_dir = './results-2022-05-17-16-22-21-136-False-False-False-False--3'

    if torch.cuda.is_available():
        type_device = "cuda"
        nb_device = torch.cuda.device_count()
    else:
        type_device = "cpu"
        nb_device = None

    print("Using {} device, nb_device is {}".format(type_device, nb_device))

    np.random.seed(config.rdm_seed)
    torch.manual_seed(config.rdm_seed)
    # prepare_output(configs)
    device = torch.device(config.device)

    # Dataset definition
    dt_args = dict(
        folder=config.dataset_folder,
        norm=True,
        reference_date=config.ref_date,
        mono_date=config.mono_date,
        target=config.target,
        sats=["S2"],
    )

    fold_sequence = [
        [[1, 2, 3], [4], [5]],
        # [[2, 3, 4], [5], [1]],
        # [[3, 4, 5], [1], [2]],
        # [[4, 5, 1], [2], [3]],
        # [[5, 1, 2], [3], [4]],
    ]

    test_metrics = init_metrics()
    test_metrics_path = os.path.join(config.res_dir, "all_metrics_last.json")

    test_metrics_acc = init_metrics()
    test_metrics_acc_path = os.path.join(config.res_dir, "all_metrics_acc.json")

    for fold, (train_fold, val_fold, test_fold) in enumerate(fold_sequence):
        val_metrics = init_metrics()
        val_metrics_path = os.path.join(config.res_dir, "Fold_{}".format(fold+1), "val_metrics.json")

        dt_train = PASTIS_Dataset(**dt_args, folds=train_fold, cache=config.cache)
        dt_val = PASTIS_Dataset(**dt_args, folds=val_fold, cache=config.cache)

        collate_fn = lambda x: utils.pad_collate(x, pad_value=config.pad_value)
        train_loader = data.DataLoader(
            dt_train,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        val_loader = data.DataLoader(
            dt_val,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        print("Train {}".format(len(dt_train)))

        # Model definition
        model = model_utils.get_model(config)
        model = torch.nn.DataParallel(model.to(device), device_ids=[0])
        config.N_params = utils.get_ntrainparams(model)

        with open(os.path.join(config.res_dir, "conf.json"), "w") as file:
            file.write(json.dumps(vars(config), indent=4))

        print(model)
        print("TOTAL TRAINABLE PARAMETERS :", config.N_params)
        print("Trainable layers:")
        model.apply(weight_init)

        # model_res = './StampSize-2-Sigma-1'
        # state_dict = torch.load(
        #         os.path.join(
        #             model_res, "Fold_{}".format(fold + 1), "model_last.pth.tar"
        #         )
        #     )["state_dict"]
        # for key in model.state_dict().keys():
        #     if state_dict.get(key) is None:
        #         state_dict[key] = model.state_dict()[key]
        # state_dict["module.centroids"] = torch.repeat_interleave(state_dict["module.centroids"], configs.num_classes // 17, dim=0)
        # state_dict["module.proto_count"] = torch.repeat_interleave(state_dict["module.proto_count"], configs.num_classes // 17, dim=0)

        for name, p in model.named_parameters():
            if p.requires_grad:
                print(name)
        # model.load_state_dict(state_dict)
        # print(model.module.centroids.mean(1))
        # print(model.module.centroids.mean(2))
        # print(model.module.centroids.mean((1, 2)))
        # for k in [0, 5, 11, 16]:
        #     plt.plot(range(31), model.module.centroids.detach().cpu().numpy()[k, :, [2, 6, 8], 0].transpose(1,0))
        #     plt.axis([-1, 32, -0.7, 0.7])
        #     plt.show()
        # exit()
        # Optimizer and Loss
        # optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)
        #
        criterion = nn.MSELoss(reduction='none')
        #
        # # Training loop
        # trainlog = {}
        # best_loss = np.inf
        # best_loss_ce = np.inf
        # best_acc = 0
        # for epoch in range(1, configs.epochs + 1):
        #     print("EPOCH {}/{}".format(epoch, configs.epochs))
        #
        #     model.train()
        #     train_metrics, _ = iterate(
        #         model,
        #         data_loader=train_loader,
        #         criterion=criterion,
        #         epoch=epoch,
        #         configs=configs,
        #         val_metrics=val_metrics,
        #         val_loader=val_loader,
        #         optimizer=optimizer,
        #         mode="train",
        #         device=device
        #     )
        #
        #     train_R_loss = train_metrics["loss"]
        #     train_CE_loss = train_metrics["ce_loss"]
        #     train_LOG_loss = train_metrics["log_reg_loss"]
        #     print("LogReg-Loss: {:.4f}, CE-Loss: {:.4f}, R-Loss: {:.4f}".format(train_LOG_loss, train_CE_loss, train_R_loss))
        #     trainlog[epoch] = {**train_metrics}
        #     checkpoint(trainlog, configs, fold=fold)
        #     if not configs.validation:
        #         if epoch % 5 == 0:
        #             model.eval()
        #             with torch.no_grad():
        #                 val_metrics, _, conf_matrix = get_metrics(configs, device, val_loader, "test", model, criterion,
        #                                                           val_metrics)
        #                 val_metrics['conf_matrix'].append(conf_matrix)
        #
        #                 val_loss_r = val_metrics["loss"]["test"][-1]
        #                 val_loss_ce = val_metrics["ce_loss"]["test"][-1]
        #                 val_loss_log_reg = val_metrics["log_reg_loss"]["test"][-1]
        #                 val_acc = val_metrics['acc']['test'][-1]
        #
        #             print("LogReg-Loss: {:.4f}, CE-Loss: {:.4f}, R-Loss: {:.4f}, Acc: {:.2f},".format(val_loss_log_reg, val_loss_ce, val_loss_r, val_acc))
        #             if val_loss_r < best_loss:
        #                 best_loss = val_loss_r
        #                 torch.save(
        #                     {
        #                         "epoch": epoch,
        #                         "state_dict": model.state_dict(),
        #                         "optimizer": optimizer.state_dict(),
        #                     },
        #                     os.path.join(
        #                         configs.res_dir, "Fold_{}".format(fold + 1), "model_loss.pth.tar"
        #                     ),
        #                 )
        #
        #             if val_loss_ce < best_loss_ce:
        #                 best_loss_ce = val_loss_ce
        #                 torch.save(
        #                     {
        #                         "epoch": epoch,
        #                         "state_dict": model.state_dict(),
        #                         "optimizer": optimizer.state_dict(),
        #                     },
        #                     os.path.join(
        #                         configs.res_dir, "Fold_{}".format(fold + 1), "model_loss_ce.pth.tar"
        #                     ),
        #                 )
        #
        #             if val_acc > best_acc:
        #                 best_acc = val_acc
        #                 torch.save(
        #                     {
        #                         "epoch": epoch,
        #                         "state_dict": model.state_dict(),
        #                         "optimizer": optimizer.state_dict(),
        #                     },
        #                     os.path.join(
        #                         configs.res_dir, "Fold_{}".format(fold + 1), "model_acc.pth.tar"
        #                     ),
        #                 )
        #             torch.save(
        #                 {
        #                     "epoch": epoch,
        #                     "state_dict": model.state_dict(),
        #                     "optimizer": optimizer.state_dict(),
        #                 },
        #                 os.path.join(
        #                     configs.res_dir, "Fold_{}".format(fold + 1), "model_last.pth.tar"
        #                 ),
        #             )
        #     else:
        #         torch.save(
        #             {
        #                 "epoch": epoch,
        #                 "state_dict": model.state_dict(),
        #                 "optimizer": optimizer.state_dict(),
        #             },
        #             os.path.join(
        #                 configs.res_dir, "Fold_{}".format(fold + 1), "model.pth.tar"
        #             ),
        #         )
        #
        #     if epoch == 5:
        #         for g in optimizer.param_groups:
        #             g['lr'] = 0.0001
        #     # if epoch == 15:
        #     #     for g in optimizer.param_groups:
        #     #         g['lr'] = 0.0001
        #
        #     with open(val_metrics_path, 'w') as file:
        #         json.dump(val_metrics, file, indent=4)

        # print("Testing best epoch . . .")
        # model.load_state_dict(
        #     torch.load(
        #         os.path.join(
        #             configs.res_dir, "Fold_{}".format(fold + 1), "model_last.pth.tar"
        #         )
        #     )["state_dict"]
        # )
        # model.eval()
        #
        # dt_test = PASTIS_Dataset(**dt_args, folds=test_fold)
        #
        # test_loader = data.DataLoader(
        #     dt_test,
        #     batch_size=1,
        #     shuffle=True,
        #     drop_last=True,
        #     collate_fn=collate_fn,
        # )
        #
        # with torch.no_grad():
        #     test_metrics, _, conf_matrix = get_metrics(configs, device, test_loader, "test", model, criterion, test_metrics)
        #     test_metrics['conf_matrix'].append(conf_matrix)
        #     test_metrics['matching'].append(list(range(17)))

        print("Testing best epoch . . .")
        model.load_state_dict(
            torch.load(
                os.path.join(
                    config.res_dir, "Fold_{}".format(fold + 1), "model_acc.pth.tar"
                )
            )["state_dict"]
        )
        model.eval()

        dt_test = PASTIS_Dataset(**dt_args, folds=test_fold)

        test_loader = data.DataLoader(
            dt_test,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

        with torch.no_grad():
            test_metrics_acc, _, conf_matrix = get_metrics(config, device, test_loader, "test", model, criterion, test_metrics_acc)
            test_metrics_acc['conf_matrix'].append(conf_matrix)
            test_metrics_acc['matching'].append(list(range(17)))
    #
    # with open(test_metrics_path, 'w') as file:
    #     json.dump(test_metrics, file, indent=4)
    #
    # with open(test_metrics_acc_path, 'w') as file:
    #     json.dump(test_metrics_acc, file, indent=4)

    return


def main_compute_mean(config):
    exp_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    exp_id = "StampSize-{}-Sigma-{}-TEST".format(config.timestep, config.sigma)
    config.res_dir = './{}'.format(exp_id)

    if torch.cuda.is_available():
        type_device = "cuda"
        nb_device = torch.cuda.device_count()
    else:
        type_device = "cpu"
        nb_device = None

    print("Using {} device, nb_device is {}".format(type_device, nb_device))

    np.random.seed(config.rdm_seed)
    torch.manual_seed(config.rdm_seed)
    prepare_output(config)
    device = torch.device(config.device)

    # Dataset definition
    dt_args = dict(
        folder=config.dataset_folder,
        norm=True,
        reference_date=config.ref_date,
        mono_date=config.mono_date,
        target=config.target,
        sats=["S2"],
    )

    fold_sequence = [
        [[1, 2, 3], [4], [5]],
        # [[2, 3, 4], [5], [1]],
        # [[3, 4, 5], [1], [2]],
        # [[4, 5, 1], [2], [3]],
        # [[5, 1, 2], [3], [4]],
    ]

    test_metrics = init_metrics()
    test_metrics_path = os.path.join(config.res_dir, "all_metrics_last.json")

    for fold, (train_fold, val_fold, test_fold) in enumerate(fold_sequence):
        dt_train = PASTIS_Dataset(**dt_args, folds=train_fold, cache=config.cache)

        collate_fn = lambda x: utils.pad_collate(x, pad_value=config.pad_value)
        train_loader = data.DataLoader(
            dt_train,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        print("Train {}".format(len(dt_train)))

        # Model definition
        model = model_utils.get_model(config)
        model = torch.nn.DataParallel(model.to(device), device_ids=[0])
        config.N_params = utils.get_ntrainparams(model)

        with open(os.path.join(config.res_dir, "conf.json"), "w") as file:
            file.write(json.dumps(vars(config), indent=4))

        print(model)
        print("TOTAL TRAINABLE PARAMETERS :", config.N_params)
        print("Trainable layers:")
        for name, p in model.named_parameters():
            if p.requires_grad:
                print(name)
        model.apply(weight_init)

        # Optimizer and Loss
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        criterion = nn.MSELoss(reduction='none')

        # Training loop
        for epoch in range(1, 2):
            print("EPOCH {}/{}".format(epoch, config.epochs))

            model.train()
            compute_mean(
                model,
                data_loader=train_loader,
                criterion=criterion,
                epoch=epoch,
                config=config,
                val_metrics=None,
                val_loader=None,
                optimizer=optimizer,
                mode="train",
                device=device
            )
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                os.path.join(
                    config.res_dir, "Fold_{}".format(fold + 1), "model_last.pth.tar"
                ),
            )

        print("Testing best epoch . . .")
        model.load_state_dict(
            torch.load(
                os.path.join(
                    config.res_dir, "Fold_{}".format(fold + 1), "model_last.pth.tar"
                )
            )["state_dict"]
        )
        model.eval()

        dt_test = PASTIS_Dataset(**dt_args, folds=test_fold)

        test_loader = data.DataLoader(
            dt_test,
            batch_size=1,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

        with torch.no_grad():
            test_metrics, _, conf_matrix = get_metrics(config, device, test_loader, "test", model, criterion, test_metrics)
            test_metrics['conf_matrix'].append(conf_matrix)
            test_metrics['matching'].append(list(range(17)))

    with open(test_metrics_path, 'w') as file:
        json.dump(test_metrics, file, indent=4)

    return test_metrics['acc']['test'][0]


def main_unsupervised(config):
    exp_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    exp_id = "{}-{}-{}-{}-{}-{}-{}".format(exp_time,
                                           config.num_classes,
                                           config.median_filtering,
                                           config.app_map, config.temp_int,
                                           config.time_trans,
                                           config.selected_class)
    config.res_dir = './results-{}'.format(exp_id)

    if torch.cuda.is_available():
        type_device = "cuda"
        nb_device = torch.cuda.device_count()
    else:
        type_device = "cpu"
        nb_device = None

    print("Using {} device, nb_device is {}".format(type_device, nb_device))

    np.random.seed(config.rdm_seed)
    torch.manual_seed(config.rdm_seed)
    prepare_output(config)
    device = torch.device(config.device)

    # Dataset definition
    dt_args = dict(
        folder=config.dataset_folder,
        norm=True,
        reference_date=config.ref_date,
        mono_date=config.mono_date,
        target=config.target,
        sats=["S2"],
    )

    dt_train = PASTIS_Dataset(**dt_args, folds=[1, 2, 3, 4, 5], cache=config.cache)

    collate_fn = lambda x: utils.pad_collate(x, pad_value=config.pad_value)
    train_loader = data.DataLoader(
        dt_train,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print("Train {}".format(len(dt_train)))

    # Model definition
    model = model_utils.get_model(config)
    model = torch.nn.DataParallel(model.to(device), device_ids=[0, 1, 2, 3])
    config.N_params = utils.get_ntrainparams(model)

    with open(os.path.join(config.res_dir, "conf.json"), "w") as file:
        file.write(json.dumps(vars(config), indent=4))

    print(model)
    print("TOTAL TRAINABLE PARAMETERS :", config.N_params)
    print("Trainable layers:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name)
    model.apply(weight_init)

    # Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    criterion = nn.MSELoss(reduction='none')

    # Training loop
    trainlog = {}
    best_loss = np.inf
    for epoch in range(1, config.epochs + 1):
        print("EPOCH {}/{}".format(epoch, config.epochs))

        model.train()
        train_metrics = iterate(
            model,
            data_loader=train_loader,
            criterion=criterion,
            epoch=epoch,
            config=config,
            optimizer=optimizer,
            mode="train",
            device=device
        )

        train_loss = train_metrics["loss"]
        print("Loss {:.4f}".format(train_loss))
        trainlog[epoch] = {**train_metrics}
        checkpoint(trainlog, config)
        if epoch >= 5 and train_loss < best_loss:
            best_loss = train_loss
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                os.path.join(
                    config.res_dir, "model.pth.tar"
                ),
            )

    print("Testing best epoch . . .")
    model.load_state_dict(
        torch.load(
            os.path.join(
                config.res_dir, "model.pth.tar"
            )
        )["state_dict"]
    )
    model.eval()

    fold_sequence = [[[1, 2, 3], [5]],
                     [[2, 3, 4], [1]],
                     [[3, 4, 5], [2]],
                     [[4, 5, 1], [3]],
                     [[5, 1, 2], [4]]]

    metrics = {'acc': {'train': [], 'test': []},
               'iou': {'train': [], 'test': []},
               'loss': {'train': [], 'test': []},
               'epoch_time': {'train': [], 'test': []},
               'conf_matrix': {},
               'matching': {}}

    json_dict_path = os.path.join(config.res_dir, "all_metrics.json")

    for fold, (train_fold, test_fold) in enumerate(fold_sequence):
        dt_args = dict(
            folder=config.dataset_folder,
            norm=True,
            reference_date=config.ref_date,
            mono_date=config.mono_date,
            target=config.target,
            sats=["S2"],
        )
        dt_train = PASTIS_Dataset(**dt_args, folds=train_fold)
        dt_test = PASTIS_Dataset(**dt_args, folds=test_fold)

        collate_fn = lambda x: utils.pad_collate(x, pad_value=config.pad_value)

        train_loader = data.DataLoader(
            dt_train,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

        test_loader = data.DataLoader(
            dt_test,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

        with torch.no_grad():
            metrics, matching, conf_matrix = get_metrics(config, device, train_loader, "train", model, criterion, metrics)
            metrics, matching, conf_matrix = get_metrics(config, device, test_loader, "test", model, criterion, metrics, matching)
            metrics['conf_matrix'][fold + 1] = conf_matrix
            metrics['matching'][fold + 1] = matching.cpu().numpy().tolist()

    with open(json_dict_path, 'w') as file:
        json.dump(metrics, file, indent=4)

    return


if __name__ == "__main__":
    config = parser.parse_args()
    for k, v in vars(config).items():
        if k in list_args and v is not None:
            v = v.replace("[", "")
            v = v.replace("]", "")
            config.__setattr__(k, list(map(int, v.split(","))))
    pprint.pprint(config)
    acc = main_compute_mean(config)
    acc = int(acc * 100) / 100.
    print(acc)
    # main(configs)
    # res = []
    # for timestep in range(1, 2):
    #     configs.timestep = timestep
    #     res_tmp = []
    #     for sigma in range(1, 31):
    #         configs.sigma = sigma
    #         acc = main_compute_mean(configs)
    #         acc = int(acc * 100) / 100.
    #         res_tmp.append(acc)
    #         print('{} time step, {} sigma: {:.2f}'.format(timestep, sigma, acc))
    #     res.append(res_tmp)
    # print(res)
