"""
Main script for semantic experiments
Author: Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""
import argparse
import os
import pprint
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchnet as tnt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from src.utils.metrics import IoU
from src.utils import utils, model_utils
from src.dataset.pastis_raw import PASTIS_Dataset

N_CLASSES = 20
IMG_SIZE = 128

parser = argparse.ArgumentParser()
# Model parameters
parser.add_argument(
    "--model",
    default="upssits",
    type=str,
    help="Type of architecture to use. Can be one of: (utae/unet3d/fpn/convlstm/convgru/uconvlstm/buconvlstm)",
)
## U-TAE Hyperparameters
parser.add_argument("--encoder_widths", default="[64,64,64,128]", type=str)
parser.add_argument("--str_conv_k", default=4, type=int)
parser.add_argument("--str_conv_s", default=2, type=int)
parser.add_argument("--str_conv_p", default=1, type=int)
parser.add_argument("--encoder_norm", default="group", type=str)
parser.add_argument("--n_head", default=16, type=int)
parser.add_argument("--d_model", default=256, type=int)
parser.add_argument("--d_k", default=4, type=int)

# Set-up parameters
parser.add_argument(
    "--weight_folder",
    type=str,
    default="",
    help="Path to the main folder containing the pre-trained weights",
)

parser.add_argument(
    "--dataset_folder",
    default="",
    type=str,
    help="Path to the folder where the results are saved.",
)
parser.add_argument(
    "--res_dir",
    default="./results-2021-11-02-17-55-34",
    help="Path to the folder where the results should be stored",
)
parser.add_argument(
    "--save_img_step",
    default=50,
    type=int,
    help="Interval in iterations between save image steps",
)
parser.add_argument(
    "--num_workers", default=8, type=int, help="Number of data loading workers"
)
parser.add_argument("--rdm_seed", default=1, type=int, help="Random seed")
parser.add_argument(
    "--device",
    default="cuda",
    type=str,
    help="Name of device to use for tensor computations (cuda/cpu)",
)
parser.add_argument(
    "--display_step",
    default=50,
    type=int,
    help="Interval in batches between display of training metrics",
)
parser.add_argument(
    "--cache",
    dest="cache",
    action="store_true",
    help="If specified, the whole dataset is kept in RAM",
)
# Training parameters
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs per fold")
parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
parser.add_argument("--mono_date", default=None, type=str)
parser.add_argument("--ref_date", default="2018-09-01", type=str)
parser.add_argument(
    "--fold",
    default=None,
    type=int,
    help="Do only one of the five fold (between 1 and 5)",
)
parser.add_argument("--num_classes", default=20, type=int)
parser.add_argument("--ignore_index", default=-1, type=int)
parser.add_argument("--pad_value", default=0, type=float)
parser.add_argument("--padding_mode", default="reflect", type=str)
parser.add_argument(
    "--val_every",
    default=1,
    type=int,
    help="Interval in epochs between two validation steps.",
)
parser.add_argument(
    "--val_after",
    default=0,
    type=int,
    help="Do validation only after that many epochs.",
)

list_args = ["encoder_widths", "decoder_widths", "out_conv"]
parser.set_defaults(cache=False)


def iterate(
    model, data_loader, criterion, config, mode="test", device=None, fold=0
):
    loss_meter = tnt.meter.AverageValueMeter()

    t_start = time.time()
    iou_meter = IoU(
        num_classes=N_CLASSES,
        ignore_index=config.ignore_index,
        cm_device=config.device,
    )
    for i, batch in enumerate(data_loader):
        if device is not None:
            batch = recursive_todevice(batch, device)
        (x, dates), y = batch
        with torch.no_grad():
            out, intensity_map, recons, indices = model(x, y, batch_positions=dates, return_recons=True)

        iou_meter.add(indices, y)

        loss = criterion(out, x).flatten(2).mean(2).mean(1).mean()
        loss_meter.add(loss.item())
        if i % config.save_img_step == 0:
            print("Saving Fold {} Batch {}...".format(fold, i))
            save_images(i, x, out, recons, intensity_map, indices, config, y, fold)
            print("Done.")
    t_end = time.time()

    total_time = t_end - t_start
    print("Epoch time : {:.1f}s".format(total_time))
    miou, acc = iou_meter.get_miou_acc()
    metrics = {
        "{}_loss".format(mode): loss_meter.value()[0],
        "{}_epoch_time".format(mode): total_time,
        "{}_miou".format(mode): miou,
        "{}_acc".format(mode): acc,
    }
    return metrics


def save_images(batch_id, inp, out, recons, intensity_map, indices, config, y, fold, nt_steps=7):
    curr_path = os.path.join(config.res_dir, "Fold_{}".format(fold + 1), "batch_{}".format(batch_id))
    os.makedirs(curr_path, exist_ok=True)
    inp = torch.tensor(norm_quantile(inp[0, :, :3, :, :].cpu().numpy()))[:nt_steps]
    out = torch.tensor(norm_quantile(out[0, :, :3, :, :].cpu().numpy()))[:nt_steps]
    propos_4 = torch.tensor(norm_quantile(recons[0, 4, :, :3, :, :].cpu().numpy()))[:nt_steps]
    propos_12 = torch.tensor(norm_quantile(recons[0, 12, :, :3, :, :].cpu().numpy()))[:nt_steps]
    intensity_map = intensity_map[0].unsqueeze(0).expand(-1, 3, -1, -1).detach().cpu()
    img = torch.cat([inp, out, propos_4, propos_12, intensity_map], dim=0).numpy()
    r_in, g_in, b_in = img[:, 2, :, :], img[:, 1, :, :], img[:, 0, :, :]
    img = np.stack((r_in, g_in, b_in), axis=1)
    for i, name in enumerate(['inp', 'out', 'propos4', 'propos12']):
        for j in range(nt_steps):
            img_path = os.path.join(curr_path, '{}_{}.png'.format(name, j))
            plt.imsave(img_path, np.transpose(img[i * nt_steps + j], (1, 2, 0)))
    colours = cm.get_cmap('nipy_spectral', N_CLASSES)
    cmap = colours(np.linspace(0, 1, N_CLASSES))  # Obtain RGB colour map
    cmap[:, -1] = 1
    indices = indices[0].detach().cpu().numpy()
    y = y[0].detach().cpu().numpy()
    indices = cmap[indices]
    indices = indices.reshape((IMG_SIZE, IMG_SIZE, -1))
    img_path = os.path.join(curr_path, 'seg_map.png')
    plt.imsave(img_path, indices)
    y = cmap[y]
    y = y.reshape((IMG_SIZE, IMG_SIZE, -1))
    img_path = os.path.join(curr_path, 'seg_map_gt.png')
    plt.imsave(img_path, y)
    img_path = os.path.join(curr_path, 'intensity_map.png')
    plt.imsave(img_path, np.transpose(img[-1], (1, 2, 0)))


def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    else:
        return [recursive_todevice(c, device) for c in x]


def norm_quantile(I, min=0.05, max=0.95):
    return np.clip((I - np.quantile(I, min))/(np.quantile(I, max) - np.quantile(I, min)), 0, 1)**0.5


def main(config):
    if torch.cuda.is_available():
        type_device = "cuda"
        nb_device = torch.cuda.device_count()
    else:
        type_device = "cpu"
        nb_device = None
    print("Using {} device, nb_device is {}".format(type_device, nb_device))

    fold_sequence = [
        [[1, 2, 3], [4], [5]],
        [[2, 3, 4], [5], [1]],
        [[3, 4, 5], [1], [2]],
        [[4, 5, 1], [2], [3]],
        [[5, 1, 2], [3], [4]],
    ]

    device = torch.device(config.device)

    fold_sequence = (
        fold_sequence if config.fold is None else [fold_sequence[config.fold - 1]]
    )
    for fold, (_, _, test_fold) in enumerate(fold_sequence):
        if config.fold is not None:
            fold = config.fold - 1

        # Dataset definition
        dt_args = dict(
            folder=config.dataset_folder,
            norm=True,
            reference_date=config.ref_date,
            mono_date=config.mono_date,
            target="instance_only",
            sats=["S2"],
        )

        dt_test = PASTIS_Dataset(**dt_args, folds=test_fold)

        collate_fn = lambda x: utils.pad_collate(x, pad_value=config.pad_value)
        test_loader = data.DataLoader(
            dt_test,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

        print("Test {}".format(len(dt_test)))

        # Model definition
        model = model_utils.get_model(config)
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
        config.N_params = utils.get_ntrainparams(model)
        print("TOTAL TRAINABLE PARAMETERS :", config.N_params)

        model = model.to(device)
        criterion = nn.MSELoss(reduction='none')
        print("Testing best epoch . . .")

        model.load_state_dict(torch.load(
                os.path.join(
                    config.res_dir, "Fold_{}".format(fold + 1), "model.pth.tar"
                )
            )["state_dict"])
        model.eval()

        test_metrics = iterate(
            model,
            data_loader=test_loader,
            criterion=criterion,
            config=config,
            mode="test",
            device=device,
            fold=fold,
        )
        print("Loss {:.4f}".format(test_metrics["test_loss"]))
        print("IoU {:.4f}".format(test_metrics["test_miou"]))
        print("Acc {:.4f}".format(test_metrics["test_acc"]))


if __name__ == "__main__":
    config = parser.parse_args()
    for k, v in vars(config).items():
        if k in list_args and v is not None:
            v = v.replace("[", "")
            v = v.replace("]", "")
            config.__setattr__(k, list(map(int, v.split(","))))

    main(config)
    # import json
    #
    # with open('/home/vincente/upssits/results_1/Fold_1/trainlog.json') as json_file:
    #     data = json.load(json_file)
    #     x,y1,y2 = [],[],[]
    #     for k, v in data.items():
    #         x.append(k)
    #         y1.append(v['train_loss'])
    #         y2.append(v['val_loss'])
    #     plt.plot(x, y1, label='train')
    #     plt.plot(x, y2, label='val')
    #     plt.legend()
    #     plt.xlabel('epoch')
    #     plt.ylabel('loss')
    #     plt.xticks(np.arange(-1, 100, step=10))
    #     plt.show()
