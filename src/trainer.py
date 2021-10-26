"""
Main script for semantic experiments
Author: Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""
import argparse
import json
import os
import pickle as pkl
import pprint
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchnet as tnt
import visdom

from src.utils import utils, model_utils
from src.dataset.pastis import PASTIS_Dataset
from src.model.weight_init import weight_init

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
    "--dataset_folder",
    default="",
    type=str,
    help="Path to the folder where the results are saved.",
)
parser.add_argument(
    "--res_dir",
    default="./results",
    help="Path to the folder where the results should be stored",
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
    model, viz, data_loader, criterion, epoch, optimizer=None, mode="train", device=None
):
    loss_meter = tnt.meter.AverageValueMeter()

    t_start = time.time()
    for i, batch in enumerate(data_loader):
        if device is not None:
            batch = recursive_todevice(batch, device)
        (x, dates), y = batch
        y = y.long()
        if mode != "train":
            with torch.no_grad():
                out, intensity_map = model(x, batch_positions=dates)
        else:
            optimizer.zero_grad()
            out, intensity_map = model(x, batch_positions=dates)
        loss = criterion(out, x).flatten(2).mean(2).mean(1).mean()

        if mode == "train":
            loss.backward()
            optimizer.step()
        loss_meter.add(loss.item())

        if (i + 1) % 100 == 0:
            print("Step [{}/{}], Loss: {:.4f}".format(i + 1, len(data_loader), loss_meter.value()[0]))
            if mode == "train":
                indices = np.random.randint(0, x.shape[1], x.shape[0])
                select_in = np.squeeze(np.take_along_axis(norm_quantile(x.cpu().numpy()[:, :, :3, :, :]),
                                               np.expand_dims(indices, (1, 2, 3, 4)), axis=1))
                select_out = np.squeeze(np.take_along_axis(norm_quantile(out.detach().cpu().numpy()[:, :, :3, :, :]),
                                                np.expand_dims(indices, (1, 2, 3, 4)), axis=1))
                intensity_map = intensity_map.expand(-1, 3, -1, -1).detach().cpu()
                img = torch.cat([torch.tensor(select_in), torch.tensor(select_out), intensity_map], dim=0).numpy()
                r_in, g_in, b_in = img[:, 2, :, :], img[:, 1, :, :], img[:, 0, :, :]
                img = np.stack((r_in, g_in, b_in), axis=1)
                viz.images(img, win='Images', nrow=4, opts=dict(caption='Training samples', store_history=True))
                viz.line([[loss_meter.value()[0]]], [(epoch-1) * 363 + i], win='Train', update='append')
            elif mode == "val":
                viz.line([[loss_meter.value()[0]]], [epoch-1], win='Val', update='append')

    t_end = time.time()
    total_time = t_end - t_start
    print("Epoch time : {:.1f}s".format(total_time))
    metrics = {
        "{}_loss".format(mode): loss_meter.value()[0],
        "{}_epoch_time".format(mode): total_time,
    }
    return metrics, viz


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


def checkpoint(fold, log, config):
    with open(
        os.path.join(config.res_dir, "Fold_{}".format(fold), "trainlog.json"), "w"
    ) as outfile:
        json.dump(log, outfile, indent=4)


def save_results(fold, metrics, conf_mat, config):
    with open(
        os.path.join(config.res_dir, "Fold_{}".format(fold), "test_metrics.json"), "w"
    ) as outfile:
        json.dump(metrics, outfile, indent=4)
    pkl.dump(
        conf_mat,
        open(
            os.path.join(config.res_dir, "Fold_{}".format(fold), "conf_mat.pkl"), "wb"
        ),
    )

def norm_quantile(I, min=0.05, max=0.95):
    return np.clip((I - np.quantile(I, min))/(np.quantile(I, max) - np.quantile(I, min)), 0, 1)**1.3

def main(config):
    viz = visdom.Visdom(port=8889, env='ups-sits')
    viz.line([[0.]], [0], win='Train', opts=dict(title='Train loss', legend=['train loss']))
    viz.line([[0.]], [0], win='Val', opts=dict(title='Val loss', legend=['val loss']))
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

    np.random.seed(config.rdm_seed)
    torch.manual_seed(config.rdm_seed)
    prepare_output(config)
    device = torch.device(config.device)

    fold_sequence = (
        fold_sequence if config.fold is None else [fold_sequence[config.fold - 1]]
    )
    for fold, (train_folds, val_fold, test_fold) in enumerate(fold_sequence):
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

        dt_train = PASTIS_Dataset(**dt_args, folds=train_folds, cache=config.cache)
        dt_val = PASTIS_Dataset(**dt_args, folds=val_fold, cache=config.cache)
        dt_test = PASTIS_Dataset(**dt_args, folds=test_fold)

        collate_fn = lambda x: utils.pad_collate(x, pad_value=config.pad_value)
        train_loader = data.DataLoader(
            dt_train,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
        val_loader = data.DataLoader(
            dt_val,
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

        print(
            "Train {}, Val {}, Test {}".format(len(dt_train), len(dt_val), len(dt_test))
        )

        # Model definition
        model = model_utils.get_model(config)
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])

        config.N_params = utils.get_ntrainparams(model)
        with open(os.path.join(config.res_dir, "conf.json"), "w") as file:
            file.write(json.dumps(vars(config), indent=4))
        print(model)
        print("TOTAL TRAINABLE PARAMETERS :", config.N_params)
        print("Trainable layers:")
        for name, p in model.named_parameters():
            if p.requires_grad:
                print(name)
        model = model.to(device)
        model.apply(weight_init)

        # Optimizer and Loss
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        weights = torch.ones(config.num_classes, device=device).float()
        weights[config.ignore_index] = 0
        criterion = nn.MSELoss(reduction='none')

        # Training loop
        trainlog = {}
        best_loss = np.inf
        for epoch in range(1, config.epochs + 1):
            print("EPOCH {}/{}".format(epoch, config.epochs))

            model.train()
            train_metrics, viz = iterate(
                model,
                viz,
                data_loader=train_loader,
                criterion=criterion,
                epoch=epoch,
                optimizer=optimizer,
                mode="train",
                device=device,
            )
            if epoch % config.val_every == 0 and epoch > config.val_after:
                print("Validation . . . ")
                model.eval()
                val_metrics, viz = iterate(
                    model,
                    viz,
                    data_loader=val_loader,
                    criterion=criterion,
                    epoch=epoch,
                    optimizer=optimizer,
                    mode="val",
                    device=device,
                )

                print("Loss {:.4f}".format(val_metrics["val_loss"]))

                trainlog[epoch] = {**train_metrics, **val_metrics}
                checkpoint(fold + 1, trainlog, config)
                if val_metrics["val_loss"] <= best_loss:
                    best_loss = val_metrics["val_loss"]
                    torch.save(
                        {
                            "epoch": epoch,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        },
                        os.path.join(
                            config.res_dir, "Fold_{}".format(fold + 1), "model.pth.tar"
                        ),
                    )
            else:
                trainlog[epoch] = {**train_metrics}
                checkpoint(fold + 1, trainlog, config)

        print("Testing best epoch . . .")
        model.load_state_dict(
            torch.load(
                os.path.join(
                    config.res_dir, "Fold_{}".format(fold + 1), "model.pth.tar"
                )
            )["state_dict"]
        )
        model.eval()

        test_metrics, conf_mat = iterate(
            model,
            data_loader=test_loader,
            criterion=criterion,
            epoch=0,
            optimizer=optimizer,
            viz=None,
            mode="test",
            device=device,
        )
        print("Loss {:.4f}".format(test_metrics["test_loss"]))
        save_results(fold + 1, test_metrics, conf_mat.cpu().numpy(), config)


if __name__ == "__main__":
    config = parser.parse_args()
    for k, v in vars(config).items():
        if k in list_args and v is not None:
            v = v.replace("[", "")
            v = v.replace("]", "")
            config.__setattr__(k, list(map(int, v.split(","))))

    pprint.pprint(config)
    main(config)
