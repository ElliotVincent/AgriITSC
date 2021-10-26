"""
Script for semantic inference with pre-trained models
Author: Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""
import argparse
import json
import os
import pprint
import torchnet as tnt
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from src.utils import utils, model_utils
from src.dataset.pastis import PASTIS_Dataset

from trainer import save_results

parser = argparse.ArgumentParser()
# Model parameters
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
    "--num_workers", default=8, type=int, help="Number of data loading workers"
)
parser.add_argument(
    "--fold",
    default=None,
    type=int,
    help="Do only one of the five fold (between 1 and 5)",
)
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


def iterate(model, data_loader, criterion, device=None):
    loss_meter = tnt.meter.AverageValueMeter()

    t_start = time.time()
    for i, batch in enumerate(data_loader):
        if device is not None:
            batch = recursive_todevice(batch, device)
        (x, dates), y = batch
        out, intensity_map = model(x, batch_positions=dates)
        loss = criterion(out, x).flatten(2).mean(2).mean(1).mean()
        loss_meter.add(loss.item())

        if (i + 1) % 100 == 0:
            print("Step [{}/{}], Loss: {:.4f}".format(i + 1, len(data_loader), loss_meter.value()[0]))

    t_end = time.time()
    total_time = t_end - t_start
    print("Epoch time : {:.1f}s".format(total_time))
    metrics = {
        "{}_loss".format("test"): loss_meter.value()[0],
        "{}_epoch_time".format("test"): total_time,
    }
    return metrics


def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    else:
        return [recursive_todevice(c, device) for c in x]


def main(config):
    fold_sequence = [
        [[1, 2, 3], [4], [5]],
        [[2, 3, 4], [5], [1]],
        [[3, 4, 5], [1], [2]],
        [[4, 5, 1], [2], [3]],
        [[5, 1, 2], [3], [4]],
    ]

    np.random.seed(config.rdm_seed)
    torch.manual_seed(config.rdm_seed)
    device = torch.device(config.device)

    model = model_utils.get_model(config)
    model = model.to(device)

    config.N_params = utils.get_ntrainparams(model)
    print(model)
    print("TOTAL TRAINABLE PARAMETERS :", config.N_params)

    fold_sequence = (
        fold_sequence if config.fold is None else [fold_sequence[config.fold - 1]]
    )
    for fold, (train_folds, val_fold, test_fold) in enumerate(fold_sequence):
        if config.fold is not None:
            fold = config.fold - 1

        # Dataset definition
        dt_test = PASTIS_Dataset(
            folder=config.dataset_folder,
            norm=True,
            reference_date=config.ref_date,
            mono_date=config.mono_date,
            target="semantic",
            sats=["S2"],
            folds=test_fold,
        )
        collate_fn = lambda x: utils.pad_collate(x, pad_value=config.pad_value)
        test_loader = data.DataLoader(
            dt_test,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

        # Load weights
        sd = torch.load(
            os.path.join(config.weight_folder, "Fold_{}".format(fold+1), "model.pth.tar"),
            map_location=device,
        )
        model.load_state_dict(sd["state_dict"])

        # Loss
        weights = torch.ones(config.num_classes, device=device).float()
        weights[config.ignore_index] = 0
        criterion = nn.CrossEntropyLoss(weight=weights)

        # Inference
        print("Testing . . .")
        model.load_state_dict(
            torch.load(
                os.path.join(
                    config.res_dir, "Fold_{}".format(fold + 1), "model.pth.tar"
                )
            )["state_dict"]
        )
        model.eval()
        test_metrics, conf_mat = iterate(model, data_loader=test_loader, criterion=criterion, device=device,)
        print("Loss {:.4f}".format(test_metrics["test_loss"]))
        save_results(fold + 1, test_metrics, conf_mat.cpu().numpy(), config)


if __name__ == "__main__":
    test_config = parser.parse_args()

    with open(os.path.join(test_config.weight_folder, "conf.json")) as file:
        model_config = json.loads(file.read())

    config = {**vars(test_config), **model_config}
    config = argparse.Namespace(**config)
    config.fold = test_config.fold

    pprint.pprint(config)
    main(config)
