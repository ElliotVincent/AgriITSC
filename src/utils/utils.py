import json
import numpy as np
import os
from pathlib import Path
import random
import torch
from src.datasets.ts2c import TS2CDataset
from src.models.dti_ts import DTI_TS, gaussian
from src.models.ncc import Ncc
from src.models.mlpltae import MLPLTAE
from src.models.oscnn import OS_CNN_easy_use
from src.models.tapnet import TapNet
from src.utils.paths import RESULTS_PATH


def coerce_to_path_and_check_exist(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError('{} does not exist'.format(path.absolute()))
    return path


def coerce_to_path_and_create_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_dataset(config, split='all'):
    name = config.get("name", "ts2c")
    dataset = {'ts2c': TS2CDataset}
    if name == 'pastis':
        return dataset[name](split=split, fold=config.get("fold", 1))
    else:
        return dataset[name](split)


def get_model(config, exp_tag=None):
    name = config.pop("name", "agritsc")
    if name == 'dtits':
        model = DTI_TS(**config)
    elif name == 'ncc':
        config.pop('supervised')
        model = Ncc(**config)
        config['supervised'] = True
    elif name == 'ltae':
        model = MLPLTAE(**config)
    elif name == 'oscnn':
        model = OS_CNN_easy_use(**config, exp_tag=exp_tag)
    elif name == 'tapnet':
        model = TapNet(**config)
    else:
        raise NameError(name)
    config["name"] = name
    return model


def get_ntrainparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Logger:
    def __init__(
            self,
            path,
    ):
        super(Logger, self).__init__()
        self.path = path
        self.metrics = {'loss': {},
                        'acc': {},
                        'mean_acc': {},
                        'tvh': {},
                        'acc_per_class': {},
                        'lr': {},
                        'proportions': {},
                        'lce': {},
                        }

    def update(self, metrics, n_iter):
        for metric in metrics.keys():
            if metric != 'conf_matrix':
                self.metrics[metric][n_iter] = metrics[metric]
        with open(self.path, "w") as file:
            file.write(json.dumps(self.metrics, indent=4))

    def load(self, logs):
        self.metrics = logs


class TransfoScheduler:
    def __init__(self, config, patience=5, threshold=0.001):
        self.config = config
        self.val_loss = None
        self.patience_count = 0
        self.patience = patience
        self.threshold = threshold
        self.curriculum = config['training']['curriculum']
        self.num_transfo = len(self.curriculum)
        self.curr_transfo = 0
        self.is_complete = False

    def update(self, val_loss, n_iter):
        if not self.is_complete:
            if self.val_loss is None:
                self.val_loss = val_loss
                print('Saved best is {:.2f}'.format(val_loss))
            elif val_loss <= (1. - self.threshold * np.sign(self.val_loss)) * self.val_loss:
                self.val_loss = val_loss
                self.patience_count = 0
                print('New saved best is {:.2f}'.format(val_loss))
            else:
                self.patience_count += 1
                print('{:.2f} is less good than {:.2f}'.format(val_loss, self.val_loss))
                if self.patience_count > self.patience:
                    print('Adding transfo')
                    self.curr_transfo += 1
                    self.patience_count = 0
                    self.curriculum[self.curr_transfo] = n_iter
                    self.config['training']['curriculum'] = self.curriculum
                    self.val_loss = None
                    if self.curr_transfo + 1 == self.num_transfo:
                        self.is_complete = True


def initialize_prototypes(config, loader, device):
    init_proto = config['model']['init_proto']
    if init_proto == 'sample':
        for i, batch in enumerate(loader):
            input_seq, mask, y = batch
            input_seq = input_seq.to(device)
            mask = mask.to(device)
            input_seq = input_seq.view(-1, config['model']['num_steps'], config['model']['input_dim']).to(torch.float32)
            mask = mask.view(-1, config['model']['num_steps']).int()
            indice = random.sample(range(input_seq.size(0)), config['model']['num_prototypes'])
            indice = torch.tensor(indice)
            sample = input_seq[indice]
            sample_mask = mask[indice]
            return sample, sample_mask
    elif init_proto == 'random':
        return None
    elif init_proto == 'means_previous':
        means_path = Path(os.path.join(RESULTS_PATH, f'{config["dataset"]["name"]}', 'init/means_previous.pt'))
        if not means_path.exists():
            means = torch.zeros((config['model']['num_classes'], config['model']['num_steps'],
                                 config['model']['input_dim']), device=device)
            mask_counts = torch.zeros((config['model']['num_classes'], config['model']['num_steps']), device=device)
            for i, batch in enumerate(loader):
                input_seq, mask, y = batch
                input_seq = input_seq.to(device).float().reshape(-1, input_seq.size(2), input_seq.size(3))
                new_input_seq = input_seq.clone()
                mask = mask.to(device).float().reshape(-1, mask.size(2))
                new_mask = mask.clone()
                y = y.to(device).long().flatten()
                for j in range(1, input_seq.size(1)):
                    mask_tmp = mask[:, j] == 0
                    new_input_seq[mask_tmp, j] = new_input_seq[mask_tmp, j-1]
                    new_mask[mask_tmp, j] = new_mask[mask_tmp, j-1]
                means.index_put_((y,), new_input_seq, accumulate=True)
                mask_counts.index_put_((y,), new_mask, accumulate=True)
            means = means / torch.where(mask_counts == 0, 1., mask_counts)[..., None]
            torch.save(means, means_path)
        else:
            means = torch.load(means_path)
        return means
    elif init_proto == 'means_avg':
        means_path = Path(os.path.join(RESULTS_PATH, f'{config["dataset"]["name"]}', 'init/means_avg.pt'))
        if not means_path.exists():
            means = torch.zeros((config['model']['num_classes'], config['model']['num_steps'],
                                 config['model']['input_dim']), device=device)
            mask_counts = torch.zeros((config['model']['num_classes'], config['model']['num_steps']), device=device)
            weights = torch.tensor([[int(abs(step-date) <= 7)/15 for date in range(config['model']['num_steps'])]
                                    for step in range(config['model']['num_steps'])], dtype=torch.float, device='cuda')
            for i, batch in enumerate(loader):
                input_seq, mask, y = batch
                input_seq = input_seq.to(device).float()
                mask = mask.to(device).float()
                y = y.to(device).long()
                input_seq = torch.einsum('ti,ybic->ybtc', weights, input_seq)
                mask = torch.einsum('ti,ybi->ybt', weights, mask.float())
                mask_nonzero = torch.where(mask[..., None] == 0,
                                           torch.ones_like(mask[..., None]),
                                           mask[..., None])
                input_seq = input_seq / mask_nonzero
                means.index_put_((y,), mask[..., None]*input_seq, accumulate=True)
                mask_counts.index_put_((y,), mask, accumulate=True)
            means = means / torch.where(mask_counts == 0, 1., mask_counts)[..., None]
            torch.save(means, means_path)
        else:
            means = torch.load(means_path)
        return means
    elif init_proto == 'means_gaussian':
        means_path = Path(os.path.join(RESULTS_PATH, f'{config["dataset"]["name"]}', 'init/means_gaussian.pt'))
        if config["dataset"]["name"] == 'pastis':
            means_path = Path(os.path.join(RESULTS_PATH, f'{config["dataset"]["name"]}', f'init/Fold_{config["dataset"]["fold"]}/means_gaussian.pt'))
        if not means_path.exists():
            means = torch.zeros((config['model']['num_classes'], config['model']['num_steps'],
                                 config['model']['input_dim']), device=device)
            mask_counts = torch.zeros((config['model']['num_classes'], config['model']['num_steps']), device=device)
            weights = torch.tensor([[gaussian(date, step, 7) for date in range(config['model']['num_steps'])]
                                    for step in range(config['model']['num_steps'])], dtype=torch.float, device='cuda')
            for i, batch in enumerate(loader):
                input_seq, mask, y = batch
                input_seq = input_seq.to(device).float()
                mask = mask.to(device).float()
                y = y.to(device).long()
                input_seq = torch.einsum('ti,ybic->ybtc', weights, input_seq)
                mask = torch.einsum('ti,ybi->ybt', weights, mask.float())
                means.index_put_((y,), input_seq, accumulate=True)
                mask_counts.index_put_((y,), mask, accumulate=True)
            means = means / torch.where(mask_counts == 0, 1., mask_counts)[..., None]
            torch.save(means, means_path)
        else:
            means = torch.load(means_path)
        return means
    elif init_proto == 'means':
        means_path = Path(os.path.join(RESULTS_PATH, f'{config["dataset"]["name"]}', 'init/means.pt'))
        masks_path = Path(os.path.join(RESULTS_PATH, f'{config["dataset"]["name"]}', 'init/masks.pt'))
        if not means_path.exists():
            means = torch.zeros((config['model']['num_classes'], config['model']['num_steps'],
                                 config['model']['input_dim']), device=device)
            mask_counts = torch.zeros((config['model']['num_classes'], config['model']['num_steps']), device=device)
            for i, batch in enumerate(loader):
                input_seq, mask, y = batch
                input_seq = input_seq.to(device).float()
                mask = mask.to(device).float()
                y = y.to(device).long()
                means.index_put_((y,), input_seq, accumulate=True)
                mask_counts.index_put_((y,), mask, accumulate=True)
            means = means / torch.where(mask_counts == 0, 1., mask_counts)[..., None]
            masks = mask_counts > 0
            torch.save(means, means_path)
            torch.save(masks, masks_path)
            return means
        else:
            means = torch.load(means_path)
            return means
    elif init_proto == 'kmeans':
        init_seed = config['model'].pop('init_seed', 1)
        if config['dataset']['name'] == 'pastis':
            path = Path(os.path.join(RESULTS_PATH, f'{config["dataset"]["name"]}', f'init/Fold_{config["dataset"]["fold"]}/kmeans32.pt'))
        else:
            path = Path(os.path.join(RESULTS_PATH, f'{config["dataset"]["name"]}', f'init/kmeans32_{init_seed}.pt'))
        if not path.exists():
            print("No kmeans centroids saved: initialize with random sample instead")
            config['model']['init_proto'] = 'sample'
            return initialize_prototypes(config, loader, device)
        else:
            kmeans_centroids = torch.load(path)
            return kmeans_centroids
    else:
        raise NameError(init_proto)
