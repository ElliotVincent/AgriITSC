import json
import numpy as np
from pathlib import Path
import random
import torch

from src.datasets.ts2c import TS2CDataset
from src.models.agritsc import AgriSits


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
    dataset = {'denethor': None,
               'sa': None,
               'ts2c': TS2CDataset,
               'pastis': None,
               }
    return dataset[name](split)


def get_model(config):
    name = config.pop("name", "agrisits")
    if name == 'agrisits':
        model = AgriSits(**config)
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
        self.curr_transfo = 1
        self.is_complete = False

    def update(self, val_loss, n_iter):
        if not self.is_complete:
            if self.val_loss is None:
                self.val_loss = val_loss
            elif val_loss <= (1. - self.threshold * np.sign(self.val_loss)) * self.val_loss:
                self.val_loss = val_loss
                self.patience_count = 0
            else:
                self.patience_count += 1
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
            break
        return (sample, sample_mask)
    elif init_proto == 'random':
        return None
    else:
        raise NameError(init_proto)