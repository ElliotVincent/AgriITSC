import collections.abc
import re

import torch
from torch.nn import functional as F
from torch.utils import data
from pathlib import Path
from src.dataset.pastis import PASTISDataset
from src.dataset.denethor import DENETHORDataset
from src.dataset.sa import SADataset
from src.dataset.breizh import BREIZHDataset
from src.dataset.ts2c import TS2CDataset
from src.model.upssits import AgriSits
from src.model.mlp import MLP
from src.model.means import Means
from src.model.kmeans import KMeans

np_str_obj_array_pattern = re.compile(r"[SaUO]")


def pad_tensor(x, l, pad_value=0):
    padlen = l - x.shape[0]
    pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, padlen]
    return F.pad(x, pad=pad, value=pad_value)


def pad_collate(batch, pad_value=0):
    # modified default_collate from the official pytorch repo
    # https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if len(elem.shape) > 0:
            sizes = [e.shape[0] for e in batch]
            m = max(sizes)
            if not all(s == m for s in sizes):
                # pad tensors which have a temporal dimension
                batch = [pad_tensor(e, m, pad_value=pad_value) for e in batch]
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError("Format not managed : {}".format(elem.dtype))

            return pad_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, collections.abc.Mapping):
        return {key: pad_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(pad_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [pad_collate(samples) for samples in transposed]

    raise TypeError("Format not managed : {}".format(elem_type))


def get_ntrainparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def coerce_to_path_and_check_exist(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError('{} does not exist'.format(path.absolute()))
    return path


def coerce_to_path_and_create_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_train_dataset(config):
    name = config.get("name", "denethor")
    if name == 'denethor':
        train_dataset = DENETHORDataset('./datasets/DENETHOR/', 'train')
    elif name == 'sa':
        train_dataset = SADataset('./datasets/SA/', 'train')
    elif name == 'ts2c':
        train_dataset = TS2CDataset('./datasets/TimeSen2Crop/', 'train')
    elif name == 'breizh':
        train_dataset = BREIZHDataset('./datasets/breizhcrops_dataset/', 'train')
    elif name == 'pastis':
        fold = config.get("train_fold", [1, 2, 3])
        use_ndvi = config.get("use_ndvi", False)
        train_dataset = PASTISDataset('./datasets/PASTIS_filtered/', fold, use_ndvi=use_ndvi)
    else:
        raise NameError(name)
    return train_dataset


def get_val_dataset(config):
    name = config.get("name", "denethor")
    if name == 'denethor':
        with_img_id = config.get("with_img_id", False)
        val_dataset = DENETHORDataset('./datasets/DENETHOR/', 'val', with_img_id=with_img_id)
    elif name == 'sa':
        with_img_id = config.get("with_img_id", False)
        val_dataset = SADataset('./datasets/SA/', 'val', with_img_id=with_img_id)
    elif name == 'ts2c':
        val_dataset = TS2CDataset('./datasets/TimeSen2Crop/', 'val')
    elif name == 'breizh':
        val_dataset = BREIZHDataset('./datasets/breizhcrops_dataset/', 'val')
    elif name == 'pastis':
        fold = config.get("val_fold", [4])
        use_ndvi = config.get("use_ndvi", False)
        val_dataset = PASTISDataset('./datasets/PASTIS_filtered/', fold, use_ndvi=use_ndvi)
    else:
        raise NameError(name)
    return val_dataset


def get_test_dataset(config):
    name = config.get("name", "denethor")
    if name == 'denethor':
        with_img_id = config.get("with_img_id", False)
        test_dataset = DENETHORDataset('./datasets/DENETHOR/', 'test', with_img_id=with_img_id)
    elif name == 'sa':
        with_img_id = config.get("with_img_id", False)
        test_dataset = SADataset('./datasets/SA/', 'test', with_img_id=with_img_id)
    elif name == 'ts2c':
        test_dataset = TS2CDataset('./datasets/TimeSen2Crop/', 'test')
    elif name == 'breizh':
        test_dataset = BREIZHDataset('./datasets/breizhcrops_dataset/', 'test')
    elif name == 'pastis':
        fold = config.get("test_fold", [5])
        use_ndvi = config.get("use_ndvi", False)
        test_dataset = PASTISDataset('./datasets/PASTIS_filtered/', fold, use_ndvi=use_ndvi)
    else:
        raise NameError(name)
    return test_dataset


def get_dataset(config, split):
    if split == 'train':
        return get_train_dataset(config)
    elif split == 'val':
        return get_val_dataset(config)
    elif split == 'test':
        return get_test_dataset(config)
    else:
        raise NameError(split)


def get_model(config):
    name = config.pop("name", "agrisits")
    if name == 'agrisits':
        model = AgriSits(**config)
    elif name == 'mlp':
        model = MLP(**config)
    elif name == 'means':
        model = Means(**config)
    elif name == 'kmeans':
        model = KMeans(**config)
    else:
        raise NameError(name)
    config["name"] = name
    return model
