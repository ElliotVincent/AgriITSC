import json
import os
import numpy as np
import torch.utils.data as tdata


MEAN = {'train': np.array([601.7648,  884.5823, 1197.0653, 2758.9124]),
        'val': np.array([602.8802,  885.1292, 1193.7650, 2760.9680]),
        'test': np.array([574.6069,  838.3013, 1183.8048, 2549.9424])
        }

STD = {'train': np.array([237.1001, 293.0519, 537.3166, 663.8234]),
       'val': np.array([237.3892, 293.1587, 536.6476, 670.5874]),
       'test': np.array([173.5428, 223.8085, 429.8163, 583.0292])
       }


class SADataset(tdata.Dataset):
    def __init__(
        self,
        folder,
        split='train',
        with_img_id=False,
    ):
        """
        Pytorch Dataset class to load samples from the DENETHOR dataset, for semantic segmentation.
        """
        super(SADataset, self).__init__()

        self.folder = folder
        self.split = split
        self.with_img_id = with_img_id
        fold_item = {}
        with open(os.path.join(self.folder, '{}/num_seq.json'.format(split)), 'rb') as f:
            len = int(json.load(f)['len'])
        self.len = len
        self.fold_item = fold_item
        print('Dataset size: {}'.format(self.len * 512))

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        data_path = os.path.join(self.folder, '{}/data_{}.npy'.format(self.split, item))
        label_path = os.path.join(self.folder, '{}/annotations_{}.npy'.format(self.split, item))
        data = (np.load(data_path) - MEAN[self.split]) / STD[self.split]
        if self.with_img_id:
            annotations = np.load(label_path) - [1, 0]
        else:
            annotations = np.load(label_path)[:, 0] - 1
        mask = np.ones((data.shape[0], data.shape[1]), dtype=int)
        return data, mask, annotations

