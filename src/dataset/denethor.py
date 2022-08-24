import json
import os
import numpy as np
import torch.utils.data as tdata


MEAN = {'train': np.array([583.3404,  800.4510,  944.2325, 2877.1177]),
        'val': np.array([585.2572,  804.8460,  950.9557, 2862.8772]),
        'test': np.array([506.5952,  698.1039,  786.7234, 2831.5496])
        }

STD = {'train': np.array([216.4638, 285.7197, 465.6355, 811.3859]),
       'val': np.array([207.4022, 273.3199, 443.6665, 824.6351]),
       'test': np.array([202.2838, 263.1939, 418.8028, 832.0331])
       }

PROP_TRAIN = np.array([21.2458, 16.4559,  9.0319,  1.9580, 17.8208, 16.1784,  0.6939,  9.8471, 6.7681])
PROP_VAL = np.array([13.2543, 15.1774,  8.7612,  1.3291, 22.2181, 21.4563,  0.4079,  7.9128, 9.4829])


class DENETHORDataset(tdata.Dataset):
    def __init__(
        self,
        folder,
        split='train',
        with_img_id=False,
    ):
        """
        Pytorch Dataset class to load samples from the DENETHOR dataset, for semantic segmentation.
        """
        super(DENETHORDataset, self).__init__()

        self.folder = folder
        self.split = split
        self.with_img_id=with_img_id
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

