import json
import os
import numpy as np
import torch.utils.data as tdata


MEAN = {'train': np.array([0.3915, 0.4584, 0.4976, 0.1969, 0.0277, 0.2844, 0.2024, 0.3668, 0.3377,
        0.3444, 0.3641, 0.4364, 0.4769]),
        'val': np.array([0.3949, 0.4623, 0.5037, 0.1888, 0.0243, 0.2967, 0.2088, 0.3703, 0.3417,
        0.3485, 0.3676, 0.4413, 0.4832]),
        'test': np.array([0.3822, 0.4493, 0.4894, 0.1901, 0.0282, 0.2885, 0.2036, 0.3574, 0.3290,
        0.3356, 0.3552, 0.4274, 0.4680])
        }

STD = {'train': np.array([0.2337, 0.1984, 0.2060, 0.1291, 0.0498, 0.1345, 0.1093, 0.2411, 0.2292,
        0.2559, 0.2438, 0.2121, 0.2090]),
       'val': np.array([0.2388, 0.2041, 0.2129, 0.1259, 0.0438, 0.1421, 0.1140, 0.2465, 0.2345,
        0.2615, 0.2485, 0.2177, 0.2154]),
       'test': np.array([0.2313, 0.1936, 0.2008, 0.1288, 0.0505, 0.1358, 0.1106, 0.2383, 0.2262,
        0.2521, 0.2395, 0.2077, 0.2045])
       }

PROP_TRAIN = np.array([21.2458, 16.4559,  9.0319,  1.9580, 17.8208, 16.1784,  0.6939,  9.8471, 6.7681])
PROP_VAL = np.array([13.2543, 15.1774,  8.7612,  1.3291, 22.2181, 21.4563,  0.4079,  7.9128, 9.4829])


class BREIZHDataset(tdata.Dataset):
    def __init__(
        self,
        folder,
        split='train'
    ):
        super(BREIZHDataset, self).__init__()

        self.folder = folder
        self.split = split
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
        annotations = np.load(label_path)
        mask = np.ones((data.shape[0], data.shape[1]), dtype=int)
        if data.shape[0] != 512:
            print(item)
        return data, mask, annotations

