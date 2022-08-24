import json
import os
import numpy as np
import torch.utils.data as tdata


MEAN = {'train': np.array([533.7825, 786.2733, 830.5689, 1297.8242, 2434.4253, 2783.7385, 3079.4771, 2202.5874, 1462.9695]),
        'val': np.array([449.5133, 718.3015, 741.5708, 1215.3044, 2449.9624, 2839.8047, 3127.3904, 2015.7549, 1262.7659]),
        'test': np.array([592.0566, 876.2975, 911.8779, 1407.2795, 2645.6729, 3038.6868, 3340.0679, 2306.5444, 1508.0461])
        }

STD = {'train': np.array([350.3590, 385.9648, 548.3232, 525.6872, 848.0714, 1055.0251, 1173.2668, 779.6679, 732.4794]),
       'val': np.array([291.3542, 339.4813, 501.5322, 469.6360, 795.7408, 1032.2260, 1169.5033, 669.2755, 606.1485]),
       'test': np.array([359.8037, 402.4475, 595.7352, 540.9295, 783.4622, 987.8573, 1092.6587, 784.7286, 766.4512])
       }


class TS2CDataset(tdata.Dataset):
    def __init__(
        self,
        folder,
        split='train',
    ):
        """
        Pytorch Dataset class to load samples from the TimeSen2Crop dataset, for semantic segmentation.
        """
        super(TS2CDataset, self).__init__()

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
        mask_path = os.path.join(self.folder, '{}/mask_{}.npy'.format(self.split, item))
        label_path = os.path.join(self.folder, '{}/annotations_{}.npy'.format(self.split, item))
        mask = np.load(mask_path)
        data = np.load(data_path)
        data = np.divide(data - mask[..., None] * np.tile(MEAN[self.split], (512, 363, 1)),
                         mask[..., None] * np.tile(STD[self.split], (512, 363, 1)),
                         out=np.zeros_like(data),
                         where=mask[..., None] != 0)
        annotations = np.load(label_path)
        return data, mask, annotations

