import json
import os
import numpy as np
import torch.utils.data as tdata


class TS2CDataset(tdata.Dataset):
    def __init__(
        self,
        split='train',
    ):
        """
        Pytorch Dataset class to load samples from the TimeSen2Crop dataset, for semantic segmentation.
        """
        super(TS2CDataset, self).__init__()

        self.folder = './datasets/TimeSen2Crop/'
        self.split = split
        self.file_size = 512
        self.num_steps = 363

        with open(os.path.join(self.folder, 'params.json'), 'rb') as f:
            params = json.load(f)
        self.MEAN = params["MEAN"]
        self.STD = params["STD"]
        for k in self.MEAN.keys():
            self.MEAN[k] = np.array(self.MEAN[k])
            self.STD[k] = np.array(self.STD[k])
        fold_item = {}
        if self.split == 'all':
            len = []
            for curr_split in ['train', 'val', 'test']:
                with open(os.path.join(self.folder, '{}/num_seq.json'.format(curr_split)), 'rb') as f:
                    len.append(int(json.load(f)['len']))
            self.len = np.sum(len)
            self.len_list = len
        else:
            with open(os.path.join(self.folder, '{}/num_seq.json'.format(split)), 'rb') as f:
                len = int(json.load(f)['len'])
            self.len = len
        self.fold_item = fold_item
        print('{} dataset size: {}'.format(self.split, self.len * self.file_size))

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        if self.split == 'all':
            if item < self.len_list[0]:
                curr_split = 'train'
                curr_item = item
            elif (item - self.len_list[0]) < self.len_list[1]:
                curr_split = 'val'
                curr_item = item - self.len_list[0]
            else:
                curr_split = 'test'
                curr_item = item - self.len_list[0] - self.len_list[1]
        else:
            curr_split = self.split
            curr_item = item
        data_path = os.path.join(self.folder, '{}/data_{}.npy'.format(curr_split, curr_item))
        mask_path = os.path.join(self.folder, '{}/mask_{}.npy'.format(curr_split, curr_item))
        label_path = os.path.join(self.folder, '{}/annotations_{}.npy'.format(curr_split, curr_item))
        mask = np.load(mask_path)
        data = np.load(data_path)
        data = np.divide(data - mask[..., None] * np.tile(self.MEAN[curr_split], (self.file_size, self.num_steps, 1)),
                         mask[..., None] * np.tile(self.STD[curr_split], (self.file_size, self.num_steps, 1)),
                         out=np.zeros_like(data),
                         where=mask[..., None] != 0)
        annotations = np.load(label_path)
        return data, mask, annotations

