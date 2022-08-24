import json
import os
import numpy as np
import torch.utils.data as tdata
import pickle


class PASTISDataset(tdata.Dataset):
    def __init__(
        self,
        folder,
        folds=None,
        use_ndvi=False,
    ):
        """
        Pytorch Dataset class to load samples from the PASTIS dataset, for semantic segmentation.
        """
        super(PASTISDataset, self).__init__()
        self.folder = folder
        self.use_ndvi = use_ndvi

        # Get metadata
        print(folder)

        # Select Fold samples
        if folds is None:
            folds = [1, 2, 3, 4, 5]

        len = 0
        fold_item = {}
        for fold in folds:
            with open(os.path.join(self.folder, 'Fold_{}/num_seq.json'.format(fold)), 'rb') as f:
                num_seq = sum(json.load(f).values())
            for k in range(num_seq // 512):
                fold_item[len+k] = (fold, k)
            len += (num_seq // 512)
        self.len = len
        self.fold_item = fold_item
        print('Dataset size: {}'.format(self.len * 512))

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        fold, item = self.fold_item[item]
        data_path = os.path.join(self.folder, 'Fold_{}/data_{}.pkl'.format(fold, item))
        mask_path = os.path.join(self.folder, 'Fold_{}/mask_{}.pkl'.format(fold, item))
        label_path = os.path.join(self.folder, 'Fold_{}/annotations_{}.npy'.format(fold, item))
        data = pickle.load(open(data_path, 'rb'))
        data = data.toarray().reshape((-1, 406, 11))
        if not self.use_ndvi:
            data = data[..., :10]
        mask = pickle.load(open(mask_path, 'rb'))
        mask = mask.toarray()
        annotations = np.load(label_path)[:, 0]
        return data, mask, annotations

