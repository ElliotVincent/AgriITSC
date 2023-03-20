"""
Taken from https://github.com/davidtvs/PyTorch-ENet/blob/master/metric/confusionmatrix.py
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MeanShift


class Metric(object):
    """Base class for all metrics.
    From: https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py
    """

    def reset(self):
        pass

    def add(self):
        pass

    def value(self):
        pass


class ConfusionMatrix(Metric):
    """Constructs a confusion matrix for a multi-class classification problems.
    Does not support multi-label, multi-class problems.
    Keyword arguments:
    - num_classes (int): number of classes in the classification problem.
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.
    Modified from: https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py
    """

    def __init__(self, num_classes, num_prototypes, normalized=False, device='cpu', lazy=True):
        super().__init__()
        if device == 'cpu':
            self.conf = np.ndarray((num_classes, num_prototypes), dtype=np.int64)
        else:
            self.conf = torch.zeros((num_classes, num_prototypes)).cuda()
        self.normalized = normalized
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.device = device
        self.reset()
        self.lazy = lazy
        self.purity_conf = None
        self.meanshift_conf = None


    def purity_assignment(self, matching=None):
        conf_matrix = np.zeros((self.num_classes, self.num_classes))
        if matching is None:
            matching = np.argmax(self.conf, axis=0)
        for i, col in enumerate(matching):
            conf_matrix[:, col] = conf_matrix[:, col] + self.conf[:, i]
        self.purity_conf = conf_matrix
        return matching

    def meanshift_assignment(self, model, matching=None):
        conf_matrix = np.zeros((self.num_classes, self.num_classes))
        if matching is None:
            prototypes = model.module.prototypes.flatten(1).detach().cpu().numpy()
            matching = iterative_mean_shift(self.num_classes, self.num_prototypes, prototypes)
        for i, col in enumerate(matching):
            conf_matrix[:, col] = conf_matrix[:, col] + self.conf[:, i]
        self.meanshift_conf = conf_matrix
        self.hungarian_match()
        return matching

    def hungarian_match(self):
        row_id, col_id = linear_sum_assignment(-self.meanshift_conf)
        self.meanshift_conf = self.meanshift_conf[:, col_id]

    def get_acc(self):
        # return float(np.diag(self.meanshift_conf).sum() / self.conf.sum() * 100), \
        return float(np.diag(self.purity_conf).sum() / self.conf.sum() * 100)

    def get_acc_per_class(self):
        # return list(np.diag(self.meanshift_conf) / self.conf.sum(1) * 100), \
        return list(np.diag(self.purity_conf) / np.maximum(self.conf.sum(1), 1) * 100)

    def reset(self):
        if self.device == 'cpu':
            self.conf.fill(0)
        else:
            self.conf = torch.zeros(self.conf.shape).cuda()

    def add(self, predicted, target):
        """Computes the confusion matrix
        The shape of the confusion matrix is K x K, where K is the number
        of classes.
        Keyword arguments:
        - predicted (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        predicted scores obtained from the model for N examples and K classes,
        or an N-tensor/array of integer values between 0 and K-1.
        - target (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        ground-truth classes for N examples and K classes, or an N-tensor/array
        of integer values between 0 and K-1.
        """

        # If target and/or predicted are tensors, convert them to numpy arrays
        if self.device == 'cpu':
            if torch.is_tensor(predicted):
                predicted = predicted.cpu().numpy()
            if torch.is_tensor(target):
                target = target.cpu().numpy()

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if len(predicted.shape) != 1:
            assert predicted.shape[1] == self.num_classes, \
                'number of predictions does not match size of confusion matrix'
            predicted = predicted.argmax(1)
        else:
            if not self.lazy:
                assert (predicted.max() < self.num_classes) and (predicted.min() >= 0), \
                    'predicted values are not between 0 and k-1'

        if len(target.shape) != 1:
            if not self.lazy:
                assert target.shape[1] == self.num_classes, \
                    'Onehot target does not match size of confusion matrix'
                assert (target >= 0).all() and (target <= 1).all(), \
                    'in one-hot encoding, target values should be 0 or 1'
                assert (target.sum(1) == 1).all(), \
                    'multi-label setting is not supported'
            target = target.argmax(1)
        else:
            if not self.lazy:
                assert (target.max() < self.num_classes) and (target.min() >= 0), \
                    'target values are not between 0 and k-1'

        # hack for bincounting 2 arrays together
        x = predicted + self.num_prototypes * target

        if self.device == 'cpu':
            bincount_2d = np.bincount(
                x.astype(np.int64), minlength=self.num_classes * self.num_prototypes)
            assert bincount_2d.size == self.num_classes * self.num_prototypes
            conf = bincount_2d.reshape((self.num_classes, self.num_prototypes))
        else:
            bincount_2d = torch.bincount(
                x, minlength=self.num_classes ** self.num_prototypes)

            conf = bincount_2d.view((self.num_classes, self.num_prototypes))
        self.conf += conf

    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf


class CustomMeter:
    def __init__(
            self,
            num_classes=17,
    ):
        super(CustomMeter, self).__init__()
        self.num_classes = num_classes
        self.vals = [0. for _ in range(self.num_classes)]
        self.counts = [0. for _ in range(self.num_classes)]

    def add(self, vals, counts):
        for class_id in range(self.num_classes):
            self.vals[class_id] += vals[class_id].item()
            self.counts[class_id] += counts[class_id]

    def value(self, mode='mean'):
        if mode == 'density':
            total = sum(self.vals)
            return [v / total for v in self.vals]
        return [v / max(c, 1) for v, c in zip(self.vals, self.counts)]

    def reset(self):
        self.vals = [0. for _ in range(self.num_classes)]
        self.counts = [0. for _ in range(self.num_classes)]


def iterative_mean_shift(num_classes, num_prototypes, prototypes):
    curr_num = np.inf
    labels = {i: i for i in range(num_prototypes)}
    while curr_num > num_classes:
        proto_clustering = MeanShift().fit(prototypes)
        matching = proto_clustering.labels_
        labels = {i: matching[labels[i]] for i in range(num_prototypes)}
        curr_num = np.max(matching) + 1
        prototypes = proto_clustering.cluster_centers_
    return list(labels.values())
