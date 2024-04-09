"""
TapNet
Credits: Xuchao Zhang, Yifeng Gao, Jessica Lin, and Chang-Tien Lu. Tapnet: Multivariate time series classification
with attentional prototypical network. In Proceedings of the AAAI Conference on Artificial Intelligence,
volume 34, pp. 6845–6852, 2020.
paper: https://ojs.aaai.org/index.php/AAAI/article/view/6165
code: https://github.com/kdd2019-tapnet/tapnet
"""

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import sklearn
import sklearn.metrics
import torch
import pandas as pd


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def loaddata(filename):
    df = pd.read_csv(filename, header=None, delimiter=",")
    a = np.array(df.as_matrix())
    return a


def load_raw_ts(path, dataset, tensor_format=True):
    path = path + "raw/" + dataset + "/"
    x_train = np.load(path + 'X_train.npy')
    y_train = np.load(path + 'y_train.npy')
    x_test = np.load(path + 'X_test.npy')
    y_test = np.load(path + 'y_test.npy')
    ts = np.concatenate((x_train, x_test), axis=0)
    ts = np.transpose(ts, axes=(0, 2, 1))
    labels = np.concatenate((y_train, y_test), axis=0)
    nclass = int(np.amax(labels)) + 1


    train_size = y_train.shape[0]

    total_size = labels.shape[0]
    idx_train = range(train_size)
    idx_val = range(train_size, total_size)
    idx_test = range(train_size, total_size)

    if tensor_format:
        # features = torch.FloatTensor(np.array(features))
        ts = torch.FloatTensor(np.array(ts))
        labels = torch.LongTensor(labels)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

    return ts, labels, idx_train, idx_val, idx_test, nclass


def normalize(mx):
    """Row-normalize sparse matrix"""
    row_sums = mx.sum(axis=1)
    mx = mx.astype('float32')
    row_sums_inverse = 1 / row_sums
    f = mx.multiply(row_sums_inverse)
    return sp.csr_matrix(f).astype('float32')


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def output_conv_size(in_size, kernel_size, stride, padding):

    output = int((in_size - kernel_size + 2 * padding) / stride) + 1

    return output


class TapNet(nn.Module):

    def __init__(self, input_dim, num_steps, num_classes, dropout, filters, kernels, dilation, layers, use_rp, rp_params,
                 use_att=True, use_metric=False, use_lstm=False, use_cnn=True, lstm_dim=128):
        super(TapNet, self).__init__()
        self.nclass = num_classes
        self.dropout = dropout
        self.use_metric = use_metric
        self.use_lstm = use_lstm
        self.use_cnn = use_cnn

        # parameters for random projection
        self.use_rp = use_rp
        self.rp_group, self.rp_dim = rp_params

        # LSTM
        self.channel = input_dim
        self.ts_length = num_steps

        self.lstm_dim = lstm_dim
        self.lstm = nn.LSTM(self.ts_length, self.lstm_dim)

        paddings = [0, 0, 0]
        if self.use_rp:
            self.conv_1_models = nn.ModuleList()
            self.idx = []
            for i in range(self.rp_group):
                self.conv_1_models.append(
                    nn.Conv1d(self.rp_dim, filters[0], kernel_size=kernels[0], dilation=dilation, stride=1,
                              padding=paddings[0]))
                self.idx.append(np.random.permutation(input_dim)[0: self.rp_dim])
        else:
            self.conv_1 = nn.Conv1d(self.channel, filters[0], kernel_size=kernels[0], dilation=dilation, stride=1,
                                    padding=paddings[0])

        self.conv_bn_1 = nn.BatchNorm1d(filters[0])

        self.conv_2 = nn.Conv1d(filters[0], filters[1], kernel_size=kernels[1], stride=1, padding=paddings[1])

        self.conv_bn_2 = nn.BatchNorm1d(filters[1])

        self.conv_3 = nn.Conv1d(filters[1], filters[2], kernel_size=kernels[2], stride=1, padding=paddings[2])

        self.conv_bn_3 = nn.BatchNorm1d(filters[2])

        # compute the size of input for fully connected layers
        fc_input = 0
        if self.use_cnn:
            conv_size = num_steps
            for i in range(len(filters)):
                conv_size = output_conv_size(conv_size, kernels[i], 1, paddings[i])
            fc_input += conv_size
            # * filters[-1]
        if self.use_lstm:
            fc_input += conv_size * self.lstm_dim

        if self.use_rp:
            fc_input = self.rp_group * filters[2] + self.lstm_dim

        # Representation mapping function
        layers = [fc_input] + layers
        self.mapping = nn.Sequential()
        for i in range(len(layers) - 2):
            self.mapping.add_module("fc_" + str(i), nn.Linear(layers[i], layers[i + 1]))
            self.mapping.add_module("bn_" + str(i), nn.BatchNorm1d(layers[i + 1]))
            self.mapping.add_module("relu_" + str(i), nn.LeakyReLU())

        # add last layer
        self.mapping.add_module("fc_" + str(len(layers) - 2), nn.Linear(layers[-2], layers[-1]))
        if len(layers) == 2:  # if only one layer, add batch normalization
            self.mapping.add_module("bn_" + str(len(layers) - 2), nn.BatchNorm1d(layers[-1]))

        # Attention
        att_dim, semi_att_dim = 128, 128
        self.use_att = use_att
        if self.use_att:
            self.att_models = nn.ModuleList()
            for _ in range(num_classes):
                att_model = nn.Sequential(
                    nn.Linear(layers[-1], att_dim),
                    nn.Tanh(),
                    nn.Linear(att_dim, 1)
                )
                self.att_models.append(att_model)
        self.x_proto = None

    def forward(self, x, labels, mode='train'):
        N = x.size(0)

        # LSTM
        if self.use_lstm:
            x_lstm = self.lstm(x)[0]
            x_lstm = x_lstm.mean(1)
            x_lstm = x_lstm.view(N, -1)

        if self.use_cnn:
            # Covolutional Network
            # input ts: # N * C * L
            if self.use_rp:
                for i in range(len(self.conv_1_models)):
                    # x_conv = x
                    x_conv = self.conv_1_models[i](x[:, self.idx[i], :])
                    x_conv = self.conv_bn_1(x_conv)
                    x_conv = F.leaky_relu(x_conv)

                    x_conv = self.conv_2(x_conv)
                    x_conv = self.conv_bn_2(x_conv)
                    x_conv = F.leaky_relu(x_conv)

                    x_conv = self.conv_3(x_conv)
                    x_conv = self.conv_bn_3(x_conv)
                    x_conv = F.leaky_relu(x_conv)

                    x_conv = torch.mean(x_conv, 2)

                    if i == 0:
                        x_conv_sum = x_conv
                    else:
                        x_conv_sum = torch.cat([x_conv_sum, x_conv], dim=1)

                x_conv = x_conv_sum
            else:
                x_conv = x
                x_conv = self.conv_1(x_conv)  # N * C * L
                x_conv = self.conv_bn_1(x_conv)
                x_conv = F.leaky_relu(x_conv)

                x_conv = self.conv_2(x_conv)
                x_conv = self.conv_bn_2(x_conv)
                x_conv = F.leaky_relu(x_conv)

                x_conv = self.conv_3(x_conv)
                x_conv = self.conv_bn_3(x_conv)
                x_conv = F.leaky_relu(x_conv)

                x_conv = x_conv.view(N, -1)

        if self.use_lstm and self.use_cnn:
            x = torch.cat([x_conv, x_lstm], dim=1)
        elif self.use_lstm:
            x = x_lstm
        elif self.use_cnn:
            x = x_conv

        # linear mapping to low-dimensional space
        x = self.mapping(x)

        # generate the class protocal with dimension C * D (nclass * dim)
        if mode == 'train':
            proto_list = []
            for i in range(self.nclass):
                idx = (labels.squeeze() == i).nonzero().squeeze(1)
                if self.use_att:
                    A = self.att_models[i](x[idx])  # N_k * 1
                    A = torch.transpose(A, 1, 0)  # 1 * N_k
                    A = F.softmax(A, dim=1)  # softmax over N_k

                    class_repr = torch.mm(A, x[idx])  # 1 * L
                    class_repr = torch.transpose(class_repr, 1, 0)  # L * 1
                else:  # if do not use attention, simply use the mean of training samples with the same labels.
                    class_repr = x[idx].mean(0)  # L * 1
                proto_list.append(class_repr.view(1, -1))
            x_proto = torch.cat(proto_list, dim=0)
        else:
            x_proto = self.x_proto.to(x.device)

        dists = euclidean_dist(x, x_proto)
        return torch.exp(-0.5 * dists)

    def get_prototypes(self, train_loader, device):
        weights = None
        for batchi, batch in enumerate(train_loader):
            x, train_mask, y = batch
            x = x.view(-1, x.size(-2), x.size(-1)).to(torch.float32).permute(0, 2, 1).to(device)
            labels = y.view(-1).long().to(device)
            N = x.size(0)

            # LSTM
            if self.use_lstm:
                x_lstm = self.lstm(x)[0]
                x_lstm = x_lstm.mean(1)
                x_lstm = x_lstm.view(N, -1)

            if self.use_cnn:
                # Covolutional Network
                # input ts: # N * C * L
                if self.use_rp:
                    for i in range(len(self.conv_1_models)):
                        # x_conv = x
                        x_conv = self.conv_1_models[i](x[:, self.idx[i], :])
                        x_conv = self.conv_bn_1(x_conv)
                        x_conv = F.leaky_relu(x_conv)

                        x_conv = self.conv_2(x_conv)
                        x_conv = self.conv_bn_2(x_conv)
                        x_conv = F.leaky_relu(x_conv)

                        x_conv = self.conv_3(x_conv)
                        x_conv = self.conv_bn_3(x_conv)
                        x_conv = F.leaky_relu(x_conv)

                        x_conv = torch.mean(x_conv, 2)

                        if i == 0:
                            x_conv_sum = x_conv
                        else:
                            x_conv_sum = torch.cat([x_conv_sum, x_conv], dim=1)

                    x_conv = x_conv_sum
                else:
                    x_conv = x
                    x_conv = self.conv_1(x_conv)  # N * C * L
                    x_conv = self.conv_bn_1(x_conv)
                    x_conv = F.leaky_relu(x_conv)

                    x_conv = self.conv_2(x_conv)
                    x_conv = self.conv_bn_2(x_conv)
                    x_conv = F.leaky_relu(x_conv)

                    x_conv = self.conv_3(x_conv)
                    x_conv = self.conv_bn_3(x_conv)
                    x_conv = F.leaky_relu(x_conv)

                    x_conv = x_conv.view(N, -1)

            if self.use_lstm and self.use_cnn:
                x = torch.cat([x_conv, x_lstm], dim=1)
            elif self.use_lstm:
                x = x_lstm
            elif self.use_cnn:
                x = x_conv

            # linear mapping to low-dimensional space
            x = self.mapping(x)

            # generate the class protocal with dimension C * D (nclass * dim)
            weight_list = torch.zeros(x.size(0), device=device)
            for i in range(self.nclass):
                idx = (labels.squeeze() == i).nonzero().squeeze(1)
                if self.use_att:
                    A = self.att_models[i](x[idx])  # N_k * 1
                    A = torch.transpose(A, 1, 0)  # 1 * N_k
                    weight_list[idx] = A.squeeze()
            if weights is None:
                weights = weight_list.unsqueeze(0)
                lab = labels
            else:
                weights = torch.cat([weights, weight_list.unsqueeze(0)], dim=0)
                lab = torch.cat([lab, labels], dim=0)
            if lab.size(0) > 1e6:
                break
        batch_max = batchi
        un, deux = weights.shape
        weights = weights.flatten()
        for i in range(self.nclass):
            idx = (lab.squeeze() == i).nonzero().squeeze()
            weights[idx] = torch.softmax(weights[idx], dim=0)
        weights = weights.reshape(un, deux)
        print("Done")

        proto_list = [None for _ in range(self.nclass)]
        for batchi, batch in enumerate(train_loader):
            x, train_mask, y = batch
            x = x.view(-1, x.size(-2), x.size(-1)).to(torch.float32).permute(0, 2, 1).to(device)
            labels = y.view(-1).long().to(device)
            N = x.size(0)

            # LSTM
            if self.use_lstm:
                x_lstm = self.lstm(x)[0]
                x_lstm = x_lstm.mean(1)
                x_lstm = x_lstm.view(N, -1)

            if self.use_cnn:
                # Covolutional Network
                # input ts: # N * C * L
                if self.use_rp:
                    for i in range(len(self.conv_1_models)):
                        # x_conv = x
                        x_conv = self.conv_1_models[i](x[:, self.idx[i], :])
                        x_conv = self.conv_bn_1(x_conv)
                        x_conv = F.leaky_relu(x_conv)

                        x_conv = self.conv_2(x_conv)
                        x_conv = self.conv_bn_2(x_conv)
                        x_conv = F.leaky_relu(x_conv)

                        x_conv = self.conv_3(x_conv)
                        x_conv = self.conv_bn_3(x_conv)
                        x_conv = F.leaky_relu(x_conv)

                        x_conv = torch.mean(x_conv, 2)

                        if i == 0:
                            x_conv_sum = x_conv
                        else:
                            x_conv_sum = torch.cat([x_conv_sum, x_conv], dim=1)

                    x_conv = x_conv_sum
                else:
                    x_conv = x
                    x_conv = self.conv_1(x_conv)  # N * C * L
                    x_conv = self.conv_bn_1(x_conv)
                    x_conv = F.leaky_relu(x_conv)

                    x_conv = self.conv_2(x_conv)
                    x_conv = self.conv_bn_2(x_conv)
                    x_conv = F.leaky_relu(x_conv)

                    x_conv = self.conv_3(x_conv)
                    x_conv = self.conv_bn_3(x_conv)
                    x_conv = F.leaky_relu(x_conv)

                    x_conv = x_conv.view(N, -1)

            if self.use_lstm and self.use_cnn:
                x = torch.cat([x_conv, x_lstm], dim=1)
            elif self.use_lstm:
                x = x_lstm
            elif self.use_cnn:
                x = x_conv

            # linear mapping to low-dimensional space
            x = self.mapping(x)

            for i in range(self.nclass):
                idx = (labels.squeeze() == i).nonzero().squeeze(1)
                if proto_list[i] is None:
                    proto_list[i] = torch.mm(weights[batchi][idx].unsqueeze(0), x[idx]).squeeze()
                else:
                    proto_list[i] += torch.mm(weights[batchi][idx].unsqueeze(0), x[idx]).squeeze()

            if batchi == batch_max:
                break
        print("Done")
        self.x_proto = torch.stack(proto_list, dim=0)

    def accuracy(self, output, labels):
        preds = output.max(1)[1].cpu().numpy()
        labels = labels.cpu().numpy()
        accuracy_score = (sklearn.metrics.accuracy_score(labels, preds))

        return accuracy_score
