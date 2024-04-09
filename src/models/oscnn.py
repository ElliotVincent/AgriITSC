"""
Omni-scale CNN
Credits: Wensi Tang, Guodong Long, Lu Liu, Tianyi Zhou, Michael Blumenstein, and Jing Jiang. Omni-scale CNNs:
a simple and effective kernel size configuration for time series classification. In International Conference
on Learning Representations, 2022
paper: https://openreview.net/forum?id=PDYs7Z2XFGv
code: https://github.com/Wensi-Tang/OS-CNN
"""

import math
import copy
import os
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def get_Prime_number_in_a_range(start, end):
    Prime_list = []
    for val in range(start, end + 1):
        prime_or_not = True
        for n in range(2, val):
            if (val % n) == 0:
                prime_or_not = False
                break
        if prime_or_not:
            Prime_list.append(val)
    return Prime_list


def get_out_channel_number(paramenter_layer, in_channel, prime_list):
    out_channel_expect = int(paramenter_layer / (in_channel * sum(prime_list)))
    return out_channel_expect


def generate_layer_parameter_list(start, end, paramenter_number_of_layer_list, in_channel=1):
    prime_list = get_Prime_number_in_a_range(start, end)
    if prime_list == []:
        print('start = ', start, 'which is larger than end = ', end)
    paramenter_number_of_layer_list[0] = paramenter_number_of_layer_list[0] * in_channel
    input_in_channel = in_channel
    layer_parameter_list = []
    for paramenter_number_of_layer in paramenter_number_of_layer_list:
        out_channel = get_out_channel_number(paramenter_number_of_layer, in_channel, prime_list)

        tuples_in_layer = []
        for prime in prime_list:
            tuples_in_layer.append((in_channel, out_channel, prime))
        in_channel = len(prime_list) * out_channel

        layer_parameter_list.append(tuples_in_layer)

    tuples_in_layer_last = []
    first_out_channel = len(prime_list) * get_out_channel_number(paramenter_number_of_layer_list[0], input_in_channel,
                                                                 prime_list)
    tuples_in_layer_last.append((in_channel, first_out_channel, start))
    tuples_in_layer_last.append((in_channel, first_out_channel, start + 1))
    layer_parameter_list.append(tuples_in_layer_last)
    return layer_parameter_list


def calculate_mask_index(kernel_length_now, largest_kernel_lenght):
    right_zero_mast_length = math.ceil((largest_kernel_lenght - 1) / 2) - math.ceil((kernel_length_now - 1) / 2)
    left_zero_mask_length = largest_kernel_lenght - kernel_length_now - right_zero_mast_length
    return left_zero_mask_length, left_zero_mask_length + kernel_length_now


def creat_mask(number_of_input_channel, number_of_output_channel, kernel_length_now, largest_kernel_lenght):
    ind_left, ind_right = calculate_mask_index(kernel_length_now, largest_kernel_lenght)
    mask = np.ones((number_of_input_channel, number_of_output_channel, largest_kernel_lenght))
    mask[:, :, 0:ind_left] = 0
    mask[:, :, ind_right:] = 0
    return mask


def creak_layer_mask(layer_parameter_list):
    largest_kernel_lenght = layer_parameter_list[-1][-1]
    mask_list = []
    init_weight_list = []
    bias_list = []
    for i in layer_parameter_list:
        conv = torch.nn.Conv1d(in_channels=i[0], out_channels=i[1], kernel_size=i[2])
        ind_l, ind_r = calculate_mask_index(i[2], largest_kernel_lenght)
        big_weight = np.zeros((i[1], i[0], largest_kernel_lenght))
        big_weight[:, :, ind_l:ind_r] = conv.weight.detach().numpy()

        bias_list.append(conv.bias.detach().numpy())
        init_weight_list.append(big_weight)

        mask = creat_mask(i[1], i[0], i[2], largest_kernel_lenght)
        mask_list.append(mask)

    mask = np.concatenate(mask_list, axis=0)
    init_weight = np.concatenate(init_weight_list, axis=0)
    init_bias = np.concatenate(bias_list, axis=0)
    return mask.astype(np.float32), init_weight.astype(np.float32), init_bias.astype(np.float32)


class build_layer_with_layer_parameter(nn.Module):
    def __init__(self, layer_parameters):
        super(build_layer_with_layer_parameter, self).__init__()

        os_mask, init_weight, init_bias = creak_layer_mask(layer_parameters)

        in_channels = os_mask.shape[1]
        out_channels = os_mask.shape[0]
        max_kernel_size = os_mask.shape[-1]

        self.weight_mask = nn.Parameter(torch.from_numpy(os_mask), requires_grad=False)

        self.padding = nn.ConstantPad1d((int((max_kernel_size - 1) / 2), int(max_kernel_size / 2)), 0)

        self.conv1d = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=max_kernel_size)
        self.conv1d.weight = nn.Parameter(torch.from_numpy(init_weight), requires_grad=True)
        self.conv1d.bias = nn.Parameter(torch.from_numpy(init_bias), requires_grad=True)

        self.bn = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, X):
        self.conv1d.weight.data = self.conv1d.weight * self.weight_mask
        # self.conv1d.weight.data.mul_(self.weight_mask)
        result_1 = self.padding(X)
        result_2 = self.conv1d(result_1)
        result_3 = self.bn(result_2)
        result = F.relu(result_3)
        return result


class OS_CNN_block(nn.Module):
    def __init__(self, layer_parameter_list, n_class, squeeze_layer=True):
        super(OS_CNN_block, self).__init__()
        self.squeeze_layer = squeeze_layer
        self.layer_parameter_list = layer_parameter_list
        self.layer_list = []

        for i in range(len(layer_parameter_list)):
            layer = build_layer_with_layer_parameter(layer_parameter_list[i])
            self.layer_list.append(layer)

        self.net = nn.Sequential(*self.layer_list)

        self.averagepool = nn.AdaptiveAvgPool1d(1)

        out_put_channel_numebr = 0
        for final_layer_parameters in layer_parameter_list[-1]:
            out_put_channel_numebr = out_put_channel_numebr + final_layer_parameters[1]

        self.hidden = nn.Linear(out_put_channel_numebr, n_class)

    def forward(self, X):

        X = self.net(X)

        if self.squeeze_layer:
            X = self.averagepool(X)
            X = X.squeeze_(-1)
            X = self.hidden(X)
        return X


def check_channel_limit(os_block_layer_parameter_list, n_input_channel, mid_channel_limit):
    out_channel_each = 0
    for conv_in in os_block_layer_parameter_list[-1]:
        out_channel_each = out_channel_each + conv_in[1]
    total_temp_channel = n_input_channel * out_channel_each
    if total_temp_channel <= mid_channel_limit:
        return os_block_layer_parameter_list
    else:

        temp_channel_each = max(int(mid_channel_limit / (n_input_channel * len(os_block_layer_parameter_list[-1]))), 1)
        for i in range(len(os_block_layer_parameter_list[-1])):
            os_block_layer_parameter_list[-1][i] = (os_block_layer_parameter_list[-1][i][0],
                                                    temp_channel_each,
                                                    os_block_layer_parameter_list[-1][i][2])
        print('reshape temp channel from ', total_temp_channel, ' to ', n_input_channel, ' * ', temp_channel_each, )
        return os_block_layer_parameter_list


class OS_CNN(nn.Module):
    def __init__(self, layer_parameter_list, n_class, n_input_channel, squeeze_layer=True):
        super(OS_CNN, self).__init__()

        self.mid_channel_limit = 1000
        self.squeeze_layer = squeeze_layer
        self.layer_parameter_list = layer_parameter_list
        self.OS_block_list = nn.ModuleList()

        os_block_layer_parameter_list = copy.deepcopy(layer_parameter_list[:-1])
        os_block_layer_parameter_list = check_channel_limit(os_block_layer_parameter_list, n_input_channel,
                                                            self.mid_channel_limit)
        for nth in range(n_input_channel):
            torch_OS_CNN_block = OS_CNN_block(os_block_layer_parameter_list, n_class, False)
            self.OS_block_list.append(torch_OS_CNN_block)

        rf_size = layer_parameter_list[0][-1][-1]
        in_channel_we_want = len(layer_parameter_list[1]) * os_block_layer_parameter_list[-1][-1][1] * n_input_channel

        layer_parameter_list = generate_layer_parameter_list(1, rf_size + 1,
                                                             [8 * 128, (5 * 128 * 256 + 2 * 256 * 128) / 2],
                                                             in_channel=in_channel_we_want)

        self.averagepool = nn.AdaptiveAvgPool1d(1)
        self.OS_net = OS_CNN_block(layer_parameter_list, n_class, True)

    def forward(self, X):
        OS_block_result_list = []
        for i_th_channel, OS_block in enumerate(self.OS_block_list):
            OS_block_result = OS_block(X[:, i_th_channel:i_th_channel + 1, :])
            OS_block_result_list.append(OS_block_result)
        result = F.relu(torch.cat(tuple(OS_block_result_list), 1))

        result = self.OS_net(result)
        return result


class OS_CNN_easy_use():

    def __init__(self, Result_log_folder,
                 dataset_name,
                 device,
                 exp_tag,
                 Max_kernel_size=89,
                 paramenter_number_of_layer_list=[8 * 128, 5 * 128 * 256 + 2 * 256 * 128],
                 max_epoch=2000,
                 batch_size=16,
                 print_result_every_x_epoch=1,
                 start_kernel_size=1,
                 quarter_or_half=4,
                 num_steps=406,
                 input_dim=10,
                 num_classes=19,
                 fold=2
                 ):

        super(OS_CNN_easy_use, self).__init__()
        if dataset_name == 'pastis':
            if not os.path.exists(Result_log_folder + '/' + dataset_name + '/' + exp_tag + '/' + f'Fold_{fold}'):
                os.makedirs(Result_log_folder + '/' + dataset_name + '/' + exp_tag + '/' + f'Fold_{fold}')
            Initial_model_path = Result_log_folder + '/' + dataset_name + '/' + exp_tag + '/' + f'Fold_{fold}' + '/' + 'initial_model.pth.tar'
            model_save_path = Result_log_folder + '/' + dataset_name + '/' + exp_tag + '/' + f'Fold_{fold}' + '/' + 'model.pth.tar'
        else:
            if not os.path.exists(Result_log_folder + '/' + dataset_name + '/' + exp_tag):
                os.makedirs(Result_log_folder + '/' + dataset_name + '/' + exp_tag)
            Initial_model_path = Result_log_folder + '/' + dataset_name + '/' + exp_tag + '/' + 'initial_model.pth.tar'
            model_save_path = Result_log_folder + '/' + dataset_name + '/' + exp_tag + '/' + 'model.pth.tar'

        self.Result_log_folder = Result_log_folder
        self.dataset_name = dataset_name
        self.model_save_path = model_save_path
        self.Initial_model_path = Initial_model_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.Max_kernel_size = Max_kernel_size
        self.paramenter_number_of_layer_list = paramenter_number_of_layer_list
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.print_result_every_x_epoch = print_result_every_x_epoch
        self.quarter_or_half = quarter_or_half
        self.num_classes = num_classes
        self.num_steps = num_steps
        self.input_dim = input_dim
        self.OS_CNN = None

        self.start_kernel_size = start_kernel_size

    def initt(self):
        n_input_channel = self.input_dim
        input_shape = self.num_steps
        n_class = self.num_classes
        receptive_field_shape = min(int(input_shape / self.quarter_or_half), self.Max_kernel_size)

        # generate parameter list
        layer_parameter_list = generate_layer_parameter_list(self.start_kernel_size, receptive_field_shape,
                                                             self.paramenter_number_of_layer_list, in_channel=1)
        torch_OS_CNN = OS_CNN(layer_parameter_list, n_class, n_input_channel, True).to(self.device)
        self.OS_CNN = torch_OS_CNN

    def fit(self, train_loader, nb_device):
        n_input_channel = self.input_dim
        input_shape = self.num_steps
        n_class = self.num_classes
        receptive_field_shape = min(int(input_shape / self.quarter_or_half), self.Max_kernel_size)

        # generate parameter list
        layer_parameter_list = generate_layer_parameter_list(self.start_kernel_size, receptive_field_shape,
                                                             self.paramenter_number_of_layer_list, in_channel=1)

        torch_OS_CNN = OS_CNN(layer_parameter_list, n_class, n_input_channel, True).to(self.device)
        torch_OS_CNN = torch.nn.DataParallel(torch_OS_CNN.to(self.device), device_ids=range(nb_device))

        # save_initial_weight
        torch.save(torch_OS_CNN.state_dict(), self.Initial_model_path)

        # loss, optimizer, scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(torch_OS_CNN.parameters())
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, min_lr=0.000001)

        torch_OS_CNN.train()
        n_iter = 0
        pred, lab = [], []
        loss = 0
        for i in range(self.max_epoch):
            for j, sample in enumerate(train_loader):
                input_seq, mask, y = sample
                input_seq = input_seq.view(-1, self.num_steps, self.input_dim).to(torch.float32).permute(0, 2, 1).to(self.device)
                label = y.view(-1).long().to(self.device)
                sample = [input_seq, label]
                optimizer.zero_grad()
                y_predict = torch_OS_CNN(sample[0])
                output = criterion(y_predict, sample[1])
                loss += output.item()
                output.backward()
                optimizer.step()
                n_iter += 1
                preds = np.argmax(y_predict.detach().cpu().numpy(), axis=1)
                labs = label.detach().cpu().numpy()
                pred.append(preds)
                lab.append(labs)
                if n_iter % 100 == 0:
                    loss /= 100
                    scheduler.step(loss)
                    a = np.concatenate(pred, axis=0)
                    b = np.concatenate(lab, axis=0)
                    acc = accuracy_score(b, a)
                    print(f'[Iter {n_iter}]')
                    for param_group in optimizer.param_groups:
                        print('   lr = ', param_group['lr'])
                    print('   loss: {:.3f}'.format(loss))
                    print('   acc: {:.2f}%'.format(acc * 100))
                    pred, lab = [], []
                    loss = 0
                if optimizer.param_groups[0]['lr'] < 1e-5:
                    break

            if optimizer.param_groups[0]['lr'] < 1e-5:
                break

        torch.save(torch_OS_CNN.state_dict(), self.model_save_path)
        self.OS_CNN = torch_OS_CNN

    def predict(self, test_loader):
        self.OS_CNN.eval()
        with torch.no_grad():
            predict_list = np.array([])
            labs_list = np.array([])
            for i, sample in enumerate(test_loader):
                input_seq, mask, y = sample
                input_seq = input_seq.view(-1, self.num_steps, self.input_dim).to(torch.float32).permute(0, 2, 1).to(self.device)
                label = y.view(-1).long().to(self.device)
                sample = [input_seq, label]
                y_predict = self.OS_CNN(sample[0])
                y_predict = y_predict.detach().cpu().numpy()
                y_predict = np.argmax(y_predict, axis=1)
                labs_list = np.concatenate((labs_list, label.detach().cpu().numpy()), axis=0)
                predict_list = np.concatenate((predict_list, y_predict), axis=0)
                if i % 500 == 0:
                    print(i)
        return predict_list, labs_list
