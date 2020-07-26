# -*- coding:utf-8 -*-

import os
import numpy as np
import mxnet as mx


class NoramlTransformer:
    def __init__(self):
        self.x_transformer_info = []
        self.y_transformer_info = []

    def __call__(self, *args, **kwargs):
        if 'x' in kwargs.keys():
            data = kwargs['x']
            mean = data.mean()
            std = data.std()
            self.x_transformer_info.append((mean, std))
            return (data-mean)/std
        elif 'y' in kwargs.keys():
            data = kwargs['y']
            mean = data.mean()
            std = data.std()
            self.y_transformer_info.append((mean, std))
            return (data-mean)/std
        else:
            raise Exception("Error transformer kwargs.key:{},which should be 'x' or 'y'".format(str(kwargs.keys())))

class MinMaxTransformer:
    def __init__(self):
        self.x_transformer_info = []
        self.y_transformer_info = []
        self.x_data_set_min = 0
        self.x_data_set_max = 0
        self.y_data_set_min = 0
        self.y_data_set_max = 0

    def set_data_set_info(self, x_min, x_max, y_min, y_max):
        self.x_data_set_min = x_min
        self.x_data_set_max = x_max
        self.y_data_set_min = y_min
        self.y_data_set_max = y_max

    def __call__(self, *args, **kwargs):
        if 'x' in kwargs.keys():
            data = kwargs['x']
            min = data.min()
            max = data.max()
            self.x_transformer_info.append((min,max))
            return (data - min) / (max - min)
        elif 'y' in kwargs.keys():
            data = kwargs['y']
            min = data.min()
            max = data.max()
            self.y_transformer_info.append((min, max))
            return (data - min) / (max - min)
        else:
            raise Exception("Error transformer kwargs.key:{},which should be 'x' or 'y'".format(str(kwargs.keys())))
def get_adjacency_matrix(distance_df_filename, num_of_vertices,
                         type_='connectivity', id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    type_: str, {connectivity, distance}

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    import csv

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx
                       for idx, i in enumerate(f.read().strip().split('\n'))}
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                A[id_dict[j], id_dict[i]] = 1
        return A

    # Fills cells in the matrix with distances.
    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if type_ == 'connectivity':
                A[i, j] = 1
                A[j, i] = 1
            elif type == 'distance':
                A[i, j] = 1 / distance
                A[j, i] = 1 / distance
            else:
                raise ValueError("type_ error, must be "
                                 "connectivity or distance!")
    return A


def construct_adj(A, steps):
    '''
    construct a bigger adjacency matrix using the given matrix

    Parameters
    ----------
    A: np.ndarray, adjacency matrix, shape is (N, N)

    steps: how many times of the does the new adj mx bigger than A

    Returns
    ----------
    new adjacency matrix: csr_matrix, shape is (N * steps, N * steps)
    '''
    N = len(A)
    adj = np.zeros([N * steps] * 2)

    for i in range(steps):
        adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A

    for i in range(N):
        for k in range(steps - 1):
            adj[k * N + i, (k + 1) * N + i] = 1
            adj[(k + 1) * N + i, k * N + i] = 1

    for i in range(len(adj)):
        adj[i, i] = 1

    return adj


def generate_from_train_val_test(data, transformer):
    mean = None
    std = None
    for key in ('train', 'val', 'test'):
        x, y = generate_seq(data[key], 12, 12)
        if transformer:
            x = transformer(x=x)
            y = transformer(y=y)
            yield x, y
        else:
            if mean is None:
                mean = x.mean()
            if std is None:
                std = x.std()
            yield (x - mean) / std, y


def generate_from_data(data, length, transformer):
    mean = None
    std = None
    train_line, val_line = int(length * 0.6), int(length * 0.8)
    for line1, line2 in ((0, train_line),
                         (train_line, val_line),
                         (val_line, length)):
        x, y = generate_seq(data['data'][line1: line2], 12, 12)
        if transformer:
            x = transformer(x=x)
            y = transformer(y=y)
            yield x, y
        else:
            if mean is None:
                mean = x.mean()
            if std is None:
                std = x.std()
            yield (x - mean) / std, y


def generate_data(graph_signal_matrix_filename, transformer=None):
    '''
    shape is (num_of_samples, 12, num_of_vertices, 1)
    '''
    data = np.load(graph_signal_matrix_filename)
    keys = data.keys()
    if 'train' in keys and 'val' in keys and 'test' in keys:
        for i in generate_from_train_val_test(data, transformer):
            yield i
    elif 'data' in keys:
        length = data['data'].shape[0]
        for i in generate_from_data(data, length, transformer):
            yield i
    else:
        raise KeyError("neither data nor train, val, test is in the data")


def generate_seq(data, train_length, pred_length):
    seq = np.concatenate([np.expand_dims(
        data[i: i + train_length + pred_length], 0)
        for i in range(data.shape[0] - train_length - pred_length + 1)],
        axis=0)[:, :, :, 0: 1]
    return np.split(seq, 2, axis=1)


def mask_np(array, null_val):
    if np.isnan(null_val):
        return (~np.isnan(null_val)).astype('float32')
    else:
        return np.not_equal(array, null_val).astype('float32')


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        mape = np.abs((y_pred - y_true) / y_true)
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def masked_mse_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mse = (y_true - y_pred) ** 2
    return np.mean(np.nan_to_num(mask * mse))


def masked_mae_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mae = np.abs(y_true - y_pred)
    return np.mean(np.nan_to_num(mask * mae))
