# -*- coding:utf-8 -*-
import mxnet as mx
import torch
import numpy as np


def z_score(x, mean, std):
    '''
    Z-score normalization

    Parameters
    ----------
    x: np.ndarray

    mean: float

    std: float

    Returns
    ----------
    np.ndarray

    '''

    return (x - mean) / std


def z_inverse(x, mean, std):
    '''
    The inverse of function z_score()

    Parameters
    ----------
    x: np.ndarray

    mean: float

    std: float

    Returns
    ----------
    np.ndarray

    '''
    return x * std + mean


def filter_to_numpy(y_true, y_pred):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    elif isinstance(y_true, mx.nd.NDArray):
        y_true = y_true.asnumpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()
    elif isinstance(y_pred, mx.nd.NDArray):
        y_pred = y_pred.asnumpy()
    return y_true, y_pred


def mask_np(array, null_val):
    if np.isnan(null_val):
        return (~np.isnan(null_val)).astype('float32')
    else:
        return np.not_equal(array, null_val).astype('float32')


def masked_mape_np(y_true, y_pred, null_val=0):
    y_true, y_pred = filter_to_numpy(y_true, y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        mape = np.abs((y_pred - y_true) / y_true)
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def RMSE(y_true, y_pred):
    '''
    Mean squared error
    '''
    y_true, y_pred = filter_to_numpy(y_true, y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def MAE(y_true, y_pred):
    '''
    Mean absolute error
    '''
    y_true, y_pred = filter_to_numpy(y_true, y_pred)
    return np.mean(np.abs(y_true - y_pred))
