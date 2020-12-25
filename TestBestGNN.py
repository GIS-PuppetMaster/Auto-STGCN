import argparse
import json
from time import time

import shutil
from gym import spaces
import os

from EVO_STGCN import is_debug
from utils.math_utils import MAE, RMSE, masked_mape_np
from mxnet import autograd
import traceback
from copy import deepcopy
from mxnet.lr_scheduler import FactorScheduler
import mxnet as mx
from ExperimentDataLogger import *
from Model import Model
from Env import GNNEnv
from utils.utils import generate_data
from utils.layer_utils import *
from copy import deepcopy, copy
import yagmail

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--train_length', type=int, default=None)
    parser.add_argument('--pred_length', type=int, default=None)
    parser.add_argument('--split_ratio', type=list, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--times', type=int, default=None)
    parser.add_argument('--ctx', type=int, default=None)
    args = parser.parse_args()
    if args.model is not None:
        model_filename = args.model
    else:
        model_filename = './Config/qlearning_2.json'
        print(f'using model {model_filename}')

    with open(model_filename, 'r') as f:
        actions = json.loads(f.read())
    config_filename = './Config/default.json'
    with open(config_filename, 'r') as f:
        config = json.loads(f.read())
    if args.load is None:
        config['epochs'] = 50
    # override default config
    dataset = args.data.upper()
    if dataset == 'PEMS03':
        config["id_filename"] = "data/PEMS03/PEMS03.txt"
        config["num_of_vertices"] = 358
    elif dataset == 'PEMS04':
        config["id_filename"] = None
        config["num_of_vertices"] = 307
    elif dataset == 'PEMS07':
        config["id_filename"] = None
        config["num_of_vertices"] = 883
    elif dataset == 'PEMS08':
        config["id_filename"] = None
        config["num_of_vertices"] = 170
    else:
        raise Exception(f'Input data is {args.data}, only support PEMS03/04/07/08')
    if args.load is not None:
        config["mode"] = 'test'
    else:
        config["mode"] = 'train'
    config["adj_filename"] = f"data/{dataset}/{dataset}.csv"
    config["graph_signal_matrix_filename"] = f"data/{dataset}/{dataset}.npz"
    config["pearsonr_adj_filename"] = f"data/{dataset}/{dataset}_pearsonr.npz"
    arg_dict = copy(vars(args))
    for key, value in vars(args).items():
        if value is None:
            arg_dict.pop(key)
    config.update(arg_dict)

    print(json.dumps(config, sort_keys=True, indent=4))
    if isinstance(config['ctx'], list):
        ctx = [mx.gpu(i) for i in config['ctx']]
    elif isinstance(config['ctx'], int):
        ctx = mx.gpu(config['ctx'])
    else:
        raise Exception("config_ctx error:" + str(config['ctx']))
    if is_debug():
        log_name = 'debug'
        if os.path.exists('Log/debug_retrain/'):
            shutil.rmtree('Log/debug_retrain/')
    else:
        log_name = input('log_name:\n')
    if args.load is not None:
        log_name += '_test'
    logger = Logger(log_name, config, False, larger_better=False)
    res = []
    res_in_train = []
    for i in range(config['times']):
        logger.reset_metric()
        env = GNNEnv(config, ctx, logger)
        for action in actions:
            ret = env.step(action)
            if ret is not None:
                res.append(ret[1])
                res_in_train.append(ret[0])
        # if args.load is None:
        #     res.append(env.train_model(deepcopy(actions)))
        # else:
        #     res.append(env.test_model(deepcopy(actions)))
        logger.append_log_file(f'res:{res}')
        logger.append_log_file(f'res_in_train:{res_in_train}')
    res = np.array(res)
    print('test set metric: MAE, MAPE, RMSE, TIME')
    logger.append_log_file(f'test set mean:{res.mean(axis=0)}')
    logger.append_log_file(f'test set std:{res.std(axis=0)}')

    res_in_train = np.array(res_in_train)
    print('test in train set metric: MAE, MAPE, RMSE, TIME')
    logger.append_log_file(f'test in train mean:{res_in_train.mean(axis=0)}')
    logger.append_log_file(f'test in train std:{res_in_train.std(axis=0)}')

    with open('./Config/mail.json', 'r') as f:
        mail = json.load(f)
    yag = yagmail.SMTP(user=mail['user'], password=mail['password'], host='smtp.qq.com')
    yag.send(to=mail['user'], subject=f'Experiment {logger.log_name} is finished',
                 contents=[f'test set mean:{res.mean(axis=0)}', f'test set std:{res.std(axis=0)}', f'test in train mean:{res_in_train.mean(axis=0)}', f'test in train std:{res_in_train.std(axis=0)}'])
