import argparse
import json
from time import time

from gym import spaces
import os
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


class TrainEnv(GNNEnv):
    def __init__(self, config, ctx, logger, test=False):
        self.ctx = ctx
        self.config = config
        self.test = test
        self.logger = logger
        # parse config
        self.epochs = config['epochs']
        self.phi = config['phi']
        self.num_of_vertices = config['num_of_vertices']
        self.adj_filename = config['adj_filename']
        self.id_filename = config['id_filename']
        self.time_series_filename = config['graph_signal_matrix_filename']
        self.pearsonr_adj_filename = config['pearsonr_adj_filename']
        self.time_max = config['time_max']
        self.n = config['n']
        # load data
        self.dataset_name = os.path.split(self.adj_filename)[1].replace(".csv", "")
        time_series_matrix = np.load(self.time_series_filename)['data'][:, :, 0]
        adj_SIPM1 = SIPM1(filepath=self.pearsonr_adj_filename, time_series_matrix=time_series_matrix,
                          num_of_vertices=self.num_of_vertices, phi=self.phi)
        adj_SIPM4 = get_adjacency_matrix(self.adj_filename, self.num_of_vertices, id_filename=self.id_filename)
        self.adj_SIPM = (adj_SIPM1, adj_SIPM4)

        # action_space = discrete(0,n+2) which will be mapped into discrete(-1,0,...,n,n+1(train_state)) as the def in the paper
        self.action_space = spaces.MultiDiscrete([4, 3, 4, self.n - 1, 1])
        self.observation_space = spaces.Box(low=np.array([-2, -1, -1, -1, -1]),
                                            high=np.array([self.n, 4, 3, 4, self.n - 1]))

        # doesn't contains training stage action
        self.action_trajectory = []
        self.actions = []
        self.state_trajectory = []
        self.current_state_phase = -1
        self.training_stage = False
        self.training_stage_action = None

        self.data = {}
        self.batch_size_option = [32, 50, 64]
        self.transformer = {}
        self.train_set_sample_num = 0
        self.eval_set_sample_num = 0
        self.test_set_sample_num = 0
        for batch_size in self.batch_size_option:
            loaders = []
            true_values = []
            for idx, (x, y) in enumerate(generate_data(self.time_series_filename)):
                if idx == 0:
                    self.train_set_sample_num = x.shape[0]
                elif idx == 1:
                    self.eval_set_sample_num = x.shape[0]
                else:
                    self.test_set_sample_num = x.shape[0]
                y = y.squeeze(axis=-1)
                print(x.shape, y.shape)
                self.logger.append_log_file(str((x.shape, y.shape)))
                loaders.append(
                    mx.io.NDArrayIter(
                        x, y,
                        batch_size=batch_size,
                        shuffle=(idx == 0),
                        label_name='label'
                    )
                )
                if idx == 0:
                    self.training_samples = x.shape[0]
                else:
                    true_values.append(y)
                self.data[batch_size] = loaders

    def train_model(self, actions: list):
        # remove [-1,-1,-1,-1]
        for idx in range(len(actions)):
            if actions[idx] == [-1, -1, -1, -1]:
                actions.pop(idx)
        # fetch training_stage_action and remove it from model structure action
        action = actions[0]
        actions.pop(0)
        self.action_trajectory = actions
        # action belongs to stage4: Training stage
        if action[0] == 1:
            # LF1
            loss = mx.gluon.loss.L2Loss()
        else:
            loss = mx.gluon.loss.HuberLoss()

        # must set batch_size before init model
        batch_size = self.batch_size_option[action[1] - 1]
        # transformer = self.transformer[batch_size]
        self.config['batch_size'] = batch_size
        model = Model(self.action_trajectory, self.config, self.ctx, self.adj_SIPM)
        model.initialize(ctx=self.ctx)
        lr_option = [1e-3, 7e-4, 1e-4]
        opt_option = ['rmsprop', 'adam', 'adam']
        lr = lr_option[action[2] - 1]
        if action[3] == 1:
            step = self.epochs / 10
            if step < 1:
                step = 1
            lr_scheduler = FactorScheduler(step, factor=0.7, base_lr=lr)
            opt = mx.gluon.Trainer(model.collect_params(), opt_option[action[3] - 1],
                                   {'lr_scheduler': lr_scheduler})
        elif action[3] == 2:
            opt = mx.gluon.Trainer(model.collect_params(), opt_option[action[3] - 1], {'learning_rate': lr})
        else:
            global_train_steps = self.training_samples // batch_size + 1
            max_update_factor = 1
            lr_sch = mx.lr_scheduler.PolyScheduler(
                max_update=global_train_steps * self.epochs * max_update_factor,
                base_lr=lr,
                pwr=2,
                warmup_steps=global_train_steps
            )
            opt = mx.gluon.Trainer(model.collect_params(), opt_option[action[3] - 1], {'lr_scheduler': lr_sch})
        # train
        train_loader, val_loader, test_loader = self.data[batch_size]
        model_structure = deepcopy(self.action_trajectory)
        model_structure.append(action)
        best_mae = float('inf')
        best_epoch = 0
        for epoch in range(config['epochs']):
            self.logger.set_episode(epoch)
            loss_value = 0
            mae = 0
            rmse = 0
            mape = 0
            train_batch_num = 0
            train_time = 0.
            for X in train_loader:
                y = X.label[0]
                X = X.data[0]
                train_batch_num += 1
                X, y = X.as_in_context(self.ctx), y.as_in_context(self.ctx)
                with autograd.record():
                    y = y.astype('float32')
                    start_time = time()
                    output = model(X)
                    train_time += time() - start_time
                    l = loss(output, y)
                if self.test:
                    return
                l.backward()
                opt.step(batch_size)
                # loss_value_raw += l.mean().asscalar()
                loss_value += loss(output, y).mean().asscalar()
                mae += MAE(y, output)
                rmse += RMSE(y, output)
                mape += masked_mape_np(y, output)
            train_loader.reset()
            # loss_value_raw /= train_batch_num
            loss_value /= train_batch_num
            mae /= train_batch_num
            rmse /= train_batch_num
            mape /= train_batch_num
            train_time = (time() - train_time) / self.train_set_sample_num
            self.logger(
                train=[epoch, loss_value, mae, mape, rmse, train_time])
            print(f"    epoch:{epoch}  ,loss:{loss_value}, MAE:{mae}, MAPE:{mape}, RMSE:{rmse}, time:{train_time}")
            # eval
            eval_loss_value = 0
            val_time = 0.
            eval_batch_num = 0
            mae = 0
            rmse = 0
            mape = 0
            val_time = time()
            for X in val_loader:
                y = X.label[0]
                X = X.data[0]
                eval_batch_num += 1
                X, y = X.as_in_context(self.ctx), y.as_in_context(self.ctx)
                y = y.astype('float32')
                start_time = time()
                output = model(X)
                val_time += time() - start_time
                # eval_loss_value_raw += loss(output, y).mean().asscalar()
                eval_loss_value += loss(output, y).mean().asscalar()
                mae += MAE(y, output)
                rmse += RMSE(y, output)
                mape += masked_mape_np(y, output)
            eval_loss_value /= eval_batch_num
            mae /= eval_batch_num
            rmse /= eval_batch_num
            mape /= eval_batch_num
            print(f"    eval_result: loss:{eval_loss_value}, MAE:{mae}, MAPE:{mape}, RMSE:{rmse}, time:{val_time}")
            val_loader.reset()
            self.logger(eval=[eval_loss_value, mae, mape, rmse, val_time])
            self.logger.save_GNN(model, model_structure, mae)
            self.logger.update_data_units()
            self.logger.flush_log()
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
            if epoch - best_epoch > 10:
                print(f'early stop at epoch:{epoch}')
                break
        # test
        # load best eval metric model parameters
        model.load_params(os.path.join(os.path.join(self.logger.log_path, 'GNN'), 'best_GNN_model.params'),
                          ctx=self.ctx)
        test_loss_value = 0
        test_batch_num = 0
        mae = 0
        rmse = 0
        mape = 0
        test_time = 0.
        for X in test_loader:
            y = X.label[0]
            X = X.data[0]
            test_batch_num += 1
            X, y = X.as_in_context(self.ctx), y.as_in_context(self.ctx)
            y = y.astype('float32')
            start_time = time()
            output = model(X)
            test_time += time() - start_time
            # test_loss_value_raw += loss(output, y).mean().asscalar()
            test_loss_value += loss(output, y).mean().asscalar()
            mae += MAE(y, output)
            rmse += RMSE(y, output)
            mape += masked_mape_np(y, output)
        test_loss_value /= test_batch_num
        mae /= test_batch_num
        rmse /= test_batch_num
        mape /= test_batch_num
        test_loader.reset()
        print(f"    test_result: loss:{test_loss_value}, MAE:{mae}, MAPE:{mape}, RMSE:{rmse}, TIME:{test_time}")
        self.logger(test=[test_loss_value, mae, mape, rmse, test_time])
        self.logger.update_data_units()
        self.logger.flush_log()
        return [mae, mape, rmse, test_time]

    def test_model(self, actions: list):
        # remove [-1,-1,-1,-1]
        for idx in range(len(actions)):
            if actions[idx] == [-1, -1, -1, -1]:
                actions.pop(idx)
        # fetch training_stage_action and remove it from model structure action
        action = actions[0]
        actions.pop(0)
        self.action_trajectory = actions
        # action belongs to stage4: Training stage
        if action[0] == 1:
            # LF1
            loss = mx.gluon.loss.L2Loss()
        else:
            loss = mx.gluon.loss.HuberLoss()

        # must set batch_size before init model
        batch_size = self.batch_size_option[action[1] - 1]
        # transformer = self.transformer[batch_size]
        self.config['batch_size'] = batch_size
        model = Model(self.action_trajectory, self.config, self.ctx, self.adj_SIPM)
        model.initialize(ctx=self.ctx)
        lr_option = [1e-3, 7e-4, 1e-4]
        opt_option = ['rmsprop', 'adam', 'adam']
        lr = lr_option[action[2] - 1]
        if action[3] == 1:
            step = self.epochs / 10
            if step < 1:
                step = 1
            lr_scheduler = FactorScheduler(step, factor=0.7, base_lr=lr)
            opt = mx.gluon.Trainer(model.collect_params(), opt_option[action[3] - 1],
                                   {'lr_scheduler': lr_scheduler})
        elif action[3] == 2:
            opt = mx.gluon.Trainer(model.collect_params(), opt_option[action[3] - 1], {'learning_rate': lr})
        else:
            global_train_steps = self.training_samples // batch_size + 1
            max_update_factor = 1
            lr_sch = mx.lr_scheduler.PolyScheduler(
                max_update=global_train_steps * self.epochs * max_update_factor,
                base_lr=lr,
                pwr=2,
                warmup_steps=global_train_steps
            )
            opt = mx.gluon.Trainer(model.collect_params(), opt_option[action[3] - 1], {'lr_scheduler': lr_sch})
        # train
        train_loader, val_loader, test_loader = self.data[batch_size]
        # test
        # load best eval metric model parameters
        model.load_params(f'./Log/{self.dataset_name.upper()}_experiment2_qlearning_2/GNN/best_GNN_model.params',
                          ctx=self.ctx)
        test_loss_value = 0
        test_batch_num = 0
        mae = 0
        rmse = 0
        mape = 0
        test_time = 0.
        for X in test_loader:
            y = X.label[0]
            X = X.data[0]
            test_batch_num += 1
            X, y = X.as_in_context(self.ctx), y.as_in_context(self.ctx)
            y = y.astype('float32')
            start_time = time()
            output = model(X)
            test_time += time() - start_time
            # test_loss_value_raw += loss(output, y).mean().asscalar()
            test_loss_value += loss(output, y).mean().asscalar()
            mae += MAE(y, output)
            rmse += RMSE(y, output)
            mape += masked_mape_np(y, output)
        test_loss_value /= test_batch_num
        mae /= test_batch_num
        rmse /= test_batch_num
        mape /= test_batch_num
        test_loader.reset()
        print(f"    test_result: loss:{test_loss_value}, MAE:{mae}, MAPE:{mape}, RMSE:{rmse}, TIME:{test_time}")
        self.logger(test=[test_loss_value, mae, mape, rmse, test_time])
        self.logger.update_data_units()
        self.logger.flush_log()
        return [mae, mape, rmse, test_time]


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
    if args.load is not None:
        model_filename = args.model
    else:
        model_filename = './Config/qlearning_2.json'
        print(f'using model {model_filename}')
    with open(model_filename, 'r') as f:
        actions = json.loads(f.read())
    config_filename = './Config/default.json'
    with open(config_filename, 'r') as f:
        config = json.loads(f.read())
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
    config["adj_filename"] = f"data/{dataset}/{dataset}.csv"
    config["graph_signal_matrix_filename"] = f"data/{dataset}/{dataset}.npz"
    config["pearsonr_adj_filename"] = f"data/{dataset}/{dataset}_pearsonr.npz"
    arg_dict = copy(vars(args))
    for key, value in vars(args).items():
        if value is None:
            arg_dict.pop(key)
    if args.load is not None:
        config.update(arg_dict)
    else:
        print('because load is not set, using default config')

    print(json.dumps(config, sort_keys=True, indent=4))
    if isinstance(config['ctx'], list):
        ctx = [mx.gpu(i) for i in config['ctx']]
    elif isinstance(config['ctx'], int):
        ctx = mx.gpu(config['ctx'])
    else:
        raise Exception("config_ctx error:" + str(config['ctx']))
    config_name = config_filename.replace('./Config/', '').replace("/", "_").split('.')[0] + '_' + \
                  model_filename.replace('./Config/', '').replace("/", "_").split('.')[0]
    if args.load is None:
        config_name += '_test'
    logger = Logger(config_name, config, False, larger_better=False)
    res = []
    for i in range(config['times']):
        env = TrainEnv(config, ctx, logger)
        if args.load is not None:
            res.append(env.train_model(deepcopy(actions)))
        else:
            res.append(env.test_model(deepcopy(actions)))
        logger.append_log_file(f'res:{res}')
    res = np.array(res)
    print('test set metric: MAE, MAPE, RMSE, TIME')
    logger.append_log_file(f'mean:{res.mean(axis=0)}')
    print(f'mean:{res.mean(axis=0)}')
    logger.append_log_file(f'std:{res.std(axis=0)}')
    print(f'std:{res.std(axis=0)}')
