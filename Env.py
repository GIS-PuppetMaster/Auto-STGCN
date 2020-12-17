from time import time
import sys
from utils.math_utils import MAE, RMSE, masked_mape_np
import gym
from gym import spaces
from mxnet import autograd
import traceback
from utils.utils import *
from Model import Model
from mxnet.lr_scheduler import FactorScheduler
from utils.layer_utils import *
import wandb
from copy import deepcopy
import dill
import os

class GNNEnv(gym.Env):
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
        self.train_length = config['train_length']
        self.pred_length = config['pred_length']
        self.split_ratio = config['split_ratio']
        self.mode = config['mode']
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
            for idx, (x, y) in enumerate(
                    generate_data(self.time_series_filename, self.train_length, self.pred_length, self.split_ratio)):
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

    def step(self, action):
        if isinstance(action, list):
            action = np.array(action)
        action = action.squeeze()
        self.actions.append(action.tolist())
        # end ST-block, need training
        if self.current_state_phase <= self.n - 1 and not (
                self.current_state_phase > 0 and (action == np.array([-1, -1, -1, -1])).all()):
            # state{-2}
            # state{-1}
            #    set training stage
            #    collect parameter and apply at the last step
            # state{0}...{n-1}
            state = np.array([self.current_state_phase] + action.tolist())
            state.astype(np.float32)
            if self.current_state_phase == -1:
                # training stage
                self.training_stage_action = action
            else:
                self.action_trajectory.append(action)
            self.current_state_phase += 1
            if self.mode == 'search':
                return state, None, False, {"exception_flag": False}
            else:
                return None
        else:
            # set the last ST-Block and start training
            # return terminal state
            state = np.array([self.current_state_phase, -1, -1, -1, -1])
            state.astype(np.float32)
            if not (action == np.array([-1, -1, -1, -1])).all():
                self.action_trajectory.append(action)
            self.current_state_phase += 1
            if self.mode == 'search':
                # 输入的training_stage_action不包括[-1,-1,-1,-1]和training stage
                reward, flag = self.train_model(self.training_stage_action)
                return state, reward, True, {"exception_flag": flag}
            else:
                return self.train_model(self.training_stage_action)

    def reset(self):
        self.action_trajectory = []
        self.actions = []
        self.state_trajectory = []
        self.current_state_phase = -1
        self.training_stage = False
        self.training_stage_action = None
        return np.array([-2, -1, -1, -1, -1])

    def train_model(self, action):
        # action belongs to stage4: Training stage
        if action[0] == 1:
            # LF1
            loss = mx.gluon.loss.L2Loss()
        else:
            loss = mx.gluon.loss.HuberLoss()
        # must set batch_size before init model
        batch_size = self.batch_size_option[action[1] - 1]
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
        self.logger(action=self.actions)
        model_structure = deepcopy(self.actions)
        try:
            train_loader, val_loader, test_loader = self.data[batch_size]
            if self.mode == 'search' or self.mode == 'train':
                # train
                train_time = 0.
                best_mae = float('inf')
                best_epoch = 0
                best_test_mae = float('inf')
                best_test_res = None
                for epoch in range(self.config['epochs']):
                    loss_value = 0
                    mae = 0
                    rmse = 0
                    mape = 0
                    train_batch_num = 0
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
                        # if self.test:
                        #     return
                        l.backward()
                        opt.step(batch_size)
                        loss_value += loss(output, y).mean().asscalar()
                        mae += MAE(y, output)
                        rmse += RMSE(y, output)
                        mape += masked_mape_np(y, output)
                    train_loader.reset()
                    loss_value /= train_batch_num
                    mae /= train_batch_num
                    rmse /= train_batch_num
                    mape /= train_batch_num
                    self.logger(
                        train=[epoch, loss_value, mae, mape, rmse, train_time])
                    print(f"    epoch:{epoch} ,loss:{loss_value}")
                    if self.mode == 'train':
                        eval_loss_value, mae, rmse, mape, val_time = self.eval_model(val_loader, model, loss)
                        self.logger(eval=[eval_loss_value, mae, mape, rmse, val_time])
                        self.logger.save_GNN(model, model_structure, mae)
                        if mae < best_mae:
                            best_mae = mae
                            best_epoch = epoch
                        if epoch - best_epoch > 10:
                            print(f'early stop at epoch:{epoch}')
                            break
                        mae, mape, rmse, test_time = self.test_model_without_load(test_loader, model, loss)
                        if mae < best_test_mae:
                            best_test_mae = mae
                            best_test_res = [mae, mape, rmse, test_time]
                        print(f'test_res:{best_test_res}')
            if self.mode == 'search':
                eval_loss_value, mae, rmse, mape, val_time = self.eval_model(val_loader, model, loss)
                # get reward
                if self.time_max - val_time > 0:
                    reward = -mae / 10 + np.power(np.e, -5) * np.log2(self.time_max - val_time)
                else:
                    reward = -10
                if np.isnan(reward) or np.isinf(reward) or reward < -100:
                    self.logger.append_log_file(f"Warning: reward={reward}")
                    reward = -10
                self.logger(eval=[eval_loss_value, mae, mape, rmse, val_time])
                self.logger.save_GNN(model, model_structure, reward / len(self.action_trajectory) + 1)
                return reward, False
            elif self.mode == 'train':
                self.logger.append_log_file(f'best_test_res:{best_test_res}')
                mae, mape, rmse, test_time = self.test_model(test_loader, loss)
                return best_test_res, [mae, mape, rmse, test_time]
            elif self.mode == 'test':
                mae, mape, rmse, test_time = self.test_model(test_loader, loss)
                return None, [mae, mape, rmse, test_time]

        except Exception as e:
            self.logger.append_log_file(e.args[0])
            self.logger(train=None, eval=None, test=None)
            traceback.print_exc()
            return -10, True

    def eval_model(self, val_loader, model, loss):
        val_loader.reset()
        eval_loss_value = 0
        eval_batch_num = 0
        mae = 0
        rmse = 0
        mape = 0
        val_time = 0.
        for X in val_loader:
            y = X.label[0]
            X = X.data[0]
            eval_batch_num += 1
            X, y = X.as_in_context(self.ctx), y.as_in_context(self.ctx)
            y = y.astype('float32')
            start_time = time()
            output = model(X)
            val_time += time() - start_time
            eval_loss_value += loss(output, y).mean().asscalar()
            mae += MAE(y, output)
            rmse += RMSE(y, output)
            mape += masked_mape_np(y, output)
        eval_loss_value /= eval_batch_num
        mae /= eval_batch_num
        rmse /= eval_batch_num
        mape /= eval_batch_num
        print(f"    eval_result: loss:{eval_loss_value}, MAE:{mae}, MAPE:{mape}, RMSE:{rmse}, TIME:{val_time}")
        return eval_loss_value, mae, rmse, mape, val_time

    def test_model(self, test_loader, loss):
        # load best eval metric model or the model described in args.load
        if 'load' in self.config.keys() and self.config['load'] is not None:
            with open(self.config['load'],'rb') as f:
                model = dill.load(f)
            # model.load_parameters(self.config['load'], ctx=self.ctx)
        else:
            with open(self.logger.log_path + f"GNN/best_GNN_model.params",'rb') as f:
                model = dill.load(f)
            # model.load_parameters(self.logger.log_path + f"GNN/best_GNN_model.params", ctx=self.ctx)
        return self.test_model_without_load(test_loader, model, loss)

    def test_model_without_load(self, test_loader, model, loss):
        # test model
        test_loader.reset()
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
        print(f"    test_result: loss:{test_loss_value}, MAE:{mae}, MAPE:{mape}, RMSE:{rmse}, TIME:{test_time}")
        self.logger(test=[test_loss_value, mae, mape, rmse, test_time])
        self.logger.update_data_units()
        self.logger.flush_log()
        return mae, mape, rmse, test_time

    def render(self, mode='human'):
        pass
