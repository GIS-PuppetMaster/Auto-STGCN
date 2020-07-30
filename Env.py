from time import time

import numpy as np
import gym
from gym import spaces
from mxnet import autograd

from utils.utils import *
from Model import Model
from mxnet.lr_scheduler import FactorScheduler
from layer_utils import *


class GNNEnv(gym.Env):
    def __init__(self, config, ctx, test=False):
        self.ctx = ctx
        self.config = config
        self.test = test
        # parse config
        self.epochs = config['epochs']
        self.epsilon = config['epsilon']
        self.num_of_vertices = config['num_of_vertices']
        self.adj_filename = config['adj_filename']
        self.id_filename = config['id_filename']
        self.time_series_filename = config['graph_signal_matrix_filename']
        self.pearsonr_adj_filename = config['pearsonr_adj_filename']
        self.time_limit = config['time_limit']
        self.dataset_name = os.path.split(self.adj_filename)[1].replace(".csv", "")
        self.n = config['n']
        # load data
        time_series_matrix = np.load(self.time_series_filename)['data'][:, :, 0]
        adj_SIPM1 = SIPM1(filepath=self.pearsonr_adj_filename, time_series_matrix=time_series_matrix,
                          num_of_vertices=self.num_of_vertices, epsilon=self.epsilon)
        adj_SIPM4 = get_adjacency_matrix(self.adj_filename, self.num_of_vertices, id_filename=self.id_filename)
        self.adj_SIPM = (adj_SIPM1, adj_SIPM4)

        # action_space = discrete(0,n+2) which will be mapped into discrete(-1,0,...,n,n+1(train_state)) as the def in the paper
        self.action_space = spaces.MultiDiscrete([4, 3, 4, self.n - 1])
        self.observation_space = spaces.Box(low=np.array([-1, -1, -1, -1, -1, -1]),
                                            high=np.array([self.n, 4, 3, 4, self.n - 1, 1]))

        self.action_trajectory = []
        self.current_state_phase = -1

        self.transformer = MinMaxTransformer()
        self.data = {}
        self.batch_size_option = [32, 50, 64]
        for batch_size in self.batch_size_option:
            loaders = []
            true_values = []
            for idx, (x, y) in enumerate(generate_data(self.time_series_filename, transformer=self.transformer)):
                y = y.squeeze(axis=-1)
                print(x.shape, y.shape)
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
        action = action.squeeze()
        assert action.shape == (4,)
        if self.current_state_phase == -1:
            # state_-1
            state = np.array([self.current_state_phase, -1, -1, -1, -1, 0])
            state.astype(np.float32)
            self.action_trajectory.append(action)
            self.current_state_phase += 1
            return state, None, False, {}
        elif self.current_state_phase < self.n - 1:
            # include state_0 and state_i if next state is not training state
            state = np.array([self.current_state_phase] + action.tolist() + [0])
            state.astype(np.float32)
            self.action_trajectory.append(action)
            self.current_state_phase += 1
            return state, None, False, {}
        elif self.current_state_phase == self.n - 1:
            # next state is training state
            state = np.array([self.current_state_phase] + action.tolist() + [1])
            state.astype(np.float32)
            self.action_trajectory.append(action)
            self.current_state_phase += 1
            return state, None, False, {}
        else:
            # training state(Terminal state)
            state = np.array([self.current_state_phase] + [-1, -1, -1, -1, -1])
            state.astype(np.float32)
            # don't append training hyper parameters action into action_trajectory
            # it's unnecessary and the model can't distinguish it from other action
            # self.action_trajectory.append(action)
            self.current_state_phase += 1
            # run model and get reward
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
            # train
            train_loader, val_loader, test_loader = self.data[batch_size]
            for epoch in range(self.config['epochs']):
                loss_value = 0
                train_num = 0
                for X in train_loader:
                    y = X.label[0]
                    X = X.data[0]
                    train_num += X.shape[0]
                    X, y = X.as_in_context(self.ctx), y.as_in_context(self.ctx)
                    with autograd.record():
                        y = y.astype('float32')
                        output = model(X)
                        l = loss(output, y).sum()
                    if self.test:
                        return
                    l.backward()
                    opt.step(batch_size)
                    loss_value += l.asscalar()
                train_loader.reset()
            # eval
            eval_loss_value = 0
            eval_loss_value_raw = 0
            eval_num = 0
            val_time = time()
            for X in val_loader:
                y = X.label[0]
                X = X.data[0]
                eval_num += X.shape[0]
                X, y = X.as_in_context(self.ctx), y.as_in_context(self.ctx)
                y = y.astype('float32')
                output = model(X)
                eval_loss_value_raw += loss(output, y).sum().asscalar()
                # 反标准化
                eval_y_min = self.transformer.y_data_set_min
                eval_y_max = self.transformer.y_data_set_max
                output = output * (eval_y_max - eval_y_min) + eval_y_min
                y = y * (self.transformer.y_transformer_info[1][1] - self.transformer.y_transformer_info[1][
                    0]) + \
                    self.transformer.y_transformer_info[1][0]
                eval_loss_value += loss(output, y).sum().asscalar()
            val_time = time() - val_time
            val_loader.reset()
            # get reward
            reward = eval_loss_value * np.power(val_time / self.time_limit, -0.07)
            return state, reward, True, {}

    def reset(self):
        self.action_trajectory = []
        self.current_state_phase = -1

    def render(self, mode='human'):
        pass
