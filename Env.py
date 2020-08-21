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


class GNNEnv(gym.Env):
    def __init__(self, config, ctx, logger, test=False):
        self.ctx = ctx
        self.config = config
        self.test = test
        self.logger = logger
        # parse config
        self.epochs = config['epochs']
        self.epsilon = config['epsilon']
        self.num_of_vertices = config['num_of_vertices']
        self.adj_filename = config['adj_filename']
        self.id_filename = config['id_filename']
        self.time_series_filename = config['graph_signal_matrix_filename']
        self.pearsonr_adj_filename = config['pearsonr_adj_filename']
        self.max_time = config['max_time']
        self.n = config['n']
        self.training_stage_last = config['training_stage_last']
        # load data
        self.dataset_name = os.path.split(self.adj_filename)[1].replace(".csv", "")
        time_series_matrix = np.load(self.time_series_filename)['data'][:, :, 0]
        adj_SIPM1 = SIPM1(filepath=self.pearsonr_adj_filename, time_series_matrix=time_series_matrix,
                          num_of_vertices=self.num_of_vertices, epsilon=self.epsilon)
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
        if self.training_stage_last:
            self.current_state_phase = 0
        else:
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
            transformer = MinMaxTransformer()
            self.transformer[batch_size] = transformer
            for idx, (x, y) in enumerate(generate_data(self.time_series_filename, transformer=transformer)):
                if idx == 0:
                    self.train_set_sample_num = x.shape[0]
                elif idx == 1:
                    self.eval_set_sample_num = x.shape[0]
                else:
                    self.test_set_sample_num = x.shape[0]
                y = y.squeeze(axis=-1)
                print(x.shape, y.shape)
                self.logger.append_log_file(str((x.shape,y.shape)))
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
        self.actions.append(action.tolist())
        if self.training_stage_last:
            if not self.training_stage:
                if self.current_state_phase == 0:
                    # state{0}
                    state = np.array([self.current_state_phase] + action[:-1].tolist())
                    state.astype(np.float32)
                    self.action_trajectory.append(action)
                    self.current_state_phase += 1
                    self.state_trajectory.append(state.tolist())
                    return np.array(self.state_trajectory), None, False, {"exception_flag": False}
                elif self.current_state_phase <= self.n:
                    # include state{1} and state{i} if next state is not training state
                    if (action == np.array([-1, -1, -1, -1, -1])).all() or self.current_state_phase == self.n:
                        # next state is training state
                        # terminal state
                        state = np.array([self.current_state_phase, -1, -1, -1, -1])
                        self.action_trajectory.append(action)
                        self.current_state_phase += 1
                        self.training_stage = True
                        self.state_trajectory.append(state.tolist())
                        return np.array(self.state_trajectory), None, False, {"exception_flag": False}
                    else:
                        # state{1}..{n-1} and self.training_stage = False
                        # next state is not training state
                        # ST-block
                        state = np.array([self.current_state_phase] + action[:-1].tolist())
                        state.astype(np.float32)
                        self.action_trajectory.append(action)
                        self.current_state_phase += 1
                        self.state_trajectory.append(state.tolist())
                        return np.array(self.state_trajectory), None, False, {"exception_flag": False}
            else:
                # training state(Terminal state)
                # state{n+1}
                state = np.array([self.current_state_phase] + [-1, -1, -1, -1])
                state.astype(np.float32)
                # don't append training hyper parameters action into action_trajectory
                # it's unnecessary and the model can't distinguish it from other action
                # self.action_trajectory.append(action)
                self.current_state_phase += 1
                # run model and get reward
                reward, flag = self.train_model(action)
                self.state_trajectory.append(state.tolist())
                return np.array(self.state_trajectory), reward, True, {"exception_flag": flag}
        else:
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
                    # training state
                    self.training_stage_action = action
                else:
                    self.action_trajectory.append(action)
                self.current_state_phase += 1
                return state, None, False, {"exception_flag": False}
            else:
                # set the last ST-Block and start training
                # return terminal state
                state = np.array([self.current_state_phase, -1, -1, -1, -1])
                state.astype(np.float32)
                if not (action == np.array([-1, -1, -1, -1])).all():
                    self.action_trajectory.append(action)
                self.current_state_phase += 1
                reward, flag = self.train_model(self.training_stage_action)
                return state, reward, True, {"exception_flag": flag}

    def reset(self):
        self.action_trajectory = []
        self.actions = []
        self.state_trajectory = []
        if self.training_stage_last:
            self.current_state_phase = 0
        else:
            self.current_state_phase = -1
        self.training_stage = False
        self.training_stage_action = None
        if self.training_stage_last:
            self.state_trajectory.append([-1 for _ in range(5)])
            return np.array(self.state_trajectory)
        else:
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
        transformer = self.transformer[batch_size]
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
        wandb.log({"action": self.actions}, sync=False)
        try:
            # train
            start_time = time()
            train_loader, val_loader, test_loader = self.data[batch_size]
            for epoch in range(self.config['epochs']):
                loss_value_raw = 0
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
                        output = model(X)
                        l = loss(output, y)
                    if self.test:
                        return
                    l.backward()
                    opt.step(batch_size)
                    loss_value_raw += l.mean().asscalar()
                    # 反标准化
                    train_y_min = transformer.y_data_set_min
                    train_y_max = transformer.y_data_set_max
                    output = output * (train_y_max - train_y_min) + train_y_min
                    y = y * (transformer.y_transformer_info[0][1] - transformer.y_transformer_info[0][0]) + \
                        transformer.y_transformer_info[0][0]
                    loss_value += loss(output, y).mean().asscalar()
                    mae += MAE(y, output)
                    rmse += RMSE(y, output)
                    mape += masked_mape_np(y, output)
                train_loader.reset()
                loss_value_raw /= train_batch_num
                loss_value /= train_batch_num
                mae /= train_batch_num
                rmse /= train_batch_num
                mape /= train_batch_num
                self.logger(
                    train=[epoch, loss_value, mae, mape, rmse, (time() - start_time) / self.train_set_sample_num])
                print(f"    epoch:{epoch} ,normal_loss:{loss_value_raw:.6f} ,loss:{loss_value}")
            model_structure = deepcopy(self.action_trajectory)
            model_structure.append(action)
            # eval
            eval_loss_value = 0
            eval_loss_value_raw = 0
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
                output = model(X)
                eval_loss_value_raw += loss(output, y).mean().asscalar()
                # 反标准化
                eval_y_min = transformer.y_data_set_min
                eval_y_max = transformer.y_data_set_max
                output = output * (eval_y_max - eval_y_min) + eval_y_min
                y = y * (transformer.y_transformer_info[1][1] - transformer.y_transformer_info[1][
                    0]) + \
                    transformer.y_transformer_info[1][0]
                eval_loss_value += loss(output, y).mean().asscalar()
                mae += MAE(y, output)
                rmse += RMSE(y, output)
                mape += masked_mape_np(y, output)
            eval_loss_value /= eval_batch_num
            mae /= eval_batch_num
            rmse /= eval_batch_num
            mape /= eval_batch_num
            raw_val_time = time() - val_time
            val_time = raw_val_time / self.eval_set_sample_num
            print(f"    eval_result: loss:{eval_loss_value}, MAE:{mae}, MAPE:{mape}, RMSE:{rmse}, time:{val_time}")
            val_loader.reset()
            # get reward
            reward = -(mae - np.power(np.e, -19) * np.log2(self.max_time/raw_val_time))
            if reward < -1e3:
                return reward/100, True
            else:
                reward /= 100
            self.logger(eval=[eval_loss_value, mae, mape, rmse, val_time])
            self.logger.save_GNN(model, model_structure, reward/len(self.action_trajectory)+1)
            # test
            test_loss_value = 0
            test_loss_value_raw = 0
            test_batch_num = 0
            mae = 0
            rmse = 0
            mape = 0
            start_time = time()
            for X in test_loader:
                y = X.label[0]
                X = X.data[0]
                test_batch_num += 1
                X, y = X.as_in_context(self.ctx), y.as_in_context(self.ctx)
                y = y.astype('float32')
                output = model(X)
                test_loss_value_raw += loss(output, y).mean().asscalar()
                # 反标准化
                test_y_min = transformer.y_data_set_min
                test_y_max = transformer.y_data_set_max
                output = output * (test_y_max - test_y_min) + test_y_min
                y = y * (transformer.y_transformer_info[2][1] - transformer.y_transformer_info[2][0]) + \
                    transformer.y_transformer_info[2][0]
                test_loss_value += loss(output, y).mean().asscalar()
                mae += MAE(y, output)
                rmse += RMSE(y, output)
                mape += masked_mape_np(y, output)
            test_loss_value /= test_batch_num
            mae /= test_batch_num
            rmse /= test_batch_num
            mape /= test_batch_num
            test_loader.reset()
            print(f"    test_result: loss:{test_loss_value}, MAE:{mae}, MAPE:{mape}, RMSE:{rmse}, time:{val_time}")
            self.logger(test=[test_loss_value, mae, mape, rmse, (time() - start_time) / self.test_set_sample_num])
            return reward, False
        except Exception as e:
            self.logger.append_log_file(e.args[0])
            # if "out of memory" in e.args[
            #     0] or "value 0 for Parameter num_args should be greater equal to 1, in operator Concat(name=\"\", num_args=\"0\", dim=\"1\")" in \
            #         e.args[0]:
            self.logger(train=None, eval=None, test=None)
            traceback.print_exc()
            return -1e-3, True
            # else:
            #     traceback.print_exc()
            #     sys.exit()

    def render(self, mode='human'):
        pass
