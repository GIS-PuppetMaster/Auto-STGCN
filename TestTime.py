import argparse
import json
from time import time

from gym import spaces

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
from copy import deepcopy


class TestEnv(GNNEnv):
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
        if self.training_stage_last:
            action = actions[-1]
            actions.pop(-1)
        else:
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
        # test
        train_loader, val_loader, test_loader = self.data[batch_size]
        test_time = []
        for i in range(5):
            test_batch_num = 0
            times = 0
            for X in test_loader:
                X = X.data[0]
                test_batch_num += 1
                X = X.as_in_context(self.ctx)
                start_time = time()
                model(X)
                times += (time() - start_time)
            test_loader.reset()
            test_time.append(times)
            print(f"    test_result: time:{times}")
            self.logger(test=[None, None, None, None, times])
            self.logger.update_data_units()
            self.logger.flush_log()
        res = np.array(test_time)
        logger.append_log_file(f'mean_time:{res.mean()}\nstd_time:{res.std()}')
        return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--wandb_id', type=str, default=None)
    args = parser.parse_args()
    model_filename = args.model
    config_filename = args.config
    with open(config_filename, 'r') as f:
        config = json.loads(f.read())
    with open(model_filename, 'r') as f:
        actions = json.loads(f.read())
    print(json.dumps(config, sort_keys=True, indent=4))
    if isinstance(config['ctx'], list):
        ctx = [mx.gpu(i) for i in config['ctx']]
    elif isinstance(config['ctx'], int):
        ctx = mx.gpu(config['ctx'])
    else:
        raise Exception("config_ctx error:" + str(config['ctx']))
    config_name = config_filename.replace('./Config/', '').replace("/", "_").split('.')[0] + '_' + \
                  model_filename.replace('./Config/', '').replace("/", "_").split('.')[0] + '_test_time'
    logger = Logger(config_name, config, args.resume, larger_better=False)
    env = TestEnv(config, ctx, logger)
    env.train_model(deepcopy(actions))
