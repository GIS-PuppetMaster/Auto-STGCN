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
        self.max_time = config['max_time']
        self.n = config['n']
        self.training_stage_last = config['training_stage_last']
        assert not self.training_stage_last
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
        self.reward = 0

        self.data = {}
        self.batch_size_option = [32, 50, 64]
        self.transformer = {}
        self.train_set_sample_num = 0
        self.eval_set_sample_num = 0
        self.test_set_sample_num = 0

    def step(self, action):
        action = action.squeeze()
        self.actions.append(action.tolist())
        action_sum = np.sum(action)
        if self.current_state_phase <= self.n - 1 and not (
                self.current_state_phase > 0 and (action == np.array([-1, -1, -1, -1])).all()):
            # state{-2}
            # state{-1}
            #    set training stage
            #    collect parameter and apply at the last step
            # state{0}...{n-1}
            state = np.array([self.current_state_phase] + action.tolist())
            state.astype(np.float32)
            if self.current_state_phase % 2 == 0:
                action_sum = -action_sum
            self.reward += action_sum
            self.current_state_phase += 1
            return state, None, False, {"exception_flag": False}
        else:
            state = np.array([self.current_state_phase, -1, -1, -1, -1])
            state.astype(np.float32)

            self.current_state_phase += 1

            return state, self.reward, True, {"exception_flag": False}

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
        self.reward = 0
        if self.training_stage_last:
            self.state_trajectory.append([-1 for _ in range(5)])
            return np.array(self.state_trajectory)
        else:
            return np.array([-2, -1, -1, -1, -1])

    def render(self, mode='human'):
        pass
