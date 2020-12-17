import argparse
import json
import mxnet as mx
import dill
from ExperimentDataLogger import *
from Env import *
import numpy as np
from utils.utils import generate_action_dict
from collections import defaultdict
import os
from copy import *
import wandb


class QTable:
    def __init__(self, config):
        self.n = config['n']
        # key: tuple, state value: dict:key:tuple, with all possible actions, values:Q_values
        self.Qtable = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.actions = generate_action_dict(self.n)
        self.block_bound = None

    def get_Q_value(self, state, action):
        if isinstance(state, np.ndarray):
            state = tuple(state.tolist())
        if isinstance(action, np.ndarray):
            action = tuple(action.tolist())
        return self.Qtable[state][action]

    def get_action(self, state):
        # return (action, max_Q_value)
        if isinstance(state, np.ndarray):
            state = tuple(state.tolist())
        Q_values = []
        actions = self.actions[state[0]]
        for action in actions:
            if state[0] <= 0 or (state[0] > 0 and ((action[:-1] == self.block_bound).all() or (action == [-1, -1, -1, -1])).all()):
                action = tuple(action.tolist())
                Q_values.append(self.get_Q_value(state, action))
        Q_values = np.array(Q_values)
        Q_value = np.max(Q_values)
        action = np.array(list(actions[np.argmax(Q_values)]))
        # the first ST-Block, record its structure
        if state[0] == 0:
            self.block_bound = action[:-1]
        return action, Q_value

    def set_Q_value(self, state, action, value):
        if isinstance(state, np.ndarray):
            state = tuple(state.tolist())
        if isinstance(action, np.ndarray):
            action = tuple(action.tolist())
        self.Qtable[state][action] = value


def train_QTable(config, log_name):
    #####################
    # set up parameters  #
    ######################
    lr = config['lr']
    episodes = config['episodes']
    gamma = config['gamma']
    n = config['n']
    epsilon = config['epsilon_initial']
    epsilon_decay_step = config["epsilon_decay_step"]
    epsilon_decay_ratio = config["epsilon_decay_ratio"]
    if isinstance(config['ctx'], list):
        ctx = [mx.gpu(i) for i in config['ctx']]
    elif isinstance(config['ctx'], int):
        ctx = mx.gpu(config['ctx'])
    else:
        raise Exception("config_ctx error:" + str(config['ctx']))
    logger = Logger(log_name, config, False)

    #######################
    # init QTable and Env #
    #######################
    q_table = QTable(config)
    env = GNNEnv(config, ctx, logger)

    ##############
    #  training  #
    ##############
    episode = 0
    exception_cnt = False
    while episode < episodes or exception_cnt >= episodes:
        logger.set_episode(episode)
        start_time = time()
        print("====================================================")
        print(f"episode:{episode + 1}/{episodes}")
        # S{-2}
        obs = env.reset()
        done = False
        exception_flag = False
        # store trajectory and edit the reward
        local_buffer = []
        while not done:
            if np.random.random() >= epsilon:
                action, _ = q_table.get_action(obs)
                print(f"state:\n{obs}\naction:{action}    QTable")
            else:
                action = generate_random_action(obs, n)
                print(f"state:\n{obs}\naction:{action}    random")
            # s{-1}-S{T}, T<=n
            # => len(local_buffer)<= T+2
            logger(state=obs, action=action)
            next_obs, reward, done, info = env.step(action)
            exception_flag = info['exception_flag']
            local_buffer.append([obs, action, reward, next_obs, done])
            obs = next_obs
        # edit reward and add into buffer
        reward = local_buffer[-1][2] / len(local_buffer)
        print(f"    reward:{reward}")
        for i in range(len(local_buffer)):
            local_buffer[i][2] = reward
        logger(reward=reward)
        wandb.log({"reward": reward}, sync=False)
        episode += 1
        # training
        for obs, action, reward, next_obs, done in local_buffer:
            if not done:
                q_S_A = q_table.get_Q_value(obs, action)
                q_table.set_Q_value(obs, action,
                                    q_S_A + lr * (reward + gamma * q_table.get_action(next_obs)[1] - q_S_A))
        # epsilon decay
        epsilon *= pow(epsilon_decay_ratio, episode / epsilon_decay_step)
        wandb.log({"epsilon": epsilon}, sync=False)
        if episode % 100 == 0:
            with open(logger.log_path + 'QTable.dill', 'wb') as f:
                dill.dump(q_table, f)
        episode_time = time() - start_time
        print(f"    episode_time_cost:{episode_time}")
        logger(time=episode_time)
        logger.update_data_units()
        logger.flush_log()
    # get best model from logger
    data_unit = np.array(logger.data_unit)
    arr = []
    for episode, data in enumerate(data_unit):
        # Compatible with 'duplicate recording reward each episode' bug in the results of paper experiment
        if isinstance(data[-2], list):
            reward = data[-2][-1]
        else:
            reward = data[-2]
        arr.append([data[1], reward, data[0], data[2], data[3], data[4], data[6], episode])
    arr = np.array(arr)
    arr = arr[np.argsort(arr[:, 1])]
    print(f'action:{arr[-1, 0]} reward:{arr[-1, 1]} episode:{arr[-1, -1]} time:{np.squeeze(arr[-1, 4])[-1]}')
    print(f'log file save to {logger.log_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--train_length', type=int, default=None)
    parser.add_argument('--pred_length', type=int, default=None)
    parser.add_argument('--split_ratio', type=list, default=None)
    parser.add_argument('--time_max', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--epsilon_initial', type=float, default=None)
    parser.add_argument('--epsilon_decay_step', type=int, default=None)
    parser.add_argument('--epsilon_decay_ratio', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--episodes', type=int, default=None)
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--ctx', type=int, default=None)
    args = parser.parse_args()

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
    config.update(arg_dict)

    print(json.dumps(config, sort_keys=True, indent=4))
    if not is_debug():
        log_name = input('log_name:\n')
    else:
        log_name = 'debug'
    wandb.init(project="GNN2", config=config, notes=log_name)
    train_QTable(config, log_name)
