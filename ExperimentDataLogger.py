import pickle as pk
import numpy as np
import os
import torch
import mxnet as mx
import logging
import pathlib
import json
import dill


class Logger:
    def __init__(self, log_name, config, resume, log_path="./Log/", larger_better=True):
        self.episode = 0
        # data unit: [episode]=[states, actions, train, eval, test, reward, time]
        # states: list(list())
        # actions: list(list())
        # train:[epoch, loss, MAE, MAPE, RMSE, Time]
        # eval:[loss, MAE, MAPE, RMSE, Time]
        # test:[loss, MAE, MAPE, RMSE, Time]
        self.log_name = log_name
        self.log_path = log_path + log_name + "/"
        self.data_unit = []
        self.data_buffer = [[], [], [], [], [], [], None]
        self.larger_better = larger_better
        if larger_better:
            self.metric = -float("inf")
        else:
            self.metric = float("inf")
        self.config = config
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
            os.makedirs(self.log_path + "GNN")
            os.makedirs(self.log_path + "DQN")
        elif not resume:
            raise Exception(f"log_path:{log_path}, log_name:{log_name} already exists")
        pathlib.Path(self.log_path + "logger.log").touch()
        # backup config
        with open(self.log_path + "config.json", "w") as f:
            json.dump(config, f)

    def update_data_units(self):
        self.data_unit.append(self.data_buffer)

    def set_episode(self, episode):
        self.episode = episode
        self.data_buffer = [[], [], [], [], [], [], None]
        with open(self.log_path + "logger.log", "a") as f:
            f.write(f"episode {episode}:\n")

    def flush_log(self):
        with open(self.log_path + "logger.pkl", "wb") as f:
            pk.dump(self, f)

    def filter_input(self, x):
        if isinstance(x, np.ndarray):
            x = np.squeeze(x).tolist()
        elif isinstance(x, torch.Tensor):
            x = x.numpy().tolist()
        elif isinstance(x, mx.nd.NDArray):
            x = x.asnumpy().tolist()
        elif not isinstance(x, list):
            logging.error(f"x type error, get{type(x)}")
        return x

    def append_log_file(self, string):
        with open(self.log_path + "logger.log", "a") as f:
            f.write(string + "\n")

    def save_GNN(self, model, model_structure, metric):
        # only save model when get a ner best model
        if (self.larger_better and metric > self.metric) or (not self.larger_better and metric < self.metric):
            model.save_parameters(self.log_path + f"GNN/best_GNN_model.params")
            with open(self.log_path + "GNN/model_structure.txt", "w") as f:
                f.write(str(model_structure) + "\n")
            self.metric = metric
            print('updated best GNN params')
        if self.episode % 10 == 0:
            model.save_parameters(self.log_path + f"GNN/GNN_model_{self.episode}.params")
            with open(self.log_path + "GNN/model_structure.txt", "w") as f:
                f.write(str(model_structure) + "\n")
            print(f'updated GNN params with checkpoint:{self.episode}')

    def save_DQN(self, model):
        torch.save(model, self.log_path + f"DQN/QNet_{self.episode}")

    def save_buffer(self, buffer):
        with open(self.log_path + "DQN/buffer.dill", "wb") as f:
            dill.dump(buffer, f)

    def __call__(self, flush=False, *args, **kwargs):
        if "state" in kwargs.keys():
            state = self.filter_input(kwargs["state"])
            self.data_buffer[0].append(state)
            with open(self.log_path + "logger.log", "a") as f:
                f.write(f"    state:{state}\n")
        if "action" in kwargs.keys():
            action = self.filter_input(kwargs["action"])
            self.data_buffer[1].append(action)
            with open(self.log_path + "logger.log", "a") as f:
                f.write(f"    action:{action}\n")
        if "train" in kwargs.keys():
            train = kwargs["train"]
            self.data_buffer[2].append(train)
            with open(self.log_path + "logger.log", "a") as f:
                f.write(f"    train:{train}\n")
        if "eval" in kwargs.keys():
            eval = kwargs["eval"]
            self.data_buffer[3].append(eval)
            with open(self.log_path + "logger.log", "a") as f:
                f.write(f"    eval:{eval}\n")
        if "test" in kwargs.keys():
            test = kwargs["test"]
            self.data_buffer[4].append(test)
            with open(self.log_path + "logger.log", "a") as f:
                f.write(f"    test:{test}\n")
        if "reward" in kwargs.keys():
            reward = kwargs["reward"]
            if isinstance(reward, float):
                self.data_buffer[5].append(reward)
            elif isinstance(reward, list):
                self.data_buffer[5] = reward
                reward = reward[-1]
            with open(self.log_path + "logger.log", "a") as f:
                f.write(f"    reward:{reward}\n")
            if reward > self.metric:
                self.metric = reward
                info = f"    best_GNN_model:\n        reward:{reward}\n        actions:{self.data_buffer[1]}\n"
                with open(self.log_path + "logger.log", "a") as f:
                    f.write(info)
                logging.info(info)
        if "time" in kwargs.keys():
            time = kwargs["time"]
            self.data_buffer[6] = time
            with open(self.log_path + "logger.log", "a") as f:
                f.write(f"    time:{time}\n")
        if flush:
            self.flush_log()
