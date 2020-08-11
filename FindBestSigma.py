import json
from time import time

import wandb

from ExperimentDataLogger import Logger
from Env import GNNEnv
import mxnet as mx
import numpy as np
import os
from utils.utils import generate_random_action

os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

config_filename = "./Config/PEMS03/find_best_sigma.json"
with open(config_filename, 'r') as f:
    config = json.loads(f.read())
print(json.dumps(config, sort_keys=True, indent=4))

if isinstance(config['ctx'], list):
    ctx = [mx.gpu(i) for i in config['ctx']]
elif isinstance(config['ctx'], int):
    ctx = mx.gpu(config['ctx'])
else:
    raise Exception("config_ctx error:" + str(config['ctx']))

wandb.init(project="GNN", config=config, name="experiment_0")
logger = Logger("experiment_0", config)
env = GNNEnv(config, ctx, logger)
episodes = 50
for episode in range(episodes):
    print("====================================================")
    print(f"episode:{episode:}/{episodes}")
    logger.set_episode(episode)
    start = time()
    global reward
    obs = env.reset()
    done = False
    reward_buffer = []
    while not done:
        action = generate_random_action(obs, config['n'], config['training_stage_last'])
        obs, reward, done, _ = env.step(action)
        reward_buffer.append(reward)
        logger(state=obs, action=action, time=time() - start)
    reward = [reward_buffer[-1] / len(reward_buffer) for _ in range(len(reward_buffer))]
    logger(reward=reward)
    start = time() - start
    print(
        f"完成第{episode + 1}次测试,用时:{start:.2f},epochs:{config['epochs']},平均每epochs用时:{start / config['epochs']:.2f},reward:{reward}")
    logger.update_data_units()
    logger.flush_log()
