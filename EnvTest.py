import json
from time import time

from Env import GNNEnv
import mxnet as mx
import numpy as np
import os

os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

config_filename = "./Config/test_env.json"
with open(config_filename, 'r') as f:
    config = json.loads(f.read())
print(json.dumps(config, sort_keys=True, indent=4))

if isinstance(config['ctx'], list):
    ctx = [mx.gpu(i) for i in config['ctx']]
elif isinstance(config['ctx'], int):
    ctx = mx.gpu(config['ctx'])
else:
    raise Exception("config_ctx error:" + str(config['ctx']))

env = GNNEnv(config, ctx)
for j in range(10):
    action_list = []
    if config['training_stage_last']:
        action_list.append([np.random.randint(low=1, high=3),
                            np.random.randint(low=1, high=4),
                            np.random.randint(low=1, high=4),
                            np.random.randint(low=1, high=3),
                            np.random.randint(low=0, high=2)])
        for i in range(config['n']):
            if i == 0:
                pre_block = 0
            else:
                pre_block = np.random.randint(low=0, high=i)
            action_list.append([np.random.randint(low=1, high=5),
                                np.random.randint(low=1, high=4),
                                np.random.randint(low=1, high=5),
                                pre_block,
                                np.random.randint(low=0, high=2)])
        action_list.append([np.random.randint(low=1, high=3),
                            np.random.randint(low=1, high=4),
                            np.random.randint(low=1, high=4),
                            np.random.randint(low=1, high=4),
                            np.random.randint(low=0, high=2)])
    else:
        action_list.append([np.random.randint(low=1, high=3),
                            np.random.randint(low=1, high=4),
                            np.random.randint(low=1, high=4),
                            np.random.randint(low=1, high=3),
                            np.random.randint(low=0, high=2)])
        action_list.append([np.random.randint(low=1, high=3),
                            np.random.randint(low=1, high=4),
                            np.random.randint(low=1, high=4),
                            np.random.randint(low=1, high=4),
                            np.random.randint(low=0, high=2)])
        for i in range(config['n']):
            if i == 0:
                pre_block = 0
            else:
                pre_block = np.random.randint(low=0, high=i)
            action_list.append([np.random.randint(low=1, high=5),
                                np.random.randint(low=1, high=4),
                                np.random.randint(low=1, high=5),
                                pre_block,
                                np.random.randint(low=0, high=2)])
    action_list = [[1, 3, 3, 1, 0],
                   [3, 1, 1, 0, 0],
                   [1, 3, 4, 1, 0],
                   [2, 3, 1, 2, 0],
                   [3, 2, 2, 3, 0],
                   [2, 3, 2, 3, 0]]
    actions = np.array(action_list)
    print("action:\n" + str(actions))
    start = time()
    global reward
    state = env.reset()
    for action in actions:
        state, reward, done, _ = env.step(action)
        if done:
            start = time() - start
            break
    print(
        f"完成第{j + 1}次测试,用时:{start:.2f},epochs:{config['epochs']},平均每epochs用时:{start / config['epochs']:.2f},reward:{reward}")
