import json
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

env = GNNEnv(config, ctx, test=True)
for j in range(10):
    action_list = []
    action_list.append([np.random.randint(low=1, high=3),
                        np.random.randint(low=1, high=4),
                        np.random.randint(low=1, high=4),
                        np.random.randint(low=1, high=3)])
    for i in range(2):
        if i == 0:
            pre_block = 0
        else:
            pre_block = np.random.randint(low=0, high=i)
        action_list.append([np.random.randint(low=1, high=5),
                            np.random.randint(low=1, high=4),
                            np.random.randint(low=1, high=5),
                            pre_block])
    action_list.append([np.random.randint(low=1, high=3),
                        np.random.randint(low=1, high=4),
                        np.random.randint(low=1, high=4),
                        np.random.randint(low=1, high=4)])

    # action_list = [[2, 2, 3, 1],
    #
    #                [3, 2, 4, 0],
    #                [2, 1, 1, 0],
    #
    #                [2, 1, 2, 3]]
    # action_list = [[2, 1, 3, 1],
    #                [2, 1, 4, 0],
    #                [4, 2, 2, 0],
    #                [1, 3, 3, 2]]
    actions = np.array(action_list)
    print("action:\n" + str(actions))
    for action in actions:
        env.step(action)
    env.reset()
    print(f"完成第{j + 1}次测试")
