import json
from Env import GNNEnv
import mxnet as mx
import numpy as np

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

action_list = np.array([
    [1, 2, 1, 1],
    [1, 2, 3, 0],
    [2, 3, 1, 1],
    [1, 2, 1, 1]
])

for action in action_list:
    env.step(action)