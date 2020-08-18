import argparse
import json
from trainer import train_DQN
from Qlearning import train_QTable

import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()
    config_filename = args.config
    with open(config_filename, 'r') as f:
        config = json.loads(f.read())
    print(json.dumps(config, sort_keys=True, indent=4))
    wandb.init(project="GNN", config=config)
    if 'model' not in config.keys():
        model_type = input('please input the model type(DQN/Qlearning): ')
    else:
        model_type = config['model']
    if model_type == 'DQN':
        print("running DQN")
        train_DQN(config, config_filename.replace('./Config/', '').replace("/", "_").split('.')[0])
    else:
        print("running Qlearning")
        train_QTable(config, config_filename.replace('./Config/', '').replace("/", "_").split('.')[0])
