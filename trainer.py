from time import time
import torch
from ReplayBuffer import *
from Env import GNNEnv
import mxnet as mx
from DQN import *
import json
import os
from utils.utils import generate_random_action
import wandb
from ExperimentDataLogger import Logger
import argparse

os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

torch.backends.cudnn.benchmark = True


def train_DQN(config, config_name):
    ######################
    # set up parameters  #
    ######################
    opt = config['opt']
    lr = config['DQN_lr']
    episodes = config['episodes']
    exploration = config['exploration']
    replay_buffer_size = config['buffer_size']
    batch_size = config['batch_size']
    gamma = config['gamma']
    warm_up_batches = config['B']
    n = config['n']
    prioritized_replay = config['prioritized_replay']
    prioritized_replay_alpha = config['prioritized_replay_alpha']
    prioritized_replay_beta = config['prioritized_replay_beta']
    prioritized_replay_eps = config['prioritized_replay_eps']
    double_dqn = config['double_dqn']
    training_stage_last = config['training_stage_last']
    target_net_update_feq = config["target_net_update_feq"]
    exploration_update_feq = config["exploration_update_feq"]
    exploration_decay_rate = config["exploration_decay_rate"]
    if isinstance(config['ctx'], list):
        ctx = [mx.gpu(i) for i in config['ctx']]
    elif isinstance(config['ctx'], int):
        ctx = mx.gpu(config['ctx'])
    else:
        raise Exception("config_ctx error:" + str(config['ctx']))
    logger = Logger(config_name, config)
    #####################
    #  build dqn model  #
    #####################
    Q_net = QNet(n, training_stage_last)

    if double_dqn:
        target_Q = QNet(n, training_stage_last)
        target_Q.state_dict().update(Q_net.state_dict())
    if opt == "adam":
        optimizer = torch.optim.Adam(Q_net.parameters(), lr)
    elif opt == "rmsprop":
        optimizer = torch.optim.RMSprop(Q_net.parameters(), lr)
    else:
        raise Exception(f"Wrong opt type:{opt}, only support \"adam\" or \"RMSprop\"")
    loss = torch.nn.L1Loss().cuda()
    loss.requires_grad = True
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(size=replay_buffer_size, alpha=prioritized_replay_alpha)
    else:
        replay_buffer = ReplayBuffer(replay_buffer_size)

    ###################
    #  build GNN Env  #
    ###################
    env = GNNEnv(config, ctx, logger)

    ################
    #  warming up  #
    ################
    for ep in range(warm_up_batches * batch_size//(n+2)):
        print(f"warming up, episode:{ep}/{warm_up_batches * batch_size//(n+2)}")
        # S{-2}
        obs = env.reset()
        done = False
        # store trajectory and edit the reward
        local_buffer = []
        while not done:
            action = generate_random_action(obs, n, training_stage_last)
            action = np.squeeze(action)
            # s{-1}-S{T}, T<=n
            # => len(local_buffer)<= T+2
            next_obs, reward, done, _ = env.step(action)
            local_buffer.append([obs, action, reward, next_obs, done])
            obs = next_obs
        # edit reward and add into buffer
        reward = local_buffer[-1][2]
        for i in range(len(local_buffer) - 1):
            local_buffer[i][2] = reward / len(local_buffer)
            replay_buffer.add(*tuple(local_buffer[i]))

    ##############
    #  training  #
    ##############
    for episode in range(episodes):
        logger.set_episode(episode)
        start_time = time()
        print("====================================================")
        print(f"episode:{episode:}/{episodes}")
        # S{-2}
        obs = env.reset()
        done = False
        # store trajectory and edit the reward
        local_buffer = []
        while not done:
            if np.random.random() >= exploration:
                with torch.no_grad():
                    action, _ = Q_net(obs)
                    action = np.squeeze(action)
                print(f"state:\n{obs}\naction:{action}    Qnet")
            else:
                action = generate_random_action(obs, n, training_stage_last)
                print(f"state:\n{obs}\naction:{action}    random")
            # s{-1}-S{T}, T<=n
            # => len(local_buffer)<= T+2
            logger(state=obs, action=action)
            next_obs, reward, done, info = env.step(action)
            local_buffer.append([obs, action, reward, next_obs, done])
            obs = next_obs
        # edit reward and add into buffer
        reward = local_buffer[-1][2] / len(local_buffer)
        wandb.log({"episode": episode, "reward": reward}, sync=False)
        print(f"    reward:{reward}")
        for i in range(len(local_buffer)):
            local_buffer[i][2] = reward
            logger(reward=reward)
            replay_buffer.add(*tuple(local_buffer[i]))

        # training
        # sampling
        samples = replay_buffer.sample(batch_size, prioritized_replay_beta)
        if prioritized_replay:
            obs, actions, rewards, next_obs, dones, weights, batch_idex = samples
        else:
            obs, actions, rewards, next_obs, dones = samples
        # construct y
        y = []
        if double_dqn:
            _, max_Q_batch = target_Q(next_obs)
        else:
            _, max_Q_batch = Q_net(next_obs)
        for i in range(batch_size):
            if dones[i] == 0:
                max_Q = torch.max(max_Q_batch[i])
                y.append(torch.unsqueeze(rewards[i] + gamma * max_Q, dim=0))
            else:
                y.append(torch.unsqueeze(torch.tensor(rewards[i], requires_grad=True), dim=0))
        y = torch.cat(y, dim=0)
        _, values = Q_net(obs)
        logits = []
        for value in values:
            logits.append(torch.unsqueeze(torch.max(value), dim=0))
        logits = torch.cat(logits, dim=0)
        # train
        l = loss(logits, y)
        l.backward()
        optimizer.step()

        # update buffer
        td_error = logits - y
        if prioritized_replay:
            replay_buffer.update_priorities(batch_idex, td_error + prioritized_replay_eps)

        # update target Q_net
        if double_dqn and episode != 0 and episode % target_net_update_feq == 0:
            target_Q.state_dict().update(Q_net.state_dict())
        if episode != 0 and episode % exploration_update_feq == 0:
            exploration_decay_rate *= exploration_decay_rate
        episode_time = time() - start_time
        print(f"    episode_time_cost:{episode_time}")
        logger(time=episode_time)
        logger.save_DQN(Q_net)
        logger.update_data_units()
        logger.flush_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()
    config_filename = args.config
    with open(config_filename, 'r') as f:
        config = json.loads(f.read())
    print(json.dumps(config, sort_keys=True, indent=4))
    wandb.init(project="GNN", config=config)
    train_DQN(config, config_filename.replace("/", "_").split('.')[1])
