import torch
from torch.nn import *
import numpy as np
from utils.utils import generate_action_dict


class QNet(Module):
    def __init__(self, n, training_stage_last):
        super(QNet, self).__init__()
        self.n = n
        state_dim = 5
        self.training_stage_last = training_stage_last
        if training_stage_last:
            action_dim = 5
        else:
            action_dim = 4
        self.model = Sequential(
            Linear(in_features=state_dim + action_dim, out_features=128),
            Tanh(),
            Linear(in_features=128, out_features=1024),
            Tanh(),
            Linear(in_features=1024, out_features=1024),
            Tanh(),
            Linear(in_features=1024, out_features=128),
            Tanh(),
            Linear(in_features=128, out_features=1),
        )
        # generate all possible action value
        self.action_dict = generate_action_dict(n, training_stage_last)

    def forward(self, x: np.ndarray):
        # x:state
        #   np.ndarray
        #   x.shape=(batch_size, 5) or (5, )
        # return:
        #   np.ndarray, tensor
        # shape=(batch_size, ...)
        ############################
        #  construct input tensor  #
        ############################
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=0)
        max_value_action_batch = []
        values_batch = []
        for idx in range(x.shape[0]):
            x_simple = x[idx, :]
            if (x_simple[1:] == np.array([-1, -1, -1, -1])).all() and x_simple[0] > 0:
                # terminal state
                # only training stage last will need an action after terminal state
                # training state
                if not self.training_stage_last:
                    # only reaching this code while trying to get Q_net(next_obs) and the samples contains terminal state
                    # last state stopped choosing ST-block, now in training stage, but this action is useless because
                    # don't need any actions after ending choosing ST-block, and with done=true, the return of Q_value is
                    # also useless
                    actions = self.action_dict[-1]
                else:
                    # last state stopped choosing ST-block, now in training stage
                    # training_stage_last
                    actions = self.action_dict[self.n]
            else:
                actions = self.action_dict[x_simple[0]]
            # get Q value
            value = []
            for i, action in enumerate(actions):
                input_array = np.concatenate((x_simple, action))
                input_array = input_array.astype(np.float32)
                ##############################################################
                #  value:Q_value, shape=(action.dim, ), type = torch.Tensor  #
                ##############################################################
                value.append(self.model(torch.as_tensor(input_array, )))
            values = torch.cat(value, dim=0)
            # get max Q value and get the action with max Q
            max_value_action = actions[torch.argmax(values), :]
            max_value_action_batch.append(np.expand_dims(max_value_action, axis=0))
            # construct Q_value batch, type:list(torch.Tensor)
            values_batch.append(values)
        # construct action batch, shape=(batch_size, action_dim), type = np.ndarray
        max_value_action_batch = np.concatenate(max_value_action_batch, axis=0)
        return max_value_action_batch, values_batch

# TODO
# class QNetLastTrainingStage(Module):
