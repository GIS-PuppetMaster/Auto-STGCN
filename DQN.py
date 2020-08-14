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
            self.hidden_size = 32
            self.lstm_layer = LSTMCell(input_size=5, hidden_size=self.hidden_size)
        else:
            action_dim = 4
        self.model = Sequential(
            Linear(in_features=(state_dim if not self.training_stage_last else self.hidden_size) + action_dim,
                   out_features=128),
            LeakyReLU(),
            Linear(in_features=128, out_features=256),
            LeakyReLU(),
            Linear(in_features=256, out_features=1),
        )
        # generate all possible action value
        self.action_dict = generate_action_dict(n, training_stage_last)

    def forward(self, x: np.ndarray):
        # x:state
        #   np.ndarray
        #   if self.training_stage_last:
        #      x.shape=(batch_size, 5) or (5, )
        #   else:
        #      x.shape=(batch_size, step, 5) or (step, 5)
        # return:
        #   np.ndarray, tensor
        # shape=(batch_size, ...)
        ############################
        #  construct input tensor  #
        ############################
        if (self.training_stage_last and len(x.shape) == 2) or (not self.training_stage_last and len(x.shape) == 1):
            x = np.expand_dims(x, axis=0)
        max_value_action_batch = []
        values_batch = []
        if self.training_stage_last:
            hidden = (torch.zeros(1, self.hidden_size),
                      torch.zeros(1, self.hidden_size))
            x_lstm = []
            for batch_idx in range(x.shape[0]):
                x_simple = x[batch_idx]
                for i in range(x_simple.shape[0]):
                    input_x = x_simple[i, :].astype(np.float32)
                    hidden = self.lstm_layer(torch.unsqueeze(torch.as_tensor(input_x), dim=0), hidden)
                x_lstm.append(hidden[0])
            x_lstm = torch.cat(x_lstm)

        for batch_idx in range(x.shape[0]):
            x_simple = x[batch_idx]
            if not self.training_stage_last:
                # terminal state
                if (x_simple[1:] == np.array([-1, -1, -1, -1])).all() and x_simple[0] > 0:
                    # only reaching this code while trying to get Q_net(next_obs) and the samples contains terminal state
                    # last state stopped choosing ST-block, now in training stage, but this action is useless because
                    # don't need any actions after ending choosing ST-block, and with done=true, the return of Q_value is
                    # also useless
                    actions = self.action_dict[-1]
                else:
                    actions = self.action_dict[x_simple[0]]
            else:
                if (x_simple[-1, 1:] == np.array([-1, -1, -1, -1])).all() and x_simple[-1, 0] > 0:
                    # last state stopped choosing ST-block, now in training stage
                    # training_stage_last
                    actions = self.action_dict[self.n]
                else:
                    actions = self.action_dict[x_simple[-1, 0]]
            # get Q value
            value = []
            for i, action in enumerate(actions):
                if self.training_stage_last:
                    action = torch.as_tensor(action, dtype=torch.float32)
                    input_array = torch.cat((x_lstm[batch_idx, :], action))
                    value.append(self.model(torch.as_tensor(input_array)))
                else:
                    input_array = np.concatenate((x_simple, action))
                    input_array = input_array.astype(np.float32)
                    # value:Q_value, shape=(action.dim, ), type = torch.Tensor
                    value.append(self.model(torch.as_tensor(input_array)))
            values = torch.cat(value, dim=0)
            # get max Q value and get the action with max Q
            max_value_action = actions[torch.argmax(values), :]
            max_value_action_batch.append(np.expand_dims(max_value_action, axis=0))
            # construct Q_value batch, type:list(torch.Tensor)
            values_batch.append(values)
        # construct action batch, shape=(batch_size, action_dim), type = np.ndarray
        max_value_action_batch = np.concatenate(max_value_action_batch, axis=0)
        return max_value_action_batch, values_batch
