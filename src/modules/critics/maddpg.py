import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from utils.encoder import obs_encoder



class MADDPGCritic(nn.Module):
    def __init__(self, scheme, args, use_encoder=True):
        super(MADDPGCritic, self).__init__()
        self.args = args
        self.use_encoder = use_encoder

        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme) + self.n_actions * self.n_agents
        self.obs_shape = self._get_obs_shape(scheme)
        self.output_type = "q"
        self.isolate_dim = 4
        self.encode_out_dim = 16

        if self.use_encoder:
            self.use_encoder = obs_encoder(1, self.encode_out_dim)
            self.input_shape = (self.encode_out_dim+self.n_actions) * self.n_agents + self.isolate_dim

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, 1)

    def forward(self, inputs, actions, hidden_state=None):
        if self.use_encoder:
            new_shape = (inputs.shape[0], self.n_agents, self.obs_shape)
            a = th.reshape(inputs, new_shape)[:, :, :-self.isolate_dim]
            a = a.view(a.shape[0]*a.shape[1], 1, a.shape[2])
            x_ = self.use_encoder(a)
            x_ = th.reshape(x_, (inputs.shape[0], self.n_agents*self.encode_out_dim))
            inputs = th.cat([x_, inputs[:, -self.isolate_dim:]], dim=1)
            pass

        if actions is not None:
            inputs = th.cat([inputs,
                             actions.contiguous().view(-1, self.n_actions * self.n_agents)], dim=-1)


        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q, hidden_state

    def _get_input_shape(self, scheme):
        # The centralized critic takes the state input, not observation
        input_shape = scheme["state"]["vshape"]
        return input_shape

    def _get_obs_shape(self, scheme):
        # The centralized critic takes the state input, not observation
        obs_shape = scheme["obs"]["vshape"]
        return obs_shape