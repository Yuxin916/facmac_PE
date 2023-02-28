import torch.nn as nn
import torch.nn.functional as F
from utils.encoder import obs_encoder
import torch


class QMIXRNNAgent(nn.Module):
    def __init__(self, input_shape, args, use_encoder=True):
        super(QMIXRNNAgent, self).__init__()
        self.args = args
        self.use_encoder = use_encoder

        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.isolate_dim = 4
        if args.obs_last_action:
            self.isolate_dim += self.n_actions
        if args.obs_agent_id:
            self.isolate_dim += self.n_agents
        encode_out_dim = 16

        if self.use_encoder:
            self.use_encoder = obs_encoder(1, encode_out_dim)
            input_shape = encode_out_dim + self.isolate_dim
        else:
            input_shape = input_shape

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        if self.use_encoder:
            a = inputs.view(inputs.shape[0], 1, inputs.shape[1])  # n_workers x 1 x state_dim
            x_ = self.use_encoder(a[:, :, :-self.isolate_dim])  # TODO -4 actions - 2 action dim - 3 agents
            inputs = torch.cat([x_, inputs[:, -self.isolate_dim:]], dim=1)
            pass

        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


class FFAgent(nn.Module):
    def __init__(self, input_shape, args,use_encoder=True):
        super(FFAgent, self).__init__()
        self.args = args
        self.use_encoder = use_encoder

        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.isolate_dim = 4
        if args.obs_last_action:
            self.isolate_dim += self.n_actions
        if args.obs_agent_id:
            self.isolate_dim += self.n_agents
        encode_out_dim = 16

        if self.use_encoder:
            self.use_encoder = obs_encoder(1, encode_out_dim)
            input_shape = encode_out_dim + self.isolate_dim
        else:
            input_shape = input_shape

        # Easiest to reuse rnn_hidden_dim variable
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        if self.use_encoder:
            a = inputs.view(inputs.shape[0], 1, inputs.shape[1])  # n_workers x 1 x state_dim
            x_ = self.use_encoder(a[:, :, :-self.isolate_dim])  # TODO -4 actions - 2 action dim - 3 agents
            inputs = torch.cat([x_, inputs[:, -self.isolate_dim:]], dim=1)
            pass

        x = F.relu(self.fc1(inputs))
        # h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = F.relu(self.fc2(x))
        q = self.fc3(h)
        return q, h