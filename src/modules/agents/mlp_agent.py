import torch.nn as nn
import torch.nn.functional as F
from utils.encoder import obs_encoder
import torch


class MLPAgent(nn.Module):
    def __init__(self, input_shape, args, use_encoder=True):
        super(MLPAgent, self).__init__()
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
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim // 2)
        self.fc3 = nn.Linear(args.rnn_hidden_dim // 2, args.n_actions)

        self.agent_return_logits = getattr(self.args, "agent_return_logits", False)

    def init_hidden(self):
        '''
        initializing the hidden state of an RNN
        returns a tensor of zeros with a shape of (1, self.args.rnn_hidden_dim)
            [self.args.rnn_hidden_dim: hyperparameter that determines the number of hidden units in the RNN]

        using the new method of the self.fc1.weight tensor,
            which creates a new tensor with the same data type and device as self.fc1.weight.
        '''
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, actions=None):
        if self.use_encoder:
            a = inputs.view(inputs.shape[0], 1, inputs.shape[1])  # n_workers x 1 x state_dim
            x_ = self.use_encoder(a[:, :, :-self.isolate_dim])  # TODO -4 actions - 2 action dim - 3 agents
            inputs = torch.cat([x_, inputs[:, -self.isolate_dim:]], dim=1)
            pass

        inputs = F.relu(self.fc1(inputs))
        inputs = F.relu(self.fc2(inputs))
        if self.agent_return_logits:
            actions = self.fc3(inputs)
        else:
            actions = torch.tanh(self.fc3(inputs))
        return {"actions": actions, "hidden_state": hidden_state}
