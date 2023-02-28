import torch.nn as nn
import torch.nn.functional as F
from torch.nn import  Conv2d, MaxPool2d
from torch import flatten
import math
from torch.nn import  Conv2d, MaxPool2d
from torch import flatten
import math
import torch

class RNNAgent(nn.Module):
    def __init__(self, input_shape, args,use_encoder=True):
        super(RNNAgent, self).__init__()
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
            self.use_encoder = obs_encoder(1,encode_out_dim)
            input_shape = encode_out_dim + self.isolate_dim
        else:
            input_shape = input_shape

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        # make sure the output layer is the same as rnn hidden dim
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

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
        '''
        input:
            inputs: (num_agents, obs_dim)
            hidden_state: (1, num_agents, rnn_hidden_dim)
        output:
            actions: (num_agents, action_dim)
            h: (num_agents, rnn_hidden_dim) updated hidden_state tensor.
        '''
        if self.use_encoder:
            a = inputs.view(inputs.shape[0], 1, inputs.shape[1]) #n_workers x 1 x state_dim
            x_ = self.use_encoder(a[:, :, :-self.isolate_dim]) # TODO -4 actions - 2 action dim - 3 agents
            inputs = torch.cat([x_, inputs[:, -self.isolate_dim:]], dim=1)
            pass

        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        actions = F.tanh(self.fc2(h))
        return {"actions": actions, "hidden_state": h}

class obs_encoder(nn.Module):
    def __init__(self, input_dim, hidden_size=16):
        super(obs_encoder, self).__init__()

        self.Cov1 = Conv2d(in_channels=input_dim, out_channels=hidden_size, kernel_size=(3,3), stride=(2,2), padding=(1,1)).cuda()
        self.Cov1.weight = torch.nn.Parameter(self.Cov1.weight.cuda().float())  # convert weight tensor to float32
        self.Cov1.bias = torch.nn.Parameter(self.Cov1.bias.cuda().float())  # convert bias tensor to float32

        self.Cov2 = Conv2d(in_channels=hidden_size, out_channels=4, kernel_size=(2,2), stride=2, padding=1)
        # self.MaxPool = MaxPool1d(3, stride=2)
        self.MaxPool = MaxPool2d(3, stride=2)
        pass

    def forward(self, x):

        x = x.view(x.shape[0], math.isqrt(x.shape[2]), math.isqrt(x.shape[2])).unsqueeze(1).cuda()
        #n_workers x 1 x 21 x 21
        x = F.relu(self.Cov1(x))
        x = F.relu(self.Cov2(x))


        x = self.MaxPool(x)
        x = x.view(x.shape[0], -1)

        return x