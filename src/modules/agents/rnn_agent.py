import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

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
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        actions = F.tanh(self.fc2(h))
        return {"actions": actions, "hidden_state": h}