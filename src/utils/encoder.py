import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d
import torch.nn.functional as F
import math


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

        x = x.view(x.shape[0]*x.shape[1], math.isqrt(x.shape[2]), math.isqrt(x.shape[2])).unsqueeze(1).cuda()
        #n_workers x 1 x 21 x 21
        x = F.relu(self.Cov1(x))
        x = F.relu(self.Cov2(x))


        x = self.MaxPool(x)
        x = x.view(x.shape[0], -1)

        return x