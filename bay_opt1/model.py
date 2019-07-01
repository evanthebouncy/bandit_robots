import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal

if torch.cuda.is_available():
  def to_torch(x, dtype, req = False):
    tor_type = torch.cuda.LongTensor if dtype == "int" else torch.cuda.FloatTensor
    x = Variable(torch.from_numpy(x).type(tor_type), requires_grad = req)
    return x
else:
  def to_torch(x, dtype, req = False):
    tor_type = torch.LongTensor if dtype == "int" else torch.FloatTensor
    x = Variable(torch.from_numpy(x).type(tor_type), requires_grad = req)
    return x

class Compl(nn.Module):
    def __init__(self, n_hidden):
        super(Compl, self).__init__()

        self.n_hidden = n_hidden

        # going from 4 positional enc, 1 value, 5 to n_hidden
        self.fc1 = nn.Linear(4 + 1, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)

        self.Q = nn.Linear(n_hidden, n_hidden)
        self.K = nn.Linear(n_hidden, n_hidden)
        self.V = nn.Linear(n_hidden, n_hidden)

        self.enc_new_x = nn.Linear(n_hidden + 4, n_hidden)
        self.fc_mu = nn.Linear(n_hidden, 1)
        self.fc_logvar = nn.Linear(n_hidden, 1)

        self.opt = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, input_xx, input_yy, output_xx):
        print (input_xx.size())
        print (input_yy.size())
        print (output_xx.size())

        # unstack and roll the xx and yy on the input together into dim 5 vectors
        input_xxs = torch.unbind(input_xx, 1)
        input_yys = torch.unbind(input_yy, 1)
        input_xxyys = [x for x in zip(input_xxs, input_yys)]
        print (input_xxyys[0])
        input_cat = [torch.cat((xy[0], torch.unsqueeze(xy[1],1)), dim=1) for xy in input_xxyys]

        # a list of length 5 (number of observed inputs)
        # each element is an encoded repr with shape n_batch x n_hidden
        input_enc = [F.relu(self.fc2(F.relu(self.fc1(xy)))) for xy in input_cat]
        print (input_enc)

if __name__ == '__main__':
    from data import gen_batch_data, to_positional

    xx,yy,xx_new,yy_new = gen_batch_data(5, 10)
    xx_pos = to_positional(xx)

    xx_pos = to_torch(xx_pos, "float")
    yy = to_torch(yy, "float")
    xx_new = to_torch(xx_new, "float")
    yy_new = to_torch(yy_new, "float")

    compl = Compl(100).cuda()

    compl(xx_pos, yy, xx_new)


