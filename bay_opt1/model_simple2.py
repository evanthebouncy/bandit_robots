import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal
from tqdm import tqdm

from data import gen_batch_data, to_positional

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

        self.A2N = nn.Linear(1, n_hidden)
        self.fc_mu = nn.Linear(n_hidden, 1)

        self.opt = torch.optim.SGD(self.parameters(), lr=1e-5)

    def forward(self, input_yy):
        input_yy = input_yy.unsqueeze(-1)
        agg =  nn.LeakyReLU()(self.A2N(input_yy))
        return self.fc_mu(agg)

    def predict(self, yy):
        mu = self(yy)
        return mu

    # the loss is negative of the log probability . . . which is . . . 
    def loss_function(self, y, mu):
        return torch.sum((y - mu) ** 2)

    def learn_once(self, yy, yy_new):
        yy = to_torch(yy, "float")
        yy_new = to_torch(yy_new, "float")

        self.opt.zero_grad()
        mu = self.predict(yy)
        loss = self.loss_function(yy_new, mu)
        loss.backward()
        self.opt.step()

        return loss

    def save(self, loc):
        torch.save(self.state_dict(), loc)

    def load(self, loc):
        self.load_state_dict(torch.load(loc))




if __name__ == '__main__':

    compl = Compl(100).cuda()

    for i in tqdm(range(1000000)):
        n_obs = random.choice(list(range(1,20)))
        xx,yy,xx_new,yy_new = gen_batch_data(n_obs, 100)
        yy = yy[:,0]
        yy_new = yy_new
        loss = compl.learn_once(yy, yy_new)

        if i % 100 == 0:
            compl.save("./saved_models/ver1.mdl")
            print ("------------------------------")
            print ("number observations ", n_obs)
            print ("loss ", loss)
            mu = compl.predict(to_torch(yy, "float"))
            print ("y0: ", yy[0])
            print ("mu: ", mu[0])
            print ("y*: ", yy_new[0])


