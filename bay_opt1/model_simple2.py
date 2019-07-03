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

        self.opt = torch.optim.RMSprop(self.parameters(), lr=1e-4)

    def forward(self, input_xx, input_yy, output_xx):
        input_yys = torch.unbind(input_yy, 1)

        agg = self.A2N(input_yys[0].unsqueeze(-1))
        return self.fc_mu(agg), None

    def input_to_torch(self, xx, yy, xx_new):
        xx_new = np.expand_dims(xx_new, 1)
        xx_pos = to_positional(xx)
        xx_pos_new = to_positional(xx_new)

        xx_pos = to_torch(xx_pos, "float")
        yy = to_torch(yy, "float")
        xx_pos_new = to_torch(xx_pos_new, "float")
        return xx_pos, yy, xx_pos_new


    def predict(self, xx_pos, yy, xx_pos_new):
        mu, sig = self(xx_pos, yy, xx_pos_new)
        return mu, sig

    # the loss is negative of the log probability . . . which is . . . 
    def loss_function(self, y, mu, sig):
        return torch.sqrt(torch.sum((y - mu) ** 2))

    def learn_once(self, xx, yy, xx_new, yy_new):
        xx, yy, xx_new = self.input_to_torch(xx,yy,xx_new)
        yy_new = to_torch(yy_new, "float")

        self.opt.zero_grad()
        mu, sig = self.predict(xx, yy, xx_new)
        loss = self.loss_function(yy_new, mu, sig)
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
        yy_new = yy_new * 100
        loss = compl.learn_once(xx, yy, xx_new, yy_new)

        if i % 100 == 0:
            compl.save("./saved_models/ver1.mdl")
            print ("------------------------------")
            print ("number observations ", n_obs)
            print ("loss ", loss)
            mu, sig = compl.predict(*compl.input_to_torch(xx,yy,xx_new))
            print ("y0: ", yy[0])
            print ("mu: ", mu[0])
            print ("y*: ", yy_new[0])


