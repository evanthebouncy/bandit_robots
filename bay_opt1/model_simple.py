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

        # going from 4 positional enc, 1 value, 5 to n_hidden
        self.fc1 = nn.Linear(4 + 1, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)

        self.enc_new_x = nn.Linear(4, n_hidden)

        self.relate = nn.Linear(n_hidden + n_hidden, n_hidden)

        self.A2N = nn.Linear(n_hidden, n_hidden)
        self.fc_mu = nn.Linear(n_hidden, 1)
        self.fc_sig = nn.Linear(n_hidden, 1)

        self.opt = torch.optim.RMSprop(self.parameters(), lr=1e-4)

    def forward(self, input_xx, input_yy, output_xx):
        # step1: PREPARE THE ENCODING INTO A LIST OF NODES

        # unstack and roll the xx and yy on the input together into dim 5 vectors
        input_xxs = torch.unbind(input_xx, 1)
        input_yys = torch.unbind(input_yy, 1)
        input_xxyys = [x for x in zip(input_xxs, input_yys)]
        input_cat = [torch.cat((xy[0], torch.unsqueeze(xy[1],1)), dim=1) for xy in input_xxyys]

        # a list of length 5 (number of observed inputs)
        # each element is an encoded repr with shape n_batch x n_hidden
        input_enc = [F.relu(self.fc2(F.relu(self.fc1(xy)))) for xy in input_cat]

        output_enc = F.relu(self.enc_new_x(output_xx.squeeze(1)))

        in_outs = [torch.cat((output_enc, input_e), dim=1) for input_e in input_enc]
        in_outs = [F.relu(self.relate(io)) for io in in_outs]

        agg = torch.mean(torch.stack(in_outs), dim=0)
        agg = F.relu(self.A2N(agg))
        mu, sig = self.fc_mu(agg), self.fc_sig(agg)**2 + 0.01
        return mu, sig

    def predict(self, xx, yy, xx_new):
        xx_new = np.expand_dims(xx_new, 1)
        xx_pos = to_positional(xx)
        xx_pos_new = to_positional(xx_new)

        xx_pos = to_torch(xx_pos, "float")
        yy = to_torch(yy, "float")
        xx_pos_new = to_torch(xx_pos_new, "float")

        mu, sig = self(xx_pos, yy, xx_pos_new)
        return mu, sig

    # the loss is negative of the log probability . . . which is . . . 
    def loss_function(self, y, mu, sig):
        mu, sig = mu.squeeze(-1), sig.squeeze(-1)
        m = Normal(mu, sig)
        nll = -m.log_prob(y)
        loss = torch.sum(nll)
        return loss

    def learn_once(self, xx, yy, xx_new, yy_new):
        self.opt.zero_grad()

        mu, sig = self.predict(xx, yy, xx_new)
        yy_new = to_torch(yy_new, "float")
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
    xx,yy,xx_new,yy_new = gen_batch_data(5, 10)

    mu, sig = compl.predict(xx, yy, xx_new)
    print (mu)
    print (sig)

    for i in tqdm(range(1000000)):
        n_obs = random.choice(list(range(1,20)))
        xx,yy,xx_new,yy_new = gen_batch_data(n_obs, 100)
        loss = compl.learn_once(xx, yy, xx_new, yy_new)

        if i % 100 == 0:
            compl.save("./saved_models/ver1.mdl")
            print ("------------------------------")
            print ("number observations ", n_obs)
            print ("loss ", loss)
            mu, sig = compl.predict(xx,yy,xx_new)
            print ("mu: ", mu[0])
            print ("std: ", sig[0])
            print ("y*: ", yy_new[0])


