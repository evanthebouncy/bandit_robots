import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal
from tqdm import tqdm

from data import gen_max_batch_data, to_positional

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

        # going from 8 positional enc, 1 value, 5 to n_hidden
        self.fc1 = nn.Linear(8 + 1, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)

        self.enc_new_x = nn.Linear(8, n_hidden)

        self.Q = nn.Linear(n_hidden, n_hidden)
        self.K = nn.Linear(n_hidden, n_hidden)
        self.V = nn.Linear(n_hidden, n_hidden)

        self.QK = nn.Linear(n_hidden + n_hidden, 1)
        self.A2N = nn.Linear(n_hidden, n_hidden)


        self.pred = nn.Linear(n_hidden, 2)

        self.opt = torch.optim.RMSprop(self.parameters(), lr=1e-4)

    def communicate(self, nodes):
        nn = len(nodes)
        qq = [self.Q(node) for node in nodes]
        kk = [self.K(node) for node in nodes]
        vv = torch.stack([self.V(node) for node in nodes]).transpose(0,1)

        ret = []
        for q in qq:
            # put the q and k together and compute the weight softmax
            qks = [torch.cat((q,k),dim=1)  for k in kk]
            qk_weights = [(self.QK(qk)).squeeze(-1) for qk in qks]
            qk_w = F.softmax(torch.stack(qk_weights).transpose(0,1), dim=1)
            aggre = torch.sum(qk_w.unsqueeze(2).repeat(1,1,self.n_hidden) * vv, dim=1)
            node_new = F.relu(self.A2N(aggre)) 
            ret.append(node_new)
        return ret


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

        nodes_enc = input_enc + [output_enc]

        # step2: serveral rounds of msg passing
        for i in range(1):
            nodes_enc = self.communicate(nodes_enc)

        agg, _ = torch.max(torch.stack(nodes_enc), dim=0)
        return F.log_softmax(self.pred(agg), dim=1)

    def predict(self, xx, yy, xx_new):
        xx_new = np.expand_dims(xx_new, 1)
        xx_pos = to_positional(xx)
        xx_pos_new = to_positional(xx_new)

        xx_pos = to_torch(xx_pos, "float")
        yy = to_torch(yy, "float")
        xx_pos_new = to_torch(xx_pos_new, "float")

        is_max_pred = self(xx_pos, yy, xx_pos_new)
        return is_max_pred

    # the loss is negative of the log probability . . . which is . . . 
    def loss_function(self, y, y_pred):
        loss = F.nll_loss(y_pred, y)
        return loss


    def learn_once(self, xx, yy, xx_new, is_max):
        self.opt.zero_grad()

        max_pred = self.predict(xx, yy, xx_new)
        is_max = to_torch(is_max, "int")
        loss = self.loss_function(is_max, max_pred)

        loss.backward()
        self.opt.step()
        return loss

    def save(self, loc):
        torch.save(self.state_dict(), loc)

    def load(self, loc):
        self.load_state_dict(torch.load(loc))




if __name__ == '__main__':

    compl = Compl(100).cuda()
    xx,yy,xx_new,is_max = gen_max_batch_data(5, 10)

    pred = compl.predict(xx, yy, xx_new)

    for i in tqdm(range(1000000)):
        n_obs = random.choice(list(range(1,20)))
        xx,yy,xx_new,is_max = gen_max_batch_data(n_obs, 100)
        loss = compl.learn_once(xx, yy, xx_new, is_max)

        if i % 100 == 0:
            compl.save("./saved_models/max_ver1.mdl")
            print ("------------------------------")
            print ("number observations ", n_obs)
            print ("loss ", loss)
            pred = compl.predict(xx,yy,xx_new)
            print ("pred logpr ", pred[0])
            print ("pred ", np.argmax(pred[0].detach().cpu().numpy()))
            print ("truth: ", is_max[0])


