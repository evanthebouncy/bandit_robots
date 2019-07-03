import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import random
from tqdm import tqdm

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
    def __init__(self):
        super(Compl, self).__init__()

        n_hidden = 100

        self.fc1 = nn.Linear(1, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 1)

        self.opt = torch.optim.Adam(self.parameters(), lr=1e-5)

    def forward(self, yy):
        yy = yy.unsqueeze(-1)
        h =  nn.LeakyReLU()(self.fc1(yy))
        return self.fc2(h)

    def loss_function(self, y, y_pred):
        return torch.sum((y - y_pred.squeeze()) ** 2)

    def learn_once(self, yy):
        yy = to_torch(yy, "float")

        self.opt.zero_grad()
        yy_pred = self(yy)
        loss = self.loss_function(yy, yy_pred)
        loss.backward()
        self.opt.step()

        return loss

    def save(self, loc):
        torch.save(self.state_dict(), loc)

    def load(self, loc):
        self.load_state_dict(torch.load(loc))




if __name__ == '__main__':

    compl = Compl().cuda()

    for i in tqdm(range(1000000)):
        yy = np.random.random((100,))
        loss = compl.learn_once(yy)

        if i % 1000 == 0:
            print ("------------------------------")
            yy_pred = compl(to_torch(yy, "float"))
            print ("loss ", loss)
            print ("yy_pred ", yy_pred[0])
            print ("yy ", yy[0])


