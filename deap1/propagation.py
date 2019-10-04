from shekel import gen_data
import torch
import torch.nn as nn
from utilities import Module
import transformer as tr
import math
import numpy as np
import torch.nn.functional as F
from torch.distributions.normal import Normal
import tqdm

class Prop(Module):
    def __init__(self, hidden_size=64):
        super(Prop, self).__init__()
        input_vertex_dim = 3
        self.embedding = nn.Linear(5, hidden_size)
        self.transformer = tr.TransformerEncoder(6, 8, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, 1)
        self.fc_sig = nn.Linear(hidden_size, 1)
        self.opt = torch.optim.RMSprop(self.parameters(), lr=1e-3)
        self.finalize()

    def encode_point(self, x, y, query_x):
        return np.concatenate((x,np.array([y]),x-query_x))

    def featurize(self, xs, ys, new_x):
        items = []
        for x, y in zip(xs, ys):
            items.append(self.encode_point(x,y,new_x))
        return items

    def forward(self, xss, yss, newxss):
        # compute the features
        xsysnewx = zip(xss, yss, newxss)
        items = self.tensor([self.featurize(*tt) for tt in xsysnewx])
        items_emb = self.embedding(items)
        # n_items is the same across batch, so literally like [3, 3, 3, 3, .. 3]
        n_items = items.size(0) * [items.size()[1]]
        transformed_items = self.transformer(items_emb, n_items)
        agg, _ = torch.max(transformed_items, dim=1)
        mu, sig = self.fc_mu(agg), torch.abs(self.fc_sig(agg)) + 0.01
        return mu, sig

    def loss(self, xss, yss, newxss, y):
        mu, sig = self(xss, yss, newxss)
        mu, sig = mu.squeeze(-1), sig.squeeze(-1)
        y = self.tensor(y)
        m = Normal(mu, sig)
        nll = -m.log_prob(y)
        loss = torch.sum(nll)
        return loss

    def learn_once(self, xss, yss, newxss, y):
        self.opt.zero_grad()
        loss = self.loss(xss, yss, newxss, y)
        loss.backward()
        # do some gradient clipping to be safe
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
        self.opt.step()
        return loss

    def save(self, loc):
        torch.save(self.state_dict(), loc)

    def load(self, loc):
        self.load_state_dict(torch.load(loc))
        
if __name__ == '__main__':
    prop = Prop()

    for i in tqdm.tqdm(range(10000000)):
        n_ob = np.random.randint(1, 11)
        xss, yss, newxss, newyss = gen_data(n_ob, 100)
        loss = prop.learn_once(xss, yss, newxss, newyss)
        if i % 1000 == 0:
            print (loss, n_ob)
            prop.save("saved_models/prop.mdl")


