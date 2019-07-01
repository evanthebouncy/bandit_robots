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

class VAE(nn.Module):
    def __init__(self, n_feature, n_hidden):
        super(VAE, self).__init__()

        self.n_feature = n_feature
        self.n_hidden = n_hidden

        self.fc1 = nn.Linear(2 * n_feature, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)

        self.fc_mu = nn.Linear(n_hidden, n_feature)
        self.fc_logvar = nn.Linear(n_hidden, n_feature)

        self.opt = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x):
        x = x.view(-1, 2 * self.n_feature)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc_mu(h2), self.fc_logvar(h2)

    # the loss is negative of the log probability . . . which is . . . 
    def loss_function(self, x, mu, logvar):
        std = torch.exp(0.5*logvar)
        var = std ** 2

        eye = torch.eye(162).type(torch.cuda.FloatTensor)
        eye = eye.reshape((1, 162, 162))
        eye = eye.repeat(40, 1, 1)

        var = var.unsqueeze(2)
        var = var.repeat(1, 1, 162)

        cov = var * eye
        m = MultivariateNormal(mu, cov) 
        nll = -m.log_prob(x)
        return torch.sum(nll)

    def learn_once(self, x_partial, x):
        self.opt.zero_grad()
        mu, logvar = vae(x_partial)
        loss = vae.loss_function(x, mu, logvar)
        loss.backward()
        vae.opt.step()
        return loss

    def save(self, loc):
        torch.save(self.state_dict(), loc)

    def load(self, loc):
        self.load_state_dict(torch.load(loc))

class UCB:

    def __init__(self, vae):
        self.vae = vae

    # takes in a partial of shape 2 x 162
    # produce an index of most profit
    def query(self, partial):
        partial_X = to_torch(np.array([partial]), "float")
        mus, logvars = self.vae(partial_X)
        mu  = mus[0].detach().cpu().numpy()
        var = logvars[0].exp().detach().cpu().numpy()

        qry_estimate = mu + var
        # blank out things that we have seen . . .
        qry_estimate[partial[1].astype(int).astype(bool)] = -99999

        return np.argmax(qry_estimate)


# VISUALIZE 
def visualise(known, mu_pred, var_pred, truth):
    plt.errorbar(range(162), mu_pred, yerr = var_pred, ecolor='r')
    plt.scatter(range(162), truth, s=2, c='green')

    known_x = np.array(list(range(162)))
    known_y = known

    none_zero_idx, = np.where(known != 0)
    known_x = known_x[none_zero_idx]
    known_y = known_y[none_zero_idx]

    plt.scatter(known_x, known_y, s=2, c='red')
    #plt.scatter(known_x, known_y, s=0.1, c='red')
    plt.savefig('hi2.png')
    plt.close()
    
if __name__ == '__main__':
    
    torch.eye(4).type(torch.cuda.FloatTensor)

    print ("hi")
    m = MultivariateNormal(torch.zeros(2), torch.eye(2))
    haha = m.sample()
    print (haha)
    nll = -m.log_prob(haha)
    print (nll)

    vae = VAE(162, 1300).cuda()
    from data import gen_train_data

    for _ in range(100000):
        partial_X, X = gen_train_data(40)
        partial_X, X = to_torch(partial_X, "float"), to_torch(X, "float")
        loss = vae.learn_once(partial_X, X)
        print (loss)
        if _ % 100 == 0:
            print (_, loss)
            mu, logvar = vae(partial_X)
            known = partial_X[0][0].detach().cpu().numpy()
            mu0 = mu[0].detach().cpu().numpy()
            var0 = logvar[0].exp().detach().cpu().numpy()
            print (np.max(mu0), " maximum mean ")
            print (np.max(var0), " maximum variance ")
            truth = X[0].detach().cpu().numpy()
            visualise(known, mu0, var0, truth)
            vae.save('vae1.mdl')


