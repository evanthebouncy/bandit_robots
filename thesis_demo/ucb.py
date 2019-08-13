from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

import numpy as np
import matplotlib.pyplot as plt
from peak_map import gen_XrXp
GPK = RBF(0.5, (0.01, 0.1))

def plot_all(name, peaks, mu, sig, xx, yy):
    x_sample = np.linspace(0, 1, len(mu))
    plt.plot([peaks(x) for x in x_sample], color='blue')
    plt.scatter(len(mu) * np.array(xx), yy, color='blue')
    plt.scatter(len(mu) * np.array(xx)[-1:], yy[-1:], s=80,color='blue')
    #plt.plot(mu)
    plt.fill_between(range(len(mu)), mu-sig, mu+sig, alpha=0.3, color='red')
    plt.savefig(name)
    plt.close()

def iterative_plot(peaks, xxs, yys, mu_sig_fun):
    for i in range(1, len(xxs)):
        xx, yy = xxs[:i], yys[:i]
        mu, sig = mu_sig_fun(xx,yy)
        plot_all('drawings/gp_infer_{}.png'.format(i), peaks, mu, sig, xx, yy)

def gp_mu_sig(xx,yy):
    # Instantiate a Gaussian Process model
    gp = GaussianProcessRegressor(kernel=GPK, n_restarts_optimizer=9)
    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(np.reshape(xx,(-1,1)), yy)

    all_x = np.resize(np.linspace(0, 1, 1000), (-1, 1))
    mu, sig = gp.predict(all_x, return_std=True)
    return mu, sig


def ucb(peaks, mu_sig_infer):
    xx = np.random.uniform(0, 1, (1,))
    yy = np.array([peaks(x) for x in xx])

    for i in range(20):
        mu, sig = mu_sig_infer(xx,yy)
        best_spot = np.argmax(mu + sig) / 1000
        # best_spot = np.random.random()
        yy_new = peaks(best_spot)
        xx = np.array(list(xx) + [best_spot])
        yy = np.array(list(yy) + [yy_new])

    return xx, yy

if __name__ == '__main__':
    peaks = gen_XrXp()
    xxs, yys = ucb(peaks, gp_mu_sig)
    iterative_plot(peaks, xxs, yys,gp_mu_sig)

