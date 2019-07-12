from data import gen_params, gen_data
# from model_simple import Compl
from model_max import Compl
import numpy as np
import matplotlib.pyplot as plt
LOW, HIGH = 0.0, 1.0
import torch.nn.functional as F

def to_np(tor):
    return tor.detach().squeeze().data.cpu().numpy()

def max_sample(compl, xx, yy, n_interv=100):
    xx_new = np.linspace(LOW, HIGH, n_interv)
    xx = np.array([xx for _ in range(n_interv)])
    yy = np.array([yy for _ in range(n_interv)])
    
    max_pred = F.softmax(compl.predict(xx, yy, xx_new), dim=1)

    return to_np(max_pred)

def plot_all(name, peaks, is_max, xx, yy):
    x_sample = np.linspace(LOW, HIGH, len(is_max))
    plt.plot([peaks(x) for x in x_sample])
    plt.scatter(len(is_max) * np.array(xx), yy)
    plt.plot([mm[1] for mm in is_max])
    plt.savefig(name)
    plt.close()

def iterative_plot(compl, peaks, xxs, yys):
    for i in range(1, len(xxs)):
        xx, yy = xxs[:i], yys[:i]
        is_max = max_sample(compl, xx, yy)
        plot_all('drawings/max_infer_{}.png'.format(i), peaks, is_max, xx, yy)

if __name__ == '__main__':
    mod = Compl(100).cuda()
    mod.load('./saved_models/max_7_12.mdl')

    peaks = gen_params()
    xx = np.random.random((20,))
    yy = np.array([peaks(x) for x in xx])
    is_max = max_sample(mod, xx, yy)

    iterative_plot(mod, peaks, xx, yy)
