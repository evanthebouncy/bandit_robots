from peak_map import gen_XrXp, rank_inputs, gen_params, make_Xr
# from model_simple import Compl
from model import Compl
import numpy as np
import matplotlib.pyplot as plt
LOW, HIGH = 0.0, 1.0

def to_np(tor):
    return tor.detach().squeeze().data.cpu().numpy()

def bay_sample(compl, xx, yy, n_interv=1000):
    xx_new = np.linspace(LOW, HIGH, n_interv)
    xx = np.array([xx for _ in range(n_interv)])
    yy = np.array([yy for _ in range(n_interv)])
    
    yy_mu, yy_sig = compl.predict(xx, yy, xx_new)
    return to_np(yy_mu), to_np(yy_sig)

def plot_all(name, peaks, mu, sig, xx, yy):
    x_sample = np.linspace(LOW, HIGH, len(mu))
    plt.plot([peaks(x) for x in x_sample])
    plt.plot(mu-sig)
    plt.plot(mu)
    plt.plot(mu+sig)
    plt.scatter(len(mu) * np.array(xx), yy)
    plt.savefig(name)
    plt.close()

def iterative_plot(peaks, xxs, yys):
    for i in range(1, len(xxs)):
        xx, yy = xxs[:i], yys[:i]
        mu, sig = bay_sample(compl, xx, yy)
        plot_all('drawings/infer_{}.png'.format(i), peaks, mu, sig, xx, yy)

def active_sample(compl, peaks):
    xx = np.random.random((1,))
    yy = np.array([peaks(x) for x in xx])
    
    for i in range(20):
        mu, sig = bay_sample(compl, xx, yy)

        # hack in this part so to avoid edges when encoding is poor
        # also to avoid sample same place twice
        for x in xx:
            x_idx = int(x * len(mu))
            sig[x_idx] = 0
        most_confuse = np.argmax(sig) / len(sig)
        # most_confuse = np.random.random()
        yy_new = peaks(most_confuse)
        xx = np.array(list(xx) + [most_confuse])
        yy = np.array(list(yy) + [yy_new])

    return xx, yy

if __name__ == '__main__':
    compl = Compl(100).cuda()
    compl.load('./saved_models/ob1.mdl')

    peaks = gen_XrXp()
    # peaks = make_Xr(gen_params())
    xxs, yys = active_sample(compl, peaks)
    iterative_plot(peaks, xxs, yys)
