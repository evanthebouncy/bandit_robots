import numpy as np
import matplotlib.pyplot as plt
import random

def gen_params():
    return np.random.random((3,))

def get_peak(peak_x, widths):
    def f(x):
        peak_y = 1.0
        w_1, w_2 = widths
        if x < peak_x:
            base_x = peak_x - w_1
            slope = peak_y / w_1
            return max(0, slope * (x - base_x))
        if x >= peak_x:
            base_x = peak_x + w_2
            slope = -peak_y / w_2
            return max(0, slope * (x - base_x))
    return f

def make_Xr(params):
    # widths_xr = np.random.uniform(low=W_RNG[0], high=W_RNG[1], size=2)
    widths_xr = 0.4, 0.4
    f_xr = [get_peak(x, widths_xr) for x in params]
    return lambda x: sum([f(x) for f in f_xr])

def make_Xp(params):
    widths_xp = 0.1, 0.1
    sin_freq1 = np.random.uniform(50, 100)
    sin_freq2 = np.random.uniform(50, 100)
    # sin_freq1 = np.random.uniform(5, 10)
    # sin_freq2 = np.random.uniform(5, 10)
    def f_noise(x):
        return 0.3 * (np.sin(sin_freq1 * x) + np.sin(sin_freq2 * x))
    f_xp = [get_peak(x, widths_xp) for x in params]
    return lambda x: sum([f(x) for f in f_xp]) + f_noise(x)

def join_world(xr, xp):
    def f(x):
        if x  < 0.5:
            return xr(2 * x)
        else:
            return xp(2 * (x - 0.5))
    return f

def plot_peak(peak_f, name):
    plt.plot([peak_f(x) for x in np.linspace(0.0, 1.0, 1000)])
    plt.savefig(name)
    plt.clf()

def rank_inputs(peak_f, rrange=None):
    rrange = (0.0, 1.0) if rrange is None else rrange
    n_div = 1000
    input_samp = np.linspace(rrange[0], rrange[1], n_div)
    noisey = 1 / n_div * np.random.random((n_div,))
    input_samp = input_samp + noisey
    samples = [(peak_f(x),x) for x in input_samp]
    return sorted(samples)

def gen_XrXp():
    params = gen_params()
    Xr = make_Xr(params)
    Xp = make_Xp(params)
    XrXp = join_world(Xr, Xp)
    return XrXp

if __name__ == '__main__':
    params = gen_params()
    Xr = make_Xr(params)
    Xp = make_Xp(params)
    XrXp = join_world(Xr, Xp)
    plot_peak(XrXp, 'XrXp.png')

    print (rank_inputs(XrXp, (0.5, 1.0)))

