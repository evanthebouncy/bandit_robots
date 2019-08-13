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
    sin_freq1 = np.random.uniform(20, 50)
    sin_freq2 = np.random.uniform(20, 50)
    def f_noise(x):
        return 0.3 * (np.sin(sin_freq1 * x) + np.sin(sin_freq2 * x))
    Xr = make_Xr(params)
    return lambda x: Xr(x) + f_noise(x)

def join_world(xr, xp):
    def f(x):
        if not (0 <= x <= 1):
            assert 0, "cmon ur out of range " + str(x)
        if x  < 0.5:
            ret = xr(2 * x)
        else:
            ret = xp(2 * (x - 0.5))

        assert abs(ret) < 10, "something wrong "+str(ret)
        return ret
    return f

def gen_XrXp():
    params = gen_params()
    Xr = make_Xr(params)
    Xp = make_Xp(params)
    XrXp = join_world(Xr, Xp)
    avgg = np.mean([XrXp(x) for x in np.linspace(0.0, 1.0, 1000)])
    return lambda x: XrXp(x) - avgg


def plot_peak(peak_f, name):
    plt.plot([peak_f(x) for x in np.linspace(0.0, 1.0, 1000)])
    plt.savefig(name)
    plt.clf()

if __name__ == '__main__':
    XrXp = gen_XrXp()
    plot_peak(XrXp, 'XrXp.png')


