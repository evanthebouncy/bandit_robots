import numpy as np
import matplotlib.pyplot as plt

W_RNG = (0.2, 0.4)
# W_RNG = (100, 101)

def gen_params():

    peak_map_x, peak_map_y = 0.5 * np.random.random(), 0.1
    peak_world_x, peak_world_y = 0.5 + peak_map_x, 1.0
    widths_map = np.random.uniform(low=W_RNG[0], high=W_RNG[1], size=2)
    widths_world = 0.02, 0.02
    sin_freq = np.random.uniform(100, 200)

    def f_map(x):
        w_1, w_2 = widths_map
        if x < peak_map_x:
            base_x = peak_map_x - w_1
            slope = peak_map_y / w_1
            return max(0, slope * (x - base_x))
        if x >= peak_map_x:
            base_x = peak_map_x + w_2
            slope = -peak_map_y / w_2
            return max(0, slope * (x - base_x))

    def f_world(x):
        w_1, w_2 = widths_world
        if x < peak_world_x:
            base_x = peak_world_x - w_1
            slope = peak_world_y / w_1
            return max(0, slope * (x - base_x))
        if x >= peak_world_x:
            base_x = peak_world_x + w_2
            slope = -peak_world_y / w_2
            return max(0, slope * (x - base_x))

    def f_noise(x):
        if x < 0.5: 
            return 0.0
        else:
            return 0.5 + 0.1 * np.sin(sin_freq * x)

    return lambda x: max(f_map(x) + f_world(x), f_noise(x))


def plot_peak(peak_f, name):
    plt.plot([peak_f(x) for x in np.linspace(0.0, 1.0, 100)])
    plt.savefig(name)

if __name__ == '__main__':
    ff = gen_params()

    plot_peak(ff, 'map_world.png')

