import numpy as np
# from peak1 import gen_params
from peak_map import gen_XrXp, rank_inputs, gen_params, make_Xr
import random

def gen_data(n_pts, peaks=None):
    if peaks is None:
        peaks = gen_XrXp()
    samples = np.random.random((n_pts + 1,))
    input_x = samples[:-1]
    input_y = np.array([peaks(x) for x in input_x])
    output_x = samples[-1] 
    output_y = peaks(samples[-1])
    return input_x, input_y, output_x, output_y

def gen_batch_data(n_pts, n_batch):
    input_xx, input_yy, output_xx, output_yy = [],[],[],[]
    for i in range(n_batch):
        input_x, input_y, output_x, output_y = gen_data(n_pts)
        input_xx.append(input_x)
        input_yy.append(input_y)
        output_xx.append(output_x)
        output_yy.append(output_y)

    return np.array(input_xx), np.array(input_yy), np.array(output_xx), np.array(output_yy)

def gen_max_data(n_pts, peaks=None):
    if peaks is None:
        peaks = gen_XrXp()
    input_x = np.random.random((n_pts,))
    input_y = np.array([peaks(x) for x in input_x])
    ranked_xs = [x[1] for x in rank_inputs(peaks, (0.5, 1.0))]
    return input_x, input_y, ranked_xs

def gen_max_batch_data(n_pts, n_batch):
    input_xx, input_yy, qry_max_xx, is_max_xx = [],[],[],[]
    for i in range(n_batch):
        a,b,rank_xs = gen_max_data(n_pts)
        input_xx.append(a)
        input_yy.append(b)
        small_ones, big_ones = rank_xs[:900], rank_xs[900:]
        if np.random.random() < 0.5:
            qry_max_xx.append(random.choice(big_ones))
            is_max_xx.append(1)
        else:
            qry_max_xx.append(random.choice(small_ones))
            is_max_xx.append(0)
    return np.array(input_xx), np.array(input_yy), np.array(qry_max_xx), np.array(is_max_xx)


def to_positional(x_batch):
    sin_range = range(1, 9)
    x_sin_enc = np.array([ np.sin(k * (2-0.1) * np.pi * x_batch) for k in sin_range ])
    return np.transpose(x_sin_enc, (1, 2, 0))

if __name__ == '__main__':
    d_obs = gen_batch_data(8, 10)
    d_maxs = gen_max_batch_data(8, 10)
    import pdb; pdb.set_trace();
