import numpy as np
# from peak1 import gen_params
from peak_map import gen_params

def gen_data(n_pts, peaks=None):
    if peaks is None:
        peaks = gen_params()
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
        peaks = gen_params()
    input_x = np.random.random((n_pts,))
    input_y = np.array([peaks(x) for x in input_x])
    dense_x = np.linspace(0, 1, 100)[:-1] + np.random.random() / 100
    max_x = max([(peaks(xx),xx) for xx in dense_x])[1]
    return input_x, input_y, max_x

def gen_max_batch_data(n_pts, n_batch):
    input_xx, input_yy, qry_max_xx, is_max_xx = [],[],[],[]
    for i in range(n_batch):
        a,b,c = gen_max_data(n_pts)
        input_xx.append(a)
        input_yy.append(b)
        if np.random.random() < 0.5:
            qry_max_xx.append(c)
            is_max_xx.append(1)
        else:
            qry_max_xx.append(np.random.random())
            is_max_xx.append(0)
    return np.array(input_xx), np.array(input_yy), np.array(qry_max_xx), np.array(is_max_xx)


def to_positional(x_batch):
    x_cos_enc = np.array([ np.cos(k * 2 * np.pi * x_batch) for k in [1,2,3,4] ])
    x_sin_enc = np.array([ np.sin(k * 2 * np.pi * x_batch) for k in [1,2,3,4] ])
    x_enc = np.concatenate((x_cos_enc, x_sin_enc))
    return np.transpose(x_enc, (1, 2, 0))

def plot_peak(peak_f, name):
    plt.plot([ff(x) for x in np.linspace(0.0, 1.0, 100)])
    plt.savefig(name)

if __name__ == '__main__':
    ff = gen_params()


    import matplotlib.pyplot as plt
    plt.plot([ff(x) for x in np.linspace(0.0, 1.0, 100)])
    plt.ylabel('some numbers')
    plt.savefig('ha.png')

    xx, yy, xx_new, yy_new = gen_batch_data(5, 10)
    print (yy[0])
    print (yy_new[0])
    # print (to_positional(xx).shape)

    xx, yy, max_x = gen_max_data(10)
    print (xx)
    print (yy)
    print (max_x)

    gen_max_batch_data(8, 10)
