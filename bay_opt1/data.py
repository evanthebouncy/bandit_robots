import numpy as np

M = 1
# W_RNG = (0.4, 0.6)
W_RNG = (100, 101)

def gen_params():
    def gen_peak():
        peak = np.random.random(), np.random.uniform(0.1, 1.0)
        widths = np.random.uniform(low=W_RNG[0], high=W_RNG[1], size=2)
        def f(x):
            peak_x, peak_y = peak
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

    peaks = [gen_peak() for _ in range(M)]

    def max_peak(x):
        return sum([peak(x) for peak in peaks])

    return max_peak

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

def to_positional(x_batch):
    x_enc = np.array([ np.cos(k * 2 * np.pi * x_batch) for k in [1,2,3,4] ])
    return np.transpose(x_enc, (1, 2, 0))

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

