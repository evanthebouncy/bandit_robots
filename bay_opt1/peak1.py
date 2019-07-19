import numpy as np
# generate some normal looking peaks of 3 things

M = 3
W_RNG = (0.2, 0.4)
# W_RNG = (100, 101)

def gen_params():
    def gen_peak():
        peak = np.random.random(), 1.0
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

