from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

try:
    import numpy as np
except:
    exit()

from deap import benchmarks

NUMMAX = 5
NDIM = 2

A = np.random.rand(NUMMAX, 2)
C = 0.01 * np.random.rand(NUMMAX)

# A = [[0.5, 0.5], [0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]]
# C = [0.002, 0.005, 0.005, 0.005, 0.005]

def shekel_arg0(sol):
    return benchmarks.shekel(sol, A, C)[0]

def gen_shekel():
    NUMMAX = 5
    A = np.random.rand(NUMMAX, NDIM)
    C = 0.01 * np.random.rand(NUMMAX)
    def f(sol):
        return benchmarks.shekel(sol, A, C)[0]
    loc = A[np.argmin(C)]
    max_val = f(loc)
    def g(x):
        return f(x) / max_val
    return g, loc

def gen_data(num_points, num_batch):
    # xys of size (num_points, num_batch, 2)
    xss, yss, newxss, newyss = [], [], [], []
    for i in range(num_batch):
        shek, max_loc = gen_shekel()
        xs = np.random.rand(num_points, NDIM)
        ys = [shek(xx) for xx in xs]
        xss.append(xs)
        yss.append(ys)
        x_new = np.random.rand(NDIM)
        y_new = shek(x_new)
        if np.random.random() < 0.5:
            newxss.append(x_new)
            newyss.append(y_new)
        else:
            newxss.append(max_loc)
            newyss.append(1.0) # this is max_val
    return xss, yss, newxss, newyss

loc = A[np.argmin(C)]
print (loc)
print (shekel_arg0(loc))


fig = plt.figure()
# ax = Axes3D(fig, azim = -29, elev = 50)
ax = Axes3D(fig)
X = np.arange(0, 1, 0.01)
Y = np.arange(0, 1, 0.01)
X, Y = np.meshgrid(X, Y)
Z = np.fromiter(map(shekel_arg0, zip(X.flat,Y.flat)), dtype=np.float, count=X.shape[0]*X.shape[1]).reshape(X.shape)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1,  norm=LogNorm(), cmap=cm.jet, linewidth=0.2)
 
plt.xlabel("x")
plt.ylabel("y")

#plt.show()
plt.savefig("ha.png")

if __name__ == '__main__':
    print(gen_data(3, 10))
