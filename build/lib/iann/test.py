from regressor import IANNModel
import numpy as np

def bound(dim):
    lb, ub = np.zeros(dim), np.ones(dim)
    return lb, ub 

def func_obj(x):
    #xs = np.square(x-0.5)
    #f = np.multiply(xs[:,0]+ 0.01*xs[:,1], xs[:,2] + xs[:,3]) + xs[:,4]
    #f = np.square(xs[:,0] + np.square(xs[:,1] + xs[:,2])-0.1)
    #f = quadratic(x)
    f = np.square(5. * x[:,0] + x[:,1] + x[:,2] + x[:,3] + x[:,4] - 4.5)
    #f = np.add(x[:,0], np.multiply(x[:,1], x[:,2]))
    #f = x[:,0] + np.square(x[:,1]-0.5) + np.multiply(x[:,1], x[:,2]-0.5)
    return f.reshape((-1, 1))

dim = 5
lb, ub = bound(dim)
iann = IANNModel(func_obj, dim, lb, ub)
iann.fit()
iann.score()
iann.IANNPlot()