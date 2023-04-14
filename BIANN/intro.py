#%%
'''
Written by Shengtong Zhang 2/22/2019

This code is generating a series of comparison between
the MDS method and the exploration method with 
total LHD samples in the d(dim) dimensional space.

Function Visualization problems.
'''

from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils 
import sklearn.preprocessing as prep
from numpy import linalg as LA
# from test_function import test
import os
import pandas as pd
from pandas.plotting import table
from sklearn.linear_model import LinearRegression
from scipy.optimize import linprog
from pyDOE import *

from func.function_class import function, Connect, Depth, Leaves, treeIANNPlot, printTree
from func.var_selection_hessian import variable_selection_tree, index_tree


def bound(dim):
    lb, ub = [-1]*8, [1]*8
    return lb, ub 

def rescale(x, lb, ub):
    # rescale x from [0, 1] to [lb, ub] using linear transformation
    # lb: lower bound; ub: upper bound
    xx = np.copy(x)
    n, m = np.array(xx).shape
    for i in range(m):
        xx[:, i] = xx[:, i] * (ub[i] - lb[i]) + lb[i]
    return np.array(xx)

def func_obj(x_input):
    # f = G * Mm/(x^2+y^2+z^2), x = [G, M, m, x, y, z]
    G = 1
    x = rescale(np.copy(x_input), lb, ub)
    # f = G * x[:,0] * x[:,1] / (np.square(x[:,2]) + np.square(x[:,3]) + np.square(x[:,4]))
    f = (np.square(x[:,0]) + np.square(x[:,1])) * (np.square(x[:,2]) + np.square(x[:,3])) \
        * (np.square(x[:,4]) + np.square(x[:,5])) * (np.square(x[:,6]) + np.square(x[:,7]))
    return f.reshape((-1, 1))


if __name__ == '__main__':
    np.random.seed(12)
    dim = 8

    lb, ub = bound(dim)
    func_dict = {'func_obj': func_obj, 'lb': lb, 'ub': ub}
    func_dict['dim'] = dim
    func_dict['original dimension'] = dim
    func_dict['additive_index'] = []
    func= function(func_dict)

    fig_name = 'intro_'

    #func.func_name = 'borehole'
    func.N1 = 50
    func.n = 200
    
    #Variable Selection
    grid_var = lhs(dim, samples=1000, criterion='m')
    
    group_indices = variable_selection_tree(func.obj, grid_var, lb, ub, reg = 50, grid_length = 0.01)
    
    index_root, tree_data = index_tree(dim, group_indices)
    var_index = Leaves(index_root)
    depth = Depth(index_root)
    printTree(index_root)
    #var_index = var_index[::-1]

    """
    def permute_index(mat, index, ):
        '''
            mat is a n-by-d matrix.
            The function permute the index-column to the first column,
            return a n-by-d matrix
        '''
        n, d = np.shape(mat)
        if np.max(index) >= d:
            print('index out of range. \n')
        return np.concatenate((mat[:, index], np.delete(mat, index, axis = 1)), axis = 1)

    """
    grid = lhs(dim, samples=int(1000), criterion = 'm')
    
    # generate data
    train_x = func.sparse_grid(grid)
    train_y = func.obj(train_x)
    x_val = np.random.uniform(0, 1, size = [int(1e3), dim])
    y_val = func.obj(x_val)

    x_test = np.random.uniform(0, 1, size = [int(1e6), dim])
    y_test = func.obj(x_test)

    train_x = train_x[:, var_index]
    x_val = x_val[:,var_index]
    x_test = x_test[:,var_index]

    regularization = 1e-6
    model = func.IANN_tree_model(index_root, regularization_rate = regularization,\
                                    activation = 'relu', base_node = 64)
    model = func.exploration_train(model, train_x, train_y, x_val, y_val,\
                                        BATCH_SIZE = 128, lr = 1e-3,\
                                        EPOCHS = 500)


    y_pred = model.predict(x_test).flatten()
    y_test = y_test.flatten()
    r2 = 1 - np.var(y_pred - y_test) / np.var(y_test)
    print('the test r^2 is', r2)

    # test t^2 for OVH IANN is: 94.87%

    # Make IANN visualization plots
    num_plot = 10**3
    x_plot = func.generate_grid(dim, num_plot)
    y_plot = func.obj(x_plot)
    x_plot = x_plot[:, var_index]
    
    heatmap = False # Whether to draw the 3D plot or 2D heatmap
    func.tree_plot(index_root, x_plot, y_plot, fig_name, model, labels = ['$x_' + str(i+1) + '$' for i in range(dim)], num_x = 50, heatmap = heatmap)

    # Generate the final IANN visualization tree
    treeIANNPlot(index_root, tree_data, fig_name)






# %%

