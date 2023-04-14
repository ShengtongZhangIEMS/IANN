#%%
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils 
import sklearn.preprocessing as prep
from numpy import linalg as LA
import os
import pandas as pd
from pandas.plotting import table
from sklearn.linear_model import LinearRegression
from scipy.optimize import linprog
from pyDOE import *
import tensorflow as tf

from func.function_class import TreeNode, function, Connect, Depth, Leaves, treeIANNPlotOverlap, printTree, convertTreeAux
from func.var_selection_hessian import overlapping_stepwise, overlapping_stepwise_tf


def bound(dim):
    lb = [0]*dim; ub = [1]*dim
    return np.array(lb), np.array(ub) 

def rescale(x, lb, ub):
    # rescale x from [0, 1] to [lb, ub] using linear transformation
    # lb: lower bound; ub: upper bound
    xx = np.copy(x)
    n, m = np.array(xx).shape
    for i in range(m):
        xx[:, i] = xx[:, i] * (ub[i] - lb[i]) + lb[i]
    return np.array(xx)

def func_obj(x_input):
    x = rescale(np.copy(x_input), lb, ub)
    # f = (x[:,0]-x[:,4]*x[:,5])*(np.square(2*x[:,1]-1)*x[:,0] + np.square(2*x[:,1]-2*x[:,2])*x[:,3])
    # f = np.square(2*x[:,0]-x[:,4]*x[:,5]) + np.square(x[:,0]*x[:,1]+3*x[:,1]*x[:,2] - np.square(x[:,2] - x[:,3]))
    # f = (x[:,3]-x[:,4]*x[:,5])*(x[:,0]/(x[:,1]+x[:,2]+0.1) + np.square(x[:,2]-x[:,3]))
    f = np.square(3*x[:,3]+1*x[:,4]+2*x[:,5] - 3) + 8*np.square(x[:,0]+x[:,1]+x[:,2]- 1.5) + 5*np.square(x[:,2] - x[:,3])
    
    return f.reshape((-1, 1))

def sublist(l, indices):
    return list(np.array(l)[indices])


if __name__ == '__main__':
    np.random.seed(seed = 1234)
    tf.random.set_seed(1234)
    dim = 6
    lb, ub = bound(dim)
    fig_name = 'overlap_'
        
    func_dict = {'func_obj': func_obj, 'lb': lb, 'ub': ub}
    func_dict['dim'] = dim
    func_dict['original dimension'] = dim
    func_dict['additive_index'] = []
    func= function(func_dict)

    #func.func_name = 'borehole'
    func.N1 = 50
    func.n = 200

    #Variable Selection
    grid_var = lhs(dim, samples=500, criterion='m')
    groups = [i for i in range(dim)]
    
    grid = lhs(dim, samples=int(2000), criterion = 'm')
    
    # Generate Samples for IANN visualization plots
    num_plot = 10**3
    x_plot = func.generate_grid(dim, num_plot)
    cnt = 0
    
    # define the root of the overlapping tree
    index_root = TreeNode([i for i in range(dim)])
    node = index_root
    objs = [(func.obj, groups, index_root, 1)]
    node_names = []
    
    node_num = [1 for _ in range(dim-1)]
    node_label = {}
    heatmap = False
    while objs:
        obj, groups, node, l = objs.pop(0)
        if len(groups) <= 1:
            continue
        if len(groups) == 2:
            # plot regarding to the original variables
            labels = ['$x_'+str(groups[0]+1)+'$', '$x_'+str(groups[1]+1)+'$'] + node_label[node]
            name = fig_name+' '.join([str(e+1) for e in groups])
            node_names.append(name)
            func.overlap_plot_leaf(obj, groups[0], groups[1], 50, labels, name, heatmap = heatmap)
            node.left = TreeNode(groups[0])
            node.right = TreeNode(groups[1])
            node_num[l] += 2
            continue
        
        groups = sorted(groups)
        if cnt == 0:
            J0, J1, J2 = overlapping_stepwise(obj, grid_var[:, groups], lb, ub, grid_length = 1e-4, thres = 1e-4, reg = 1)
        else:
            J0, J1, J2 = overlapping_stepwise_tf(obj, grid_var[:, groups], thres = 1e-3, reg = 1)
            
        # Compute the original index
        group_0, group_1, group_2 = sublist(groups, J0), sublist(groups, J1), sublist(groups, J2)
        node.left = TreeNode([i for i in sorted(group_0+group_1)])
        node.right = TreeNode([i for i in sorted(group_0+group_2)])
        print(group_0, group_1, group_2)
        
        # generate data
        train_x = func.sparse_grid(grid[:,groups])
        # train_x = grid[:,groups]
        train_y = obj(train_x)
        x_val = np.random.uniform(0, 1, size = [int(1e3), len(groups)])
        y_val = obj(x_val)

        x_test = np.random.uniform(0, 1, size = [int(1e6), len(groups)])
        # x_test = lhs(dim, samples=int(5000), criterion = 'm')
        y_test = obj(x_test)
        if tf.is_tensor(y_test):
            y_test = y_test.numpy()
    
        regularization = 1e-7
        model = func.IANN_overlapping(len(groups), J0, J1, J2, regularization_rate = regularization,\
                                        activation = 'selu', base_node = 128)
        model = func.exploration_train(model, train_x, train_y, x_val, y_val,\
                                            BATCH_SIZE = 128, lr = 1e-3,\
                                            EPOCHS = 500)
        
        y_pred = model.predict(x_test).flatten()
        y_test = y_test.flatten()
        r2 = 1 - np.var(y_pred - y_test) / np.var(y_test)
        print('the test r^2 is', r2)
        
        model_1, model_2, model_dc = func.overlapping_models(model, J0, J1, J2)
         
        objs.append((model_1, sorted(group_0+group_1), node.left, l+1))
        objs.append((model_2, sorted(group_0+group_2), node.right, l+1))

        y_plot = func.obj(x_plot)
        
        n = node_num[l]
        node_num[l] += 2
        if len(group_0+group_1) > 1:
            xlabel = ['$h_{'+ str(l) + ', ' + str(n) + '}$'] 
            node_label[node.left] = xlabel
        else:
            J = group_0+group_1
            xlabel = ['$x_' + str(J[0]+1) + '$']
        if len(group_0+group_2) > 1:
            ylabel = ['$h_{'+ str(l) + ', ' + str(n+1) + '}$'] 
            node_label[node.right] = ylabel
        else:
            J = group_0+group_2
            ylabel = ['$x_' + str(J[0]+1) + '$']
        # zlabel = ['h('+ ' '.join([str(e+1) for e in sorted(J0+J1+J2)]) + ')']  
        if cnt == 0:    zlabel = ['f']
        else:   zlabel = node_label[node]
        labels = xlabel+ylabel+zlabel
        name = fig_name+' '.join([str(e+1) for e in sorted(group_0+group_1+group_2)])
        node_names.append(name)
        if len(group_0) > 0:
            y1 = model_1.predict(train_x[:,J0+J1])
            y2 = model_2.predict(train_x[:,J0+J2])
            corr = np.corrcoef(y1.flatten(), y2.flatten())
            print('The correlation is: ', corr[0,1])
            func.overlap_plot_density(model_1, model_2, model_dc, x_plot, group_0, group_1, group_2, 25, labels, name, heatmap = heatmap)
        else:
            func.overlap_plot(model_1, model_2, model_dc, x_plot, group_0, group_1, group_2, 25, labels, name, heatmap = heatmap)



        cnt += 1
    # Generate the final IANN visualization tree
    newick = []
    convertTreeAux(index_root, newick)
    newick = [str(e) for e in newick]
    tree_data = [''.join(newick)]
    treeIANNPlotOverlap(tree_data, fig_name, node_names)





# %%
