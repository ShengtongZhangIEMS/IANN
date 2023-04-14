"""
Created on Mon Oct 22 15:29:15 2018

@author: Shengtong Zhang
"""

import numpy as np
from numpy import linalg as LA

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LightSource

import scipy as sp
import sys
sys.path.append('..')
from copy import copy
import sklearn 
from sklearn.datasets import load_digits
from sklearn.manifold import Isomap

from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LinearRegression
import sklearn.preprocessing as prep
from scipy.spatial import distance_matrix
import pandas as pd

import random, math
from pyDOE import *
from scipy.optimize import linprog
import os
import tensorflow as tf


from keras.layers import Input, Dense, Reshape, Lambda
from keras.models import Model
import keras
from keras.regularizers import l1
from keras import regularizers
from PIL import Image
from keras.callbacks import TensorBoard

from func.var_selection import variable_selection, active_subspace, disjoint_subspace, disjoint_selection, additive
from func.function_class import function


def MinMax(x, weight):
    length = len(x)
    min = 0
    max = 0
    weight = weight.flatten()
    for i in range(length):
        if weight[i] >= 0:
            max += weight[i]
        else:
            min += weight[i]
    return min, max

def plot_first_level(func, func_d, grid, fig_name, plot_var):
    num_group = func_d
    test_r2 = []
    for ind in range(func_d):
        # disjoint_group = disjoint_selection(func.obj, grid_var, init_index=[ind], grid_length = 0.01)
        disjoint_group = [[ind]]+[[i] for i in range(func_d) if i != ind]

        x_val = np.random.uniform(0, 1, size = [int(1e4), func_d])
        #v_val = backward_trans(func.obj, x_val, disjoint_group, eig_vec)
        y_val = func.obj(x_val)

        x_test = lhs(func_d, samples=int(5000), criterion = 'm')
        #v_test = backward_trans(func.obj, x_test, disjoint_group, eig_vec)
        y_test = func.obj(x_test)

        # LHD total for training set and set boundary to the extreme points.
        train_x = func.sparse_grid(grid)
        #v_train = backward_trans(func.obj, train_x, disjoint_group, eig_vec)
        train_y = func.obj(train_x)

        var_index = sum(disjoint_group, [])

        train_x = train_x[:, var_index]
        x_val = x_val[:,var_index]
        x_test = x_test[:,var_index]

        plot_level = 1
        # regularization=1e-5
        regularization = 0.
        exploration = func.IANN_model(disjoint_group, plot_level = plot_level, regularization_rate = regularization,\
                                        activation = 'relu', base_node = 256)
        exploration = func.exploration_train(exploration, train_x, train_y, x_val, y_val,\
                                            BATCH_SIZE = 128, lr = 5e-3,\
                                            EPOCHS = 1000)

        # Fit a linear regression model

        y_pred = exploration.predict(train_x).flatten()
        y_train = train_y.flatten()
        r2 = 1 - np.var(y_pred - y_train)/ np.var(y_train)
        print('the training r^2 is', r2)

        y_pred = exploration.predict(x_test).flatten()
        y_test = y_test.flatten()
        r2 = 1 - np.var(y_pred - y_test)/ np.var(y_test)
        print('the test r^2 is', r2)



        lr_model = LinearRegression()
        lr_model.fit(train_x, train_y)
        y_pred_lr = lr_model.predict(x_test).flatten()
        lr_r2 = 1 - np.var(y_pred_lr - y_test)/np.var(y_test)
        print('linear model has test r^2', lr_r2)

        test_r2 += [r2]
        print('The test r2 is %.4f' % r2)

        # Get weights in linear transformation
        weights = []
        for i in range(plot_level):
            layer = exploration.get_layer('linear_'+str(i))
            weights += [layer.get_weights()]

        print(weights)

        v_min = []; v_max = []
        for i in range(plot_level):
            if len(weights[i]) > 0:
                Min, Max = MinMax(disjoint_group[i], weights[i][0])
                v_min += [Min]
                v_max += [Max]
            else:
                index = disjoint_group[i][0]
                v_min += [func.lb[index]]
                v_max += [func.ub[index]]


        #evaulation#
        num_y = func.n
        num_x = func.N1
        name = fig_name + str(ind+1)
        nonadd_id = [i for i in range(func.d) if i not in func.add_id]
        
        stage = 1

        # 2d path grids for first stage visualization 
        grid_x=func.generate_grid(len(disjoint_group[stage-1]), num_x) # Take LHD in the x variable's space
        grid_y=func.generate_grid(func_d-len(disjoint_group[stage-1]), num_y) # Take LHD in the y variable's space
        grid_y=func.sparse_grid(grid_y) # Take LHD in the y variable's space
        # Generate Uniform Grid on v_j axis
        if 1:
            i = stage - 1
            if len(weights[i]) > 0:
                weight = weights[i][0]
                c = np.zeros(len(disjoint_group[stage-1])).astype(np.double)
                A = np.array(weight).reshape((1,-1))
                for j in range(num_x):
                    v_value = v_min[i] + j*(v_max[i] - v_min[i]) / num_x
                    b = np.array(v_value).astype(np.double)
                    res = linprog(c, A_eq=A, b_eq=b, bounds=[0, 1])
                    grid_x[j, :] = res.x
        x_plot = func.backward_path_grid(grid_x, grid_y, disjoint_group[stage-1])
        #x_plot = np.multiply(x_plot, np.array(func.ub)-np.array(func.lb)) + np.array(func.lb)
        y_plot = func.obj(x_plot)
        # y_plot = np.reshape(exploration.predict(x_plot), (-1,1))
        
        '''
        length = len(disjoint_group[stage-1])
        if length > 1:
            weight = weights[stage-1][0]
            dx = np.arange(100)/100
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_ylabel('$v$'+str(stage))
            for i in range(length):
                v = dx * weight[i]
                index = disjoint_group[stage-1][i]
                x0 = dx #* (func.ub[i] - func.lb[i]) + func.lb[i]
                x0 = x0.flatten()
                plt.plot(dx, v, label = 'x_'+str(index+1),)

            plt.legend(loc = 'upper right')
            file_path = os.getcwd() + '\\image\\'
            plt.savefig(file_path+name+str(stage)+'_linear.png', dpi = 100)
            plt.show()
        '''
        # length = len(disjoint_group[stage-1])
        # row_labels = ['x' + str(nonadd_id[e]+1) for e in disjoint_group[stage-1]]
        # df = pd.DataFrame({'coeff': [None for i in range(length)],
        #                 'lower bound': [None for i in range(length)],
        #                 'upper bound': [None for i in range(length)]},
        #                 index = row_labels)
        # if length == 1:
        #     i = disjoint_group[stage-1][0]
        #     table_i = nonadd_id[i]
        #     df.iloc[0] = [1, func.lb[table_i], func.ub[table_i]] 
        # else:
        #     for i in range(length):
        #         index = disjoint_group[stage-1][i]
        #         table_index = nonadd_id[index]
        #         coeff = weights[stage-1][0].flatten()
        #         df.iloc[i] = [coeff[i].round(decimals = 3), func.lb[table_index], func.ub[table_index]]

        '''
        if stage == num_group-1:
            length2 = len(disjoint_group[stage])
            if length2 == 1:
                i = disjoint_group[stage][0]
                df.iloc[i] = [1, func.lb[i], func.ub[i],'x'+str(i+1)] 
            else:
                for i in range(length2):
                    index = disjoint_group[stage][i]
                    coeff = weights[stage][0].flatten()
                    df.iloc[index] = [coeff[i], func.lb[index], func.ub[index], 'v'+str(stage+1)]
        '''        
        x_plot_reorder = x_plot[:, var_index]

        func.first_stage_plot(exploration, plot_level, stage, x_plot_reorder, y_plot, disjoint_group,\
                name, num_x, num_y, nonadd_id, plot_var, r2,)

    return test_r2, exploration

if __name__ == '__main__':

    def func_obj(x_input): return np.mean(x_input, axis = 1)

    dim = 5
    lb, ub = [-0.1, -0.1, -0.4, 0, 0], [1.5, 0.4, 0.4, 1, 1]
    fig_name = 'doe_'
    func_dict = {'func_obj': func_obj, 'lb': lb, 'ub': ub}

    func_dict['dim'] = dim
    func_dict['original dimension'] = dim
    func_dict['additive_index'] = []

    func= function(func_dict)

    grid = lhs(dim, samples=200, criterion = 'm')
    fig_name = 'test'
    plot_var = [i for i in range(dim)], ['x_'+str(i+1) for i in range(dim)]

    test_r2, y_pred, y_test = plot_first_level(func, dim, grid, fig_name, plot_var)