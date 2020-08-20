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
from module import iann_class
import sklearn.preprocessing as prep
from var_selection import variable_selection, active_subspace
from numpy import linalg as LA
from test_function import test
import os

from pyDOE import *

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


class IANNModel():   
    def __init__(self, func_obj, dim, lb, ub, name = 'iann_',\
                    l2_reg = 0., base_node=16, BATCH_SIZE = 32,\
                    lr = 5e-2, EPOCHS = 50):
        self.make_dir()
        self.dim = 5
        self.lb = lb
        self.ub = ub

        #lb, ub = bound(dim)
        func_dict = {'func_obj': func_obj, 'lb': self.lb, 'ub': self.ub}
        func_dict['dim'] = self.dim
        func_dict['original dimension'] = self.dim
        func_dict['additive_index'] = []
        self.func= iann_class(func_dict)

        self.name = name
        self.regularization = l2_reg
        self.base_node = base_node
        self.BATCH_SIZE = BATCH_SIZE
        self.lr = lr
        self.EPOCHS = EPOCHS

        # generate training set and test set
        self.var_select()
        self.X_train, self.y_train, self.X_test, self.y_test = self.train_and_test()
        self.exploration = self.model()
        #func.N1 = 50
        #func.n = 200

    def var_select(self, samples = 100):
        #Variable Selection
        grid_var = lhs(self.dim, samples=samples, criterion='m')
        self.var_index = variable_selection(self.func.obj, grid_var, N_j = 50, grid_length = 0.01)
        print(self.var_index)
        return self.var_index
        #var_index = var_index[::-1]

    def train_and_test(self):
        self.grid = lhs(self.dim, samples=np.int(1000), criterion = 'm')

        X_test = np.random.uniform(0, 1, size = [int(1e6), self.dim])
        y_test = self.func.obj(X_test)

        # LHD total for training set and set boundary to the extreme points.
        X_train = self.func.sparse_grid(self.grid)
        y_train = self.func.obj(X_train)
        return X_train, y_train, X_test, y_test

        '''
        train_x = np.multiply(train_x, np.array(func.ub)-np.array(func.lb)) + np.array(func.lb)
        x_val = np.multiply(x_val, np.array(func.ub)-np.array(func.lb)) + np.array(func.lb)
        x_test = np.multiply(x_test, np.array(func.ub)-np.array(func.lb)) + np.array(func.lb)
        '''


    def model(self):
        # Construct the exploration neural network model 
        #xploration = func.active_subspace_model(regularization_rate = regularization,\
        #                                 activation = 'sigmoid', node = 16, skip = False)
        exploration = self.func.exploration_model(regularization_rate = self.regularization,\
                                        activation = 'relu', base_node = self.base_node)
        return exploration

    def fit(self):
        train_x = self.X_train[:, self.var_index]
        test_x = self.X_test[:,self.var_index]
        y_train = self.y_train
        self.exploration = self.func.exploration_train(self.exploration, train_x, y_train,\
                                            BATCH_SIZE = self.BATCH_SIZE, lr = self.lr,\
                                            EPOCHS = self.EPOCHS)
        #regularization_rate = [1e-3*np.power(.1, n) for n in range(6)]
        """
        regularization_rate = [ 1e-3, 1e-5, 1e-7, 1e-10, 1e-12, 0]
        train_score = []; test_score = []
        for regularization in regularization_rate:
            train_rep = test_rep = []
            for rep in range(1):
                exploration = func.exploration_model(regularization_rate = regularization,\
                                                activation = 'relu', base_node = 16)
                exploration = func.exploration_train(exploration, train_x, train_y, x_test, y_test,\
                                                    BATCH_SIZE = 32, lr = 5e-3,\
                                                    EPOCHS = 50)
                test_rep += [exploration.evaluate(x_test, y_test, verbose = 0)]
                del exploration

            train_score += [np.mean(train_rep)]
            #print('the training error is %e' % train_score[-1])
            test_score += [np.mean(test_rep)]
            #print('the test error is %e' % test_score[-1])

        CV_IND = np.argmin(test_score)
        regularization = regularization_rate[CV_IND]
        """
    def score(self):
        test_x = self.X_test[:,self.var_index]
        y_pred = self.exploration.predict(test_x).flatten()
        y_test = self.y_test.flatten()
        r2 = 1 - np.var(y_pred - y_test) / np.var(y_test)
        print('the test r^2 is', r2)
        return r2

    def make_dir(self):
        cur_dir = os.getcwd()
        folder_name = 'image'
        folder = os.path.join(cur_dir, folder_name)
        if not os.path.exists(folder):
            os.mkdir(folder)
        folder_name = 'tmp'
        folder = os.path.join(folder, folder_name)
        if not os.path.exists(folder):
            os.mkdir(folder)

    def IANNPlot(self):
        # Make IANN Visualization Plots
        num_y = self.func.n
        num_x = self.func.N1
        disjoint_group = [[self.var_index[i]] for i in range(self.dim)]
        grid_mds = self.func.generate_grid(self.dim-1, self.func.n)
        nonadd_id = list(range(self.dim))
        for stage in np.arange(1, self.dim):

            # 2d path grids for first stage visualization 
            x_plot = self.func.input_path_grid(grid_mds, 0, self.func.n,  self.var_index[stage-1])
            #x_plot = np.multiply(x_plot, np.array(func.ub)-np.array(func.lb)) + np.array(func.lb)
            y_plot = self.func.obj(x_plot)
            x_plot = x_plot[:, self.var_index]

            h_show = self.func.stage_plot(stage, x_plot, y_plot, self.var_index[stage-1], self.var_index, self.name, self.exploration, nonadd_id)

            # h_show = func.backward_stage_plot(stage, x_plot, y_plot, disjoint_group,\
            #          name, exploration, num_x, num_y, list(range(dim)))
        #func.compound_img(x,  h, index, autoencoder, x_test = x_train, h_num = 5, name = 'sin4.jpg', plot = 2)
        print('concatenate the plots')
        self.func.exp_plot_in_one(self.name, self.dim-1, 1)

        # r^2 0.9994931951672645

