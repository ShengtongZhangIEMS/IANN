"""
Created on Mon Oct 22 15:29:15 2018

@author: Shengtong Zhang
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import scipy as sp
import sys
from copy import copy
from sklearn.datasets import load_digits
from sklearn.manifold import Isomap
import sklearn 

from sklearn.gaussian_process.kernels import RBF
from numpy import linalg as LA
from scipy.spatial import distance_matrix

import random, math
from pyDOE import *
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os
import tensorflow as tf
import sklearn.preprocessing as prep

from keras.layers import Input, Dense, Reshape, Lambda
from keras.models import Model
import keras
from keras.regularizers import l1
from keras import regularizers
from PIL import Image
from keras.callbacks import TensorBoard
from test_function import test
from var_selection import variable_selection, active_subspace, disjoint_subspace, disjoint_selection, additive

from matplotlib.colors import LightSource

def rosenbrock(x):
    # compute the rosenbrock function in each row of x;
    # the input x is an 2-d array;
    # the output is an 1-d array with function values.
    y = x
    return np.sum(np.square(y[:, 1:] - np.square(y[:, :-1]))*100\
                  + np.square((1 - y[:, :-1])), axis = 1)

def func1(x):
    return np.power((x[:,0] - 1), 4) + np.sum((np.exp(x[:, 1:]) - 1) / (np.exp(x[:, 1:]) + 1), axis = 1)

def func2(x):
    return np.power((x[:,0] + np.square(x[:, 1]) - 1), 4) + \
           np.sum(np.exp(x[:, 1:] - 1) / np.exp(x[:, 1:] + 1))

def quadratic(x):
    f1 = np.multiply(np.square(x[:,0] - 0.5), np.square(np.sum(x[:, 1:4], axis = 1)))
    #f1 =  np.square(np.sum(x[:, :3], axis = 1))
    f = np.multiply(f1, (np.sum(x[:, 4:], axis = 1)))
    return f

def sin(A, x, b):
    return np.sin(A * x[:,0] + np.dot(x[:, 1:], b))

def ackley(x):
    # compute the ackley function in each row of x;
    # the input x is an 2-d array;
    # the output is an 1-d array with function values.
    a = 20; b = 0.2; c = 2 * np.pi
    return -a * np.exp(-b * np.sqrt(np.mean(np.power(x,2),axis = 1, keepdims = True))) \
           - np.exp(np.mean(np.cos(c * x), axis=1, keepdims=True)) + a + np.exp(1)
          
def Rastrigin(x):
    A = 10
    # Rescale the domain into [-5.12, 5.12]^n
    y = (x - 0.5)*10.24
    return np.sum(np.power(y,2) - A * np.cos(2 * np.pi*y), axis = 1)
    

def cmdscale(D):
    """                                                                                       
    Classical multidimensional scaling (MDS)                                                  
                                                                                               
    Parameters                                                                                
    ----------                                                                                
    D : (n, n) array                                                                          
        Symmetric distance matrix.                                                            
                                                                                               
    Returns                                                                                   
    -------                                                                                   
    Y : (n, p) array                                                                          
        Configuration matrix. Each column represents a dimension. Only the                    
        p dimensions corresponding to positive eigenvalues of B are returned.                 
        Note that each dimension is only determined up to an overall sign,                    
        corresponding to a reflection.                                                        
                                                                                               
    e : (n,) array                                                                            
        Eigenvalues of B.                                                                     
                                                                                               
    """
    # Number of points                                                                        
    n = len(D)
 
    # Centering matrix                                                                        
    H = np.eye(n) - np.ones((n, n))/n
 
    # YY^T                                                                                    
    B = -H.dot(D**2).dot(H)/2
 
    # Diagonalize                                                                             
    evals, evecs = np.linalg.eigh(B)
 
    # Sort by eigenvalue in descending order                                                  
    idx   = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:,idx]
 
    # Compute the coordinates using positive-eigenvalued components only                      
    w, = np.where(evals > 0)
    L  = np.diag(np.sqrt(evals[w]))
    V  = evecs[:,w]
    Y  = V.dot(L)
 
    return Y, evals



class iann_class():
    def __init__(self, func_dict):
        #self.obj = obj
        self.components = 1
        self.n = 200
        self.d = func_dict['dim']
        self.lb = func_dict['lb']
        self.ub = func_dict['ub']
        self.func_obj = func_dict['func_obj']
        self.org_d = func_dict['original dimension'] 
        self.add_id = func_dict['additive_index']
        self.b1 = 4.*np.pi
        self.st1 = 0
        self.ed1 = 1
        self.N1 = 50
        self.d1 = (self.ed1 - self.st1) / self.N1
        self.func_name = []
        self.id2 = 0
        self.add_flag = False


        self.dx1 = np.reshape(np.array([i*self.d1 for i in range(self.N1)]), [1, self.N1])
    
        self.b = np.array(np.matrix([random.randint(1, 1) for i in range(self.d - 1)]).transpose())

    def generate_indices(self):
        self.id = random.randint(1, self.d)-1
        return self.id

    def obj(self, x):
        x0 = np.zeros((len(x), self.org_d))
        rm_list = [i for i in range(self.org_d) if i not in self.add_id]
        x0[:, rm_list] = x
        for i in range(len(x0)):
            for j in self.add_id:
                x0[i,j] = 0.5
        f = self.func_obj(x0)
        """
        if self.add_flag:
            x = x0.copy()
            for i in self.add_id:
                for j in range(len(x)):
                    x[j, i] = (0.5*(self.ub[i] - self.lb[i]) + self.lb[i])
        else:
            x = x0
        
        t = test()
        #f = np.square(3*x[:,0] - np.array([mvn.pdf(xi, mean=[0.25, 0.25], cov = 0.05)+mvn.pdf(xi, mean=[0.75, 0.75], cov=0.05) for xi in x[:,1:]]))
        #f = np.prod(x, axis = 1).reshape((-1, 1))
        #f = np.sin(4 * np.pi*x[:,0] + np.sum(x[:, 1:], axis = 1)/2)
        #f = -np.sum(np.power((x-0.5),2), axis = 1)
        #f = np.prod(x[:,1:], axis = 1) / (1 + x[:,0])
        b = [0.2 *i for i in range(1,self.d+1)]
        #f = np.sum(np.multiply(np.square(x-0.5), b), axis = 1)
        f = quadratic(x)
        #xs = np.square(x)
        #f = np.multiply(xs[:,0]+ 0.01*xs[:,1], xs[:,2] + xs[:,3]) + np.sin(5*xs[:,4])
        #f = -np.square(np.sum(x[:,:3]-0.5, axis = 1)) - 1.*np.square(np.sum(x[:,3:]-0.5, axis = 1))
        if self.func_name == 'OTL':
            f, self.lb, self.ub = t.OTL(x)
        elif self.func_name == 'piston':
            f, self.lb, self.ub = t.piston(x)
        elif self.func_name == 'borehole':
            f, self.lb, self.ub = t.borehole(x)
        elif self.func_name == 'Rastrigin':
            f = np.sin(4 * np.pi*x[:,0] + np.sum(x[:, 1:], axis = 1))
            #f = np.square(x[:,0]) + np.square(np.mean(x[:,1:], axis = 1))
            #f = Rastrigin(x)
            self.lb = -5.12*np.ones(self.d)
            self.ub = -self.lb
        elif self.func_name == 'Gaussian':
            mu = np.array([[0.25, 0.25],[0.25, 0.75], [0.75, 0.25],[0.75, 0.75]])
            f = Gaussian(x, mu)
            self.lb = np.ones(self.d)
            self.ub = -self.lb
        """
        return f.reshape((-1, 1))




    def fmod(self, x, y):
        '''
            generate a residual between 0 and y if y > 0;
        '''

        if y==0:
            raise ZeroDivisionError
        else:
            d = x//y
            res = x - d * y
            return res

    def CCS(self, n, d, width, index = 0, x0 = None, 
        insert_value = 0.5):
        '''
        n: the number of samples.
        d: dimension of the samples.
        width: distance between two neighboring grids.
        index: the index of x_j
                here  we choose x_j = 0.5 for function evaluation.
                as long as the gradient is nonzero.
        x0: the starting samples with shape (n0, d).
        '''
        n_sample = 0
        id = 0
        x = np.zeros((n, d))
        if x0 is None:
            n0 = 0
        else:
            n0 = len(x0) #number of starting samples
        while n_sample < n:
            if id >= n0:
                # randomly generate a starting point
                x_st = np.random.uniform(0, 1, d)
                #print(x_st)
                x[n_sample, :] = x_st
                n_sample += 1
            else:
                x_st = x0[id, :]
                x[n_sample, :] = x_st
                n_sample += 1
                
            # find the samples along gradient descent direction
            x_k = x_st
            while n_sample < n: 
                f0 = self.obj(np.reshape(np.insert(x_k, index, insert_value), (1,-1)))
                for i in range(d):
                    delta = np.insert(np.zeros(d-1), i, width)
                    fl = self.obj(np.reshape(np.insert(x_k-delta, index, insert_value), (1,-1)))
                    fr = self.obj(np.reshape(np.insert(x_k + delta, index, insert_value), (1,-1)))
                    # check the descent direction
                    if fl < f0:
                        x_new = x_k - width
                    elif fr < f0:
                        x_new = x_k + width
                    else:
                        x_new = x_k
                if (x_k - x_new).any() and all([0<j<1 for j in x_new]) :
                    # check whether the sample is inside the domain
                    x_k = x_new
                    x[n_sample, :] = x_k
                    n_sample += 1
                else:
                    break
            x_k = x_st
            # find the samples along gradient ascent direction
            while n_sample < n: 
                f0 = self.obj(np.reshape(np.insert(x_k, index, insert_value), (1,-1)))
                for i in range(d):
                    delta = np.insert(np.zeros(d-1), i, width)
                    fl = self.obj(np.reshape(np.insert(x_k-delta, index, insert_value), (1,-1)))
                    fr = self.obj(np.reshape(np.insert(x_k + delta, index, insert_value), (1,-1)))
                    if fl > f0:
                        x_new = x_k - width
                    elif fr > f0:
                        x_new = x_k + width
                    else:
                        x_new = x_k
                if (x_k - x_new).any() and all([0<j<1 for j in x_new]) :

                    # check whether the sample is inside the domain
                    x_k = x_new
                    x[n_sample, :] = x_k
                    n_sample += 1
                else:
                    break
            id += 1
        return x
    
    def generate_grid(self, dim, samples):
        grid = lhs(dim, samples=samples, criterion = 'm')
        return grid
    
    def distance_matrix(self, x, index):
        D = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                #print(np.insert(np.repeat(np.matrix(x[i, :]), self.N1, axis = 0), index, values = self.dx1, axis = 1))
                path1 = self.obj(np.insert(np.repeat(np.matrix(x[i, :]), self.N1, axis = 0), index, values = self.dx1, axis = 1))
                path2 = self.obj(np.insert(np.repeat(np.matrix(x[j, :]), self.N1, axis = 0), index, values = self.dx1, axis = 1))
                exp1 = self.d1 * np.sum(path1) / (self.ed1 - self.st1)
                exp2 = self.d1 * np.sum(path2) / (self.ed1 - self.st1)
                if 1:
                    exp1 = 0; exp2 = 0
                    
                dissimilar = self.d1 * np.sum(np.square((path1 - exp1 - (path2 - exp2))))
                D[i, j] = np.sqrt(dissimilar)
        return D

    def cmdscale(self, D):
        Y, e = cmdscale(D)
        e0 = 1e-6   # Set a threshold for the eigenvalue
        e = e/np.sqrt(np.sum(np.power(e, 2)))
        e_truncate = [i for i in e if i > e0]
        p = len(e_truncate)
        rel_error = 1 - np.power(e[0],2)/np.sum(np.power(e, 2))
        component_error = 1 - np.sum(np.power(e_truncate,2))/np.sum(np.power(e, 2))
        print('the scaled dimension is', p)
        print('the relative error is', rel_error)
        #print(e)

        plot = 0
        if plot:
            if p == 1:
                plt.plot(Y[:, 0], np.zeros(len(Y)), '.')
            elif p == 2:
                plt.plot(Y[:, 0], Y[:, 1], '.')
        return Y[:, list(range(p))], e_truncate, rel_error, component_error, p
    def isomap(self, D, Y, n_neighbors = 10, components = 1):
        embedding = Isomap(n_neighbors=n_neighbors, n_components = components)
        X_transformed = embedding.fit_transform(Y)
        err = embedding.reconstruction_error()
        #print(err)
        if 0:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            if self.components == 1:
                ax.plot(X_transformed[:, 0], np.zeros(len(X_transformed)), '.')
            elif self.components == 2:
                ax.plot(X_transformed[:, 0], X_transformed[:,1], '.')
        return X_transformed, err
     
    def input_data(self, x, index_st, batch_size, index):
        batch_func = np.zeros([batch_size, self.N1])
        for i in range(batch_size):
            path = self.obj(np.insert(np.repeat(np.matrix(x[self.fmod(i+index_st, self.n), :]), self.N1, axis = 0), index, values = self.dx1, axis = 1))
            batch_func[i, :] = np.reshape(path, [1,-1])
        return batch_func
    
    def input_data_model2(self, x_exp, index_st, batch_size):
        print(x_exp.shape)
        batch_x= x_exp[[self.fmod(index_st+i, self.n) for i in range(batch_size)], :]
        return np.matrix(batch_x)

    def input_path_grid(self, x, index_st, batch_size, index):
        batch_grid = np.zeros([batch_size * self.N1, self.d])
        for i in range(batch_size):
            path_grid = np.insert(np.repeat(np.matrix(x[self.fmod(i+index_st, self.n), :]), self.N1, axis = 0), index, values = self.dx1, axis = 1)
            batch_grid[i*self.N1 : (i+1)*self.N1, :] = np.reshape(path_grid, [-1,self.d])
        return np.matrix(batch_grid)

    def backward_path_grid(self, grid_x, grid_y, x_index):
        num_x, dim_x = np.shape(grid_x)
        num_y, dim_y = np.shape(grid_y)
        if len(x_index) != dim_x:
            print('x_grid dimension mismatch with x_index')
        batch_grid = np.zeros([num_x*num_y, self.d])
        for i in range(num_y):
            y_index = [ind for ind in range(self.d) if ind not in x_index]
            path_grid = np.repeat(np.matrix(grid_y[i, :]), num_x, axis = 0)
            batch_grid[i*num_x : (i+1)*num_x, y_index] = np.reshape(path_grid, [-1, dim_y])
            batch_grid[i*num_x : (i+1)*num_x, x_index] = np.reshape(grid_x, [-1, dim_x])
        return np.matrix(batch_grid)

    def sparse_grid(self, x):
        n, m = np.shape(x)
        batch_grid = np.tile(x, (2*m+1,1))
        for i in range(m):
            batch_grid[2*i*n:(2*i+1)*n, i] = 0
            batch_grid[(2*i+1)*n:(2*i+2)*n,i] = 1
        return np.matrix(batch_grid)

        

    def store_image(self, x, h, mds_index, h_num = 5, plot = 0):
        '''
            x: The sample grid;
            h: the low-dimensional reconstruction sample points;
            h_num: number of h-paths we are going to plot;
            plot: a Bool parameter to determine whether store the images of MDS or not.
            
            return h_show: the h-values to present in the images.
        '''
        file_path = os.path.dirname(os.getcwd()) + '/image/'
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        h = h.flatten()
        h_index = np.argsort(h)
        h_sort = h[h_index]
        h_show = h_sort[[int(i * np.floor(len(h) / h_num)) for i in range(h_num)]]
        if plot:
            for i in range(h_num):
                plot_y = np.reshape(self.obj(np.insert(np.repeat(np.matrix(x[h_index[int(i * np.floor(len(h) / h_num))], :]),\
                                                            self.N1, axis = 0), mds_index, values = self.dx1, axis = 1)), [-1])
                plot_y = plot_y.flatten()
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.set_xlabel('$x_j$')
                ax.set_ylabel('f')
                plot_y = np.reshape(self.obj(np.insert(np.repeat(np.matrix(x[h_index[int(i * np.floor(len(h) / h_num))], :]),\
                                                            self.N1, axis = 0), mds_index, values = self.dx1, axis = 1)), [-1])
                plot_y = plot_y.flatten()
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.set_xlabel('$x_j$')
                ax.set_ylabel('f')
                ax.set_title('two dimensional function image')
                label = ['h = '+str(h_show[i])]
                plot_x = np.reshape(self.dx1, plot_y.shape)
                ax.plot(plot_x, plot_y, 'r-o')
                save_name = os.path.dirname(os.getcwd()) + '/image/' + str(i+1) + '.jpg'
                plt.legend(label, loc=0, ncol=2)
                #plt.show()
                plt.savefig(save_name)
                plt.close()
                
        #print(h_show)
        return h_show

    def autoencoder_model(self, bottle_neck_dim = 1, regularization_rate = 0.01,\
                          activation='sigmoid'):
        #     Construct auto-encoder with multiple layers
        n_input = self.N1
        input_path = Input(shape=(n_input,))
        encoded = Dense(64, activation=activation, kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate),
                        activity_regularizer=l1(0.))(input_path)
        encoded = Dense(8, activation=activation, kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate),
                        activity_regularizer=l1(0.))(encoded)
        encoded = Dense(bottle_neck_dim, activation='linear', kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate))(encoded)
        decoded = Dense(8, activation=activation, kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate),
                        activity_regularizer=l1(0.))(encoded)
        decoded = Dense(64, activation=activation, kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate),
                        activity_regularizer=l1(0.0))(decoded)
        decoded = Dense(n_input, activation='linear', kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate))(decoded)
        autoencoder = Model(input=input_path, output=decoded)

        return autoencoder
    
    def autoencoder_train(self, autoencoder, x_train, x_test, lr = 0.01, BATCH_SIZE = 5, EPOCHS = 100):
        adam = keras.optimizers.Adam(lr=lr) #lr := learning rate
        autoencoder.compile(optimizer=adam, loss='mean_squared_error')
        #normalization
        '''
        x_train -= np.mean(x_train, axis = 1).reshape([-1, 1])
        x_train = prep.normalize(x_train, axis = 1)
        
        
        x_test -= np.mean(x_train, axis = 1).reshape([-1, 1])
        x_test = prep.normalize(x_test, axis = 1)
        '''
        tbCallBack = TensorBoard(log_dir='.\logs',  # log 目录
                         histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
        #                  batch_size=32,     # 用多大量的数据计算直方图
                         write_graph=True,  # 是否存储网络结构图
                         write_grads=True, # 是否可视化梯度直方图
                         write_images=True,# 是否可视化参数
                         embeddings_freq=0, 
                         embeddings_layer_names=None, 
                         embeddings_metadata=None)
        
        autoencoder.fit(x_train,x_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        verbose = 2,
                        validation_data=(x_test,x_test),
                        callbacks=[tbCallBack])       
        return autoencoder
    
    def autoencoder_evaluation(self, h, autoencoder, bottle_neck_dim = 0):
        '''
            evaluation: the evaluation model where the bottle neck layer is the input.
        '''
        
        # feed h-values into the bottle neck layer. It requires a new computation graph
        bottle_neck_input = Input(shape=(bottle_neck_dim,))
        h_decoded = bottle_neck_input
        for layer in autoencoder.layers[4:]:
            h_decoded = layer(h_decoded)
            
        #Create the h-value model
        evaluation = Model(input = bottle_neck_input, output = h_decoded)    
        h_path = evaluation.predict(h)
        return h_path
    
    def autoencoder_encode(self, autoencoder, x_train, h_id = 0):
        '''
            evaluation:
            the evaluation model for encode part where the bottle neck layer is the output.
            h_id: the index of h-nodes in the visualization
        '''
        
        # feed h-values into the bottle neck layer. It requires a new computation graph
        x_input = Input(shape=(self.N1,))
        encoded = x_input
        for layer in autoencoder.layers[:4]:
            encoded = layer(encoded)
            
        #Create the h-value model
        h_encode = Model(input = x_input, output =encoded)    
        h = h_encode.predict(x_train)
        return h[:,h_id]
    
    
    def compound_img(self, x,  h, mds_index, autoencoder, x_test, h_num = 5, plot = 0):
        '''
            x: The sample grid;
            h: the low-dimensional reconstruautoencoder_patction sample points;
            h_num: number of h-paths we are going to plot;
            plot: a integer parameter to store the images of visualization
                    generated from MDS and autoencoder.
            
            return h_show: the h-values to present in the images.
        '''
        file_path = os.getcwd() + '/compare_image/'
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        h = h.flatten()
        h_index = np.argsort(h)
        h_sort = h[h_index]
        h_show = h_sort[[int(i * np.floor(len(h) / h_num)) for i in range(h_num)]]
        '''
        mean = np.mean(x_test, axis = 1).reshape([-1, 1])
        mean_x = x_test - mean
        x_test, norms = prep.normalize(mean_x, axis = 1, return_norm = True)
        '''
        if plot == 3:
            x_discrete = self.dx1 * (self.ub[mds_index] - self.lb[mds_index]) + self.lb[mds_index]
            x_grid_2d, h_grid_2d = np.meshgrid(x_discrete, h)
            #plt.axis([0,1,-1,1])
            fig = plt.figure()
            
            ######################################################
            #   plot MDS
            ######################################################
            
            ax1 = fig.add_subplot(111, projection = '3d')
            ax1.set_xlabel('$x_j$')
            ax1.set_ylabel('h')
            ax1.set_zlabel('f')
            ax1.set_title('MDS')
            for i in range(len(h)):
                path = np.reshape(np.array(self.obj(np.insert(np.repeat(np.matrix(x[i, :]),\
                                                            self.N1, axis = 0), mds_index, values = self.dx1, axis = 1))), [1, -1])
                if i == 0:
                    z_mds = path
                else:
                    z_mds = np.concatenate((z_mds, path), axis = 0)
            
            ######################################################
            #   plot auto-encoder
            ###################################################### 

            h_auto = self.autoencoder_encode(autoencoder, x_test,).flatten()
            x_auto_2d, h_auto_2d = np.meshgrid(x_discrete, h_auto)
            
            #ax2 = fig.add_subplot(122, projection = '3d')
            fig = plt.figure()
            ax2 = fig.add_subplot(111, projection = '3d')
            ax2.set_xlabel('$x_j$')
            ax2.set_ylabel('h')
            ax2.set_zlabel('f')
            ax2.set_title('auto-encoder')
            
            
            
            #autoencoder_path = autoencoder.predict(x_test)
            #z_auto = np.multiply(np.transpose(x_test), norms).transpose() + mean
            z_auto = x_test
            #print(x_test.shape)
            
            dx1 = self.dx1.flatten()
            spline_mds = sp.interpolate.interp2d(dx1,h,z_mds, kind = 'cubic')
            spline_auto = sp.interpolate.interp2d(dx1,h_auto,z_auto, kind = 'cubic')

            #z_mds = spline_mds(dx1, h)
            #z_auto = spline_auto(dx1, h_auto)
            
            cmap = matplotlib.colors.ListedColormap("red")            
            ax1.plot_surface(x_grid_2d, h_grid_2d, z_mds, cmap = cm.coolwarm,
                 rstride = 1, cstride = 1, antialiased = True)
            save_name = os.getcwd() + '/img/OTL_auto_' + str(mds_index)+'.png'
            #plt.savefig(save_name)
            ax2.plot_surface(x_auto_2d, h_auto_2d, z_auto, cmap = cm.coolwarm,
                 rstride = 1, cstride = 1, antialiased = True)
            save_name = os.getcwd() + '/img/OTL_auto_' + str(mds_index)+'.png'
            #plt.savefig(save_name)

            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xlabel('h_mds')
            ax.set_ylabel('h_auto')
            ax.plot(np.sort(h), np.sort(h_auto))
            
            plt.show()
            #plt.close()
            h_show = h_auto
        if plot == 2:
            fig = plt.figure()
            
            ######################################################
            #   plot MDS
            ######################################################
            ax1 = fig.add_subplot(121)
            ax1.set_xlabel('$x_j$')
            ax1.set_ylabel('f')
            ax1.set_title('MDS')
            
            for i in range(h_num):
                plot_y = np.reshape(np.array(self.obj(np.insert(np.repeat(np.matrix(x[h_index[int(i * np.floor(len(h) / h_num))], :]),\
                                                            self.N1, axis = 0), mds_index, values = self.dx1, axis = 1))), [-1])
                plot_x = np.reshape(self.dx1, plot_y.shape)
                ax1.plot(plot_x, plot_y, '-')
                
            ######################################################
            #   plot auto-encoder
            ######################################################   
            ax2 = fig.add_subplot(122)
            ax2.set_xlabel('$x_j$')
            ax2.set_ylabel('f')
            ax2.set_title('auto-encoder')
            
            autoencoder_path = autoencoder.predict(x_test)
            z_auto = autoencoder_path
            #print(z_auto)
            #z_auto = np.multiply(np.transpose(autoencoder_path), norms).transpose() + mean
            #print(z_auto)
            for i in range(h_num):
                plot_y = np.reshape(np.array(self.obj(np.insert(np.repeat(np.matrix(x[h_index[int(i * np.floor(len(h) / h_num))], :]),\
                                                            self.N1, axis = 0), mds_index, values = self.dx1, axis = 1))), [-1])
                ax2.plot(plot_x, z_auto[h_index[int(i * np.floor(len(h) / h_num))],:], '-r')
                ax2.plot(plot_x, plot_y, '-b')
            
            save_name = os.getcwd() + '/compare_image/' + name
            plt.legend(loc=0, ncol=2)
            plt.savefig(save_name)
            plt.show()
            plt.close()
            h_show = z_auto
        #print(h_show)
        return h_show

    def plot_all(self, x,  h, mds_index, autoencoder, head, x_test, name, h_num = 5):
        '''
            x: The sample grid;
            h: the low-dimensional reconstruautoencoder_patction sample points;
            h_num: number of h-paths we are going to plot;
            plot: a integer parameter to store the images of visualization
                    generated from MDS and autoencoder;
            head: title of the image.
            
            return h_show: the h-values to present in the images.
        '''
        file_path = os.getcwd() + '/img/'
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        h = h.flatten()
        h_index = np.argsort(h)
        h_sort = h[h_index]
        h_show = h_sort[[int(i * np.floor(len(h) / h_num)) for i in range(h_num)]]
        '''
        mean = np.mean(x_test, axis = 1).reshape([-1, 1])
        mean_x = x_test - mean
        x_test, norms = prep.normalize(mean_x, axis = 1, return_norm = True)
        '''
        x_discrete = self.dx1 * (self.ub[mds_index] - self.lb[mds_index]) + self.lb[mds_index]
        x_grid_2d, h_grid_2d = np.meshgrid(x_discrete, h)
        #plt.axis([0,1,-1,1])
        fig = plt.figure()
        
        ######################################################
        #   plot MDS
        ######################################################
        
        ax1 = fig.add_subplot(111, projection = '3d')
        for i in range(len(h)):
            path = np.reshape(np.array(self.obj(np.insert(np.repeat(np.matrix(x[i, :]),\
                                                        self.N1, axis = 0), mds_index, values = self.dx1, axis = 1))), [1, -1])
            if i == 0:
                z_mds = path
            else:
                z_mds = np.concatenate((z_mds, path), axis = 0)
        
        ######################################################
        #   plot auto-encoder
        ###################################################### 

        h_auto = self.autoencoder_encode(autoencoder, x_test,).flatten()
        x_auto_2d, h_auto_2d = np.meshgrid(x_discrete, h_auto)    
        
        #autoencoder_path = autoencoder.predict(x_test)
        #z_auto = np.multiply(np.transpose(x_test), norms).transpose() + mean
        z_auto = x_test
        #print(x_test.shape)
        
        dx1 = self.dx1.flatten()
        spline_mds = sp.interpolate.interp2d(dx1,h,z_mds, kind = 'cubic')
        spline_auto = sp.interpolate.interp2d(dx1,h_auto,z_auto, kind = 'cubic')

        #z_mds = spline_mds(dx1, h)
        #z_auto = spline_auto(dx1, h_auto)
                
        cmap = matplotlib.colors.ListedColormap("red")            
        ax1.plot_surface(x_grid_2d, h_grid_2d, z_mds, cmap = cm.coolwarm,
             rstride = 1, cstride = 1, antialiased = True)
        ax1.set_xlabel(head[mds_index])
        ax1.set_ylabel('h')
        ax1.set_zlabel('f')
        plt.title('MDS for ' + head[mds_index])
        
        save_name_mds = file_path + name+'_mds_' + str(mds_index)+'.png'
        plt.savefig(save_name_mds, dpi=100)

        
        fig = plt.figure()
        ax2 = fig.add_subplot(111, projection = '3d')
        
        
        ax2.plot_surface(x_auto_2d, h_auto_2d, z_auto, cmap = cm.coolwarm,
             rstride = 1, cstride = 1, antialiased = True)
        ax2.set_xlabel(head[mds_index])
        ax2.set_ylabel('h')
        ax2.set_zlabel('f')
        plt.title('auto-encoder for ' + head[mds_index])
        save_name = file_path + name+'_auto_' + str(mds_index)+'.png'
        plt.savefig(save_name, dpi=100)
        plt.show()
        plt.close()
        
        h_show = h_auto
        return h_show

    def plot_in_one(self, name, row = 2, column = 3):
        file_path = os.getcwd() + '/img/'
        count = 0
        pic_mds = Image.open(file_path+name+'_mds_'+str(count)+'.png')
        width, height = pic_mds.size
        toImage_mds = Image.new('RGBA', (column*width, row*height))
        toImage_auto = Image.new('RGBA', (column*width, row*height))
        for i in range(row):
            for j in range(column):
                pic_mds = Image.open(file_path+name+'_mds_'+str(count)+'.png')
                pic_auto = Image.open(file_path+name+'_auto_'+str(count)+'.png')
                #print(file_path+name+'_auto_'+str(count)+'.png')
                toImage_mds.paste(pic_mds, (j*width, i*height))
                toImage_auto.paste(pic_auto, (j*width, i*height))
                count += 1
                if count >= self.d:
                    break
        toImage_mds.save(file_path+name+'_mds.png')
        toImage_auto.save(file_path+name+'_auto.png')
    
    '''    
    def exploration_model(self, bottle_neck_dim = 1,  regularization_rate = 0.,\
                          activation = 'sigmoid'):
        def slice(x, st, ed):
            return x[:,st:ed]
        #     Construct auto-encoder with multiple layers
        n_input = self.d
        input_path = Input(shape=(n_input,))
        input0 = Lambda(slice,output_shape=(1,),arguments={'st':0, 'ed':1}  )(input_path)
        input_rs = Lambda(slice,output_shape=(n_input-1,),arguments={'st':1, 'ed':n_input})(input_path)
        input0 = Reshape((1,))(input0)
        input_rs = Reshape((n_input-1,))(input_rs)
        encoded = Dense(256, activation=activation, kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate))(input_rs)
        encoded = Dense(32, activation=activation, kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate))(encoded)
        encoded = Dense(bottle_neck_dim, activation='linear', kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate))(encoded)
        bottle_neck = keras.layers.concatenate([input0, encoded], axis = 1)
        decoded = Dense(256, activation=activation, kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate))(bottle_neck)
        decoded = Dense(32, activation=activation, kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate))(decoded)
        decoded = Dense(1, activation='linear', kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate))(decoded)
        exploration = Model(input=input_path, output=decoded)

        return exploration    
    '''
    def exploration_model(self, regularization_rate = 0.,\
                          activation = 'sigmoid', base_node = 8):
        def slice(x, st, ed):
            return x[:,st:ed]
        #     Construct auto-encoder with multiple layers
        n_input = self.d
        input_path = Input(shape=(n_input,))
        input_node = {}
        for i in range(n_input):
            input0 = Lambda(slice,output_shape=(1,),arguments={'st':n_input-i-1, 'ed':n_input-i})(input_path)
            #input_rs = Lambda(slice,output_shape=(n_input-i-1,),arguments={'st':0, 'ed':n_input-i-1})(input_path)
            input0 = Reshape((1,))(input0)
            #input_rs = Reshape((n_input-i-1,))(input_rs)
            input_node['xj'+str(n_input-i-1)] = input0
        bottle_neck = keras.layers.concatenate([input_node['xj'+str(n_input-2)], input_node['xj'+str(n_input-1)]], axis = 1)
        for i in range(n_input-2):
            encoded = Dense(base_node*(i+1), activation=activation, kernel_initializer = 'random_normal',
                            kernel_regularizer=regularizers.l2(regularization_rate))(bottle_neck)
            encoded = Dense(1, activation='linear', kernel_initializer = 'random_normal',
                            kernel_regularizer=regularizers.l2(regularization_rate))(encoded)
            bottle_neck = keras.layers.concatenate([input_node['xj'+str(n_input-i-3)], encoded], axis = 1)

            
        decoded = Dense(base_node*(n_input-1), activation=activation, kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate))(bottle_neck)
        decoded = Dense(1, activation='linear', kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate))(decoded)
        exploration = Model(input=input_path, output=decoded)

        return exploration

    def exploration_model_additive(self, regularization_rate = 0.,\
                          activation = 'sigmoid', base_node = 8):
        def slice(x, st, ed):
            return x[:,st:ed]
        #     Construct auto-encoder with multiple layers
        n_input = self.d
        input_path = Input(shape=(n_input,))
        input_node = {}
        for i in range(n_input):
            input0 = Lambda(slice,output_shape=(1,),arguments={'st':n_input-i-1, 'ed':n_input-i})(input_path)
            #input_rs = Lambda(slice,output_shape=(n_input-i-1,),arguments={'st':0, 'ed':n_input-i-1})(input_path)
            input0 = Reshape((1,))(input0)
            #input_rs = Reshape((n_input-i-1,))(input_rs)
            input_node['xj'+str(n_input-i-1)] = input0
        bottle1 = input_node['xj'+str(n_input-2)]
        bottle2 = input_node['xj'+str(n_input-1)]
        for i in range(n_input-2):
            encoded1 = Dense(base_node*(i+1), activation=activation, kernel_initializer = 'random_normal',
                            kernel_regularizer=regularizers.l2(regularization_rate))(bottle1)
            encoded1 = Dense(1, activation='linear', kernel_initializer = 'random_normal',
                            kernel_regularizer=regularizers.l2(regularization_rate))(encoded1)
            encoded2 = Dense(base_node*(i+1), activation=activation, kernel_initializer = 'random_normal',
                            kernel_regularizer=regularizers.l2(regularization_rate))(bottle2)
            encoded2 = Dense(1, activation='linear', kernel_initializer = 'random_normal',
                            kernel_regularizer=regularizers.l2(regularization_rate))(encoded2)
            encoded = keras.layers.Add()([encoded1, encoded2])
            bottle1 = input_node['xj'+str(n_input-i-3)]
            bottle2 = encoded
            bottle_neck = keras.layers.concatenate([bottle1, encoded], axis = 1)

            
        decoded = Dense(base_node*(n_input-1), activation=activation, kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate))(bottle_neck)
        decoded = Dense(1, activation='linear', kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate))(decoded)
        exploration = Model(input=input_path, output=decoded)

        return exploration

    def active_subspace_model(self, regularization_rate = 0.,\
                          activation = 'sigmoid', node = 16, skip = False):
        def slice(x, st, ed):
            return x[:,st:ed]
        #     Construct auto-encoder with multiple layers
        n_input = self.d
        input_path = Input(shape=(n_input,))
        if skip:
            encoded = Dense(node, activation=activation, kernel_initializer = 'random_normal',
                            kernel_regularizer=regularizers.l2(regularization_rate))(input_path)
        else:
            encoded = Dense(node, activation='linear', kernel_initializer = 'random_normal',
                            kernel_regularizer=regularizers.l2(regularization_rate))(input_path)
        encoded = Dense(2, activation='linear', kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate))(encoded)
        #bottle_neck = keras.layers.concatenate([input_node['xj'+str(n_input-i-3)], encoded], axis = 1)

            
        decoded = Dense(node*2, activation=activation, kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate))(encoded)
        decoded = Dense(1, activation='linear', kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate))(decoded)
        exploration = Model(input=input_path, output=decoded)

        return exploration
    
    def backward_subspace_model(self, disjoint_group, regularization_rate = 0.,\
                          activation = 'sigmoid', base_node = 8):

        def slice(x, st, ed):
            return x[:,st:ed]
        #     Construct auto-encoder with multiple layers
        n_input = self.d
        num_group = len(disjoint_group)
        input_path = Input(shape=(n_input,))
        if num_group == 1:
            linear_node = Dense(1, activation='linear', use_bias = False, kernel_initializer = 'random_normal',
                            kernel_regularizer=regularizers.l2(0.), name = 'linear_0')(input_path)
            hidden=Dense(base_node, activation=activation, kernel_initializer = 'random_normal',
                                kernel_regularizer=regularizers.l2(regularization_rate), name = 'dense_0')(linear_node)
            output = Dense(1, activation='linear', kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate))(hidden)
            exploration = Model(input=input_path, output=output)
            return exploration

        input_node = {}
        tot_len = 0
        for i in range(num_group):
            length = len(disjoint_group[i])
            input0 = Lambda(slice,output_shape=(length,),arguments={'st':tot_len, 'ed':tot_len+length})(input_path)
            #input_rs = Lambda(slice,output_shape=(n_input-i-1,),arguments={'st':0, 'ed':n_input-i-1})(input_path)
            input0 = Reshape((length,))(input0)
            #input_rs = Reshape((n_input-i-1,))(input_rs)
            if length == 1:
                linear_node = Lambda(lambda x: x, name = 'linear_'+str(i))(input0)
            else:
                linear_node = Dense(1, activation='linear', use_bias = False, kernel_initializer = 'random_normal',
                                kernel_regularizer=regularizers.l2(0.), name = 'linear_'+str(i))(input0)
            input_node['group_'+str(i)] = linear_node
            tot_len += length
        bottle_neck = keras.layers.concatenate([input_node['group_'+str(num_group-2)],\
             input_node['group_'+str(num_group-1)]], axis = 1, name = 'con_0')
        for i in range(num_group-2):
            encoded = Dense(base_node*(i+1), activation=activation, kernel_initializer = 'random_normal',
                            kernel_regularizer=regularizers.l2(regularization_rate))(bottle_neck)
            encoded = Dense(1, activation='linear', kernel_initializer = 'random_normal',
                            kernel_regularizer=regularizers.l2(regularization_rate))(encoded)
            bottle_neck = keras.layers.concatenate([input_node['group_'+str(num_group-i-3)], encoded],\
                 axis = 1, name = 'con_' + str(i+1))

            
        decoded = Dense(base_node*(num_group-1), activation=activation, kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate))(bottle_neck)
        if self.func_name == 'harmonic':
            decoded = Dense(128, activation=activation, kernel_initializer = 'random_normal',
                            kernel_regularizer=regularizers.l2(regularization_rate))(decoded)
        decoded = Dense(1, activation='linear', kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate))(decoded)
        exploration = Model(input=input_path, output=decoded)

        return exploration

    def backward_subspace_model_additive(self, disjoint_group, additive_layer, regularization_rate = 0.,\
                          activation = 'sigmoid', base_node = 8,):
        # additive_layer:  d-1 layer in total, count from the last layer/level.
        def slice(x, st, ed):
            return x[:,st:ed]
        #     Construct auto-encoder with multiple layers
        n_input = self.d
        num_group = len(disjoint_group)
        input_path = Input(shape=(n_input,))
        if num_group == 1:
            linear_node = Dense(1, activation='linear', use_bias = False, kernel_initializer = 'random_normal',
                            kernel_regularizer=regularizers.l2(0.), name = 'linear_0')(input_path)
            hidden=Dense(base_node, activation=activation, kernel_initializer = 'random_normal',
                                kernel_regularizer=regularizers.l2(regularization_rate), name = 'dense_0')(linear_node)
            output = Dense(1, activation='linear', kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate))(hidden)
            exploration = Model(input=input_path, output=output)
            return exploration

        input_node = {}
        tot_len = 0
        for i in range(num_group):
            length = len(disjoint_group[i])
            input0 = Lambda(slice,output_shape=(length,),arguments={'st':tot_len, 'ed':tot_len+length})(input_path)
            #input_rs = Lambda(slice,output_shape=(n_input-i-1,),arguments={'st':0, 'ed':n_input-i-1})(input_path)
            input0 = Reshape((length,))(input0)
            #input_rs = Reshape((n_input-i-1,))(input_rs)
            if length == 1:
                linear_node = Lambda(lambda x: x, name = 'linear_'+str(i))(input0)
            else:
                linear_node = Dense(1, activation='linear', use_bias = False, kernel_initializer = 'random_normal',
                                kernel_regularizer=regularizers.l2(0.), name = 'linear_'+str(i))(input0)
            input_node['group_'+str(i)] = linear_node
            tot_len += length

        bottle1 = input_node['group_'+str(num_group-2)]
        bottle2 = input_node['group_'+str(num_group-1)]
        bottle_neck = keras.layers.concatenate([input_node['group_'+str(num_group-2)],\
             input_node['group_'+str(num_group-1)]], axis = 1, name = 'con_0')
        for i in range(num_group-2):
            bottle1 = input_node['group_'+str(num_group-i-3)]
            if additive_layer[i]:
                encoded1 = Dense(base_node*(i+1), activation=activation, kernel_initializer = 'random_normal',
                                kernel_regularizer=regularizers.l2(regularization_rate), name = 'add1_' + str(i))(bottle1)
                encoded1 = Dense(1, activation='linear', kernel_initializer = 'random_normal',
                                kernel_regularizer=regularizers.l2(regularization_rate), name = 'add1_out_' + str(i))(encoded1)
                encoded2 = Dense(base_node*(i+1), activation=activation, kernel_initializer = 'random_normal',
                                kernel_regularizer=regularizers.l2(regularization_rate), name = 'add2_' + str(i))(bottle2)
                encoded2 = Dense(1, activation='linear', kernel_initializer = 'random_normal',
                                kernel_regularizer=regularizers.l2(regularization_rate), name = 'add1_out_' + str(i))(encoded2)
                encoded = keras.layers.Add()([encoded1, encoded2])
            else:
                encoded = Dense(base_node*(i+1), activation=activation, kernel_initializer = 'random_normal',
                                kernel_regularizer=regularizers.l2(regularization_rate), name = 'nonadd_dense_' + str(i))(bottle_neck)
                encoded = Dense(1, activation='linear', kernel_initializer = 'random_normal',
                                kernel_regularizer=regularizers.l2(regularization_rate), name = 'nonadd_dense_out_' + str(i))(encoded)
            bottle_neck = keras.layers.concatenate([bottle1, encoded],\
                axis = 1, name = 'con_' + str(i+1))

            
        decoded = Dense(base_node*(num_group-1), activation=activation, kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate))(bottle_neck)
        decoded = Dense(1, activation='linear', kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate))(decoded)
        exploration = Model(input=input_path, output=decoded)

        return exploration
    '''
    def exploration_model_stage2(self, bottle_neck_dim = 1,  regularization_rate = 0.,\
                          activation = 'sigmoid', trainable = True):
        # exploration model with second stage visualization architecture.
        def slice(x, st, ed):
            return x[:,st:ed]
        #     Construct auto-encoder with multiple layers
        n_input = self.d
        input_path = Input(shape=(n_input,))
        input0 = Lambda(slice,output_shape=(1,),arguments={'st':0, 'ed':1}  )(input_path)
        input1 = Lambda(slice,output_shape=(1,),arguments={'st':1, 'ed':2}  )(input_path)
        input_rs = Lambda(slice,output_shape=(n_input-2,),arguments={'st':2, 'ed':n_input})(input_path)
        input0 = Reshape((1,))(input0)
        input1 = Reshape((1,))(input1)
        input_rs = Reshape((n_input-2,))(input_rs)
        
        ### Compute h2
        encoded1 = Dense(64, activation=activation, kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate), trainable = trainable)(input_rs)
        encoded1 = Dense(32, activation=activation, kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate), trainable = trainable)(encoded1)
        encoded1 = Dense(bottle_neck_dim, activation='linear', kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate), trainable = trainable)(encoded1)
        bottle_neck1 = keras.layers.concatenate([input1, encoded1], axis = 1)
        ### Compute h1
        encoded2 = Dense(64, activation=activation, kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate), trainable = trainable)(bottle_neck1)
        encoded2= Dense(32, activation=activation, kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate), trainable = trainable)(encoded2)
        encoded2= Dense(1, activation='linear', kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate), trainable = trainable)(encoded2)
        bottle_neck = keras.layers.concatenate([input0, encoded2], axis = 1)
        ### Compute f
        decoded = Dense(256, activation=activation, kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate))(bottle_neck)
        decoded = Dense(32, activation=activation, kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate))(decoded)
        decoded = Dense(1, activation='linear', kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate))(decoded)
        exploration = Model(input=input_path, output=decoded)

        return exploration
        '''
    def exploration_model_stage2(self, bottle_neck_dim = 1,  regularization_rate = 0.,\
                          activation = 'sigmoid', trainable = True):
        # exploration model with second stage visualization architecture.
        def slice(x, st, ed):
            return x[:,st:ed]
        #     Construct auto-encoder with multiple layers
        n_input = self.d
        input_path = Input(shape=(n_input,))
        input0 = Lambda(slice,output_shape=(1,),arguments={'st':0, 'ed':1}  )(input_path)
        input1 = Lambda(slice,output_shape=(1,),arguments={'st':1, 'ed':2}  )(input_path)
        input_rs = Lambda(slice,output_shape=(n_input-2,),arguments={'st':2, 'ed':n_input})(input_path)
        input0 = Reshape((1,))(input0)
        input1 = Reshape((1,))(input1)
        input_rs = Reshape((n_input-2,))(input_rs)
        
        ### Compute h2
        encoded1 = Dense(16, activation=activation, kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate), trainable = trainable)(input_rs)
        encoded1 = Dense(bottle_neck_dim, activation='linear', kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate), trainable = trainable)(encoded1)
        bottle_neck1 = keras.layers.concatenate([input1, encoded1], axis = 1)
        ### Compute h1
        encoded2 = Dense(16, activation=activation, kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate), trainable = trainable)(bottle_neck1)
        encoded2= Dense(1, activation='linear', kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate), trainable = trainable)(encoded2)
        bottle_neck = keras.layers.concatenate([input0, encoded2], axis = 1)
        ### Compute f
        decoded = Dense(32, activation=activation, kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate))(bottle_neck)
        decoded = Dense(1, activation='linear', kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate))(decoded)
        exploration = Model(input=input_path, output=decoded)

        return exploration
        
    def exploration_train(self, exploration, x_train, y_train, lr = 0.01, BATCH_SIZE = 5, EPOCHS = 50):
        adam = keras.optimizers.Adagrad(lr=lr) #lr := learning rate
        exploration.compile(optimizer=adam, loss='mean_squared_error')
        """
        tbCallBack = TensorBoard(log_dir='./logs',  # log 目录
                         histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
        #                  batch_size=32,     # 用多大量的数据计算直方图
                         write_graph=True,  # 是否存储网络结构图
                         write_grads=True, # 是否可视化梯度直方图
                         write_images=True,# 是否可视化参数
                         embeddings_freq=0, 
                         embeddings_layer_names=None, 
                         embeddings_metadata=None)
        """
        history = exploration.fit(x_train,y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        verbose = 2,)
                        #callbacks=[tbCallBack]) 
        if 0:
            plt.plot(history.history['loss'][5:])
            plt.plot(history.history['val_loss'][5:])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.show()
        return exploration
    
    def exploration_grid(self, x, path):
        n, d = np.shape(x)
        m = path.shape[-1]
        grid = np.zeros((n*m, d + 1))
        for i in range(n):
            grid[i*m:(i+1)*m, :] = np.hstack((np.repeat(np.matrix(x[i, :]), m, axis = 0), np.transpose(path)))
        return grid

    def exploration_get_layer(self, exploration):
        # generate index for dense and concatenate layers
        dense = [] # input layer
        cat = []
        cnt = 0
        for layer in exploration.layers:
            if layer.name.split('_')[0] == 'dense':
                dense += [cnt]
            elif layer.name.split('_')[0] == 'concatenate':
                cat += [cnt]
            cnt += 1
        return dense, cat    
        
    def backward_get_layer(self, exploration):
        # generate index for dense and concatenate layers
        dense = [] # input layer
        cat = []
        linear = []
        cnt = 0
        for layer in exploration.layers:
            if layer.name.split('_')[0] == 'dense':
                dense += [cnt]
            elif layer.name.split('_')[0] == 'con':
                cat += [cnt]
            elif layer.name.split('_')[0] == 'linear':
                linear += [cnt]
            cnt += 1
        return dense, cat, linear

    def exploration_decode_single_group(self, exploration, vp):
        '''
            evaluation: the exploration model with input v.
            Output prediction.
        '''
        
        # feed h-values into the bottle neck layer. It requires a new computation graph
        input_path = Input(shape=(1,))
        
        layer = exploration.get_layer('dense_0')
        hidden = layer(input_path)

        layer = exploration.get_layer('dense_1')
        output = layer(hidden)

        y = Model(inputs = input_path, outputs = output)    
        yp = y.predict(vp)
        return yp
    
    def exploration_encode(self, exploration, x_train, stage, dense, cat):
        def slice(x, st, ed):
            return x[:,st:ed]
        '''
            generate h_i and h_{i+1} from the exploration neural network 
        '''
        if stage > self.d:
            raise IndexError('stage out of range')
        
        n_input = self.d
        input_path = Input(shape=(n_input,))
        input_node = {}
        for i in range(n_input):
            input0 = Lambda(slice,output_shape=(1,),arguments={'st':n_input-i-1, 'ed':n_input-i})(input_path)
            input_rs = Lambda(slice,output_shape=(n_input-i-1,),arguments={'st':0, 'ed':n_input-i-1})(input_path)
            input0 = Reshape((1,))(input0)
            input_rs = Reshape((n_input-i-1,))(input_rs)
            input_node['xj'+str(n_input-i-1)] = input0
        cat_stage = [l for l in cat if l <= cat[n_input - stage - 1]] # the layer before h_mid

        h2_encode = input_node['xj'+str(n_input-1)]
        cnt = 1
        dense_id = 0
        for j in cat_stage:
            while dense[dense_id] < j:
                i = dense[dense_id]
                layer = exploration.layers[i]
                h2_encode = layer(h2_encode)
                dense_id += 1
            if cnt < n_input - stage:
                h2_encode = keras.layers.concatenate([input_node['xj'+str(n_input-cnt-1)], h2_encode], axis = 1)
            cnt += 1

        h1_encode =  keras.layers.concatenate([input_node['xj'+str(n_input-cnt)], h2_encode], axis = 1)
        for i in dense[dense_id:dense_id+2]:
            layer = exploration.layers[i]
            h1_encode = layer(h1_encode)

        # Create the h-value model
        h2_val = Model(inputs = input_path, outputs = h2_encode)                
        h_mid = h2_val.predict(x_train)
        if stage == n_input - 1:
            h_mid = x_train[:, n_input-1]
   
        h1_val = Model(inputs = input_path, outputs = h1_encode)   
        h_out = h1_val.predict(x_train)
        
        return h_mid, h_out     

    def backward_encode(self, exploration, x_train, stage, disjoint_group):
        def slice(x, st, ed):
            return x[:,st:ed]
        '''
            generate h_i and h_{i+1} from the exploration neural network 
        '''
        dense, cat, linear = self.backward_get_layer(exploration)
        num_group = len(disjoint_group)
        if stage > num_group:
            raise IndexError('stage out of range')
        
        n_input = self.d
        input_path = Input(shape=(n_input,))
        input_node = {}
        tot_len = 0
        for i in range(num_group):
            length = len(disjoint_group[i])
            input0 = Lambda(slice,output_shape=(length,),arguments={'st':tot_len, 'ed':tot_len+length})(input_path)
            #input_rs = Lambda(slice,output_shape=(n_input-i-1,),arguments={'st':0, 'ed':n_input-i-1})(input_path)
            input0 = Reshape((length,))(input0)
            #input_rs = Reshape((n_input-i-1,))(input_rs)
            if length == 1:
                encoded = input0
            else:
                layer = exploration.get_layer('linear_' + str(i))
                encoded = layer(input0)
            input_node['group_'+str(i)] = encoded
            tot_len += length
        bottle_neck = keras.layers.concatenate([input_node['group_'+str(num_group-2)],\
             input_node['group_'+str(num_group-1)]], axis = 1, name = 'con_0')

        cat_stage = [l for l in cat if l <= cat[num_group - stage - 1]] # the layer before h_mid
        h2_encode = input_node['group_'+str(num_group-1)]
        cnt = 1
        dense_id = 0
        for j in cat_stage:
            while dense[dense_id] < j:
                i = dense[dense_id]
                layer = exploration.layers[i]
                h2_encode = layer(h2_encode)
                dense_id += 1
            if cnt < num_group - stage:
                h2_encode = keras.layers.concatenate([input_node['group_'+str(num_group-cnt-1)], h2_encode], axis = 1)
            cnt += 1

        v_encode = input_node['group_'+str(num_group-cnt)] # linear layer
        h1_encode =  keras.layers.concatenate([v_encode, h2_encode], axis = 1)
        for i in dense[dense_id:dense_id+2]:
            layer = exploration.layers[i]
            h1_encode = layer(h1_encode)

        # Create the h-value model
        v_val = Model(inputs = input_path, outputs = v_encode)
        v = v_val.predict(x_train)

        h2_val = Model(inputs = input_path, outputs = h2_encode)                
        h_mid = h2_val.predict(x_train)
        if stage == n_input - 1:
            h_mid = x_train[:, n_input-1]
   
        h1_val = Model(inputs = input_path, outputs = h1_encode)   
        h_out = h1_val.predict(x_train)
        
        return v, h_mid, h_out, h1_val    
    
    def comparison_exp(self, x, h, exp_grid, mds_index, exploration, bottle_neck_dim = 1, name = 'exp_sin4_3d.jpg'):
        x_discrete = self.dx1 * (self.ub[mds_index] - self.lb[mds_index]) + self.lb[mds_index]
        x_grid_2d, h_grid_2d = np.meshgrid(x_discrete, h)
        #plt.axis([0,1,-1,1])
        fig = plt.figure()
            
        ######################################################
        #   plot MDS
        ######################################################
        
        ax1 = fig.add_subplot(111, projection = '3d')
        ax1.set_xlabel('$x_1$')
        ax1.set_ylabel('h')
        ax1.set_zlabel('f')
        ax1.set_title('MDS')
        for i in range(len(h)): 
            path = np.reshape(np.array(self.obj(np.insert(np.repeat(np.matrix(x[i, :]),\
                                                        self.N1, axis = 0), mds_index, values = self.dx1, axis = 1))), [1, -1])
            if i == 0:
                z_mds = path
            else:
                z_mds = np.concatenate((z_mds, path), axis = 0)
        
        ######################################################
        #   plot auto-encoder
        ###################################################### 
        
        x_exp = exp_grid[:,mds_index]
        h_exp = self.exploration_encode(exploration, exp_grid,)
        x_exp = np.reshape(x_exp, x_grid_2d.shape)
        h_exp = np.reshape(h_exp, h_grid_2d.shape)
        
        z_exp = self.exploration_decode(exploration, x_grid_2d.reshape((-1, 1)), h_grid_2d.reshape((-1, bottle_neck_dim)))

        z_exp = np.reshape(z_exp, x_grid_2d.shape)
        #ax2 = fig.add_subplot(122, projection = '3d')
        fig = plt.figure()
        ax2 = fig.add_subplot(111, projection = '3d')
        ax2.set_xlabel('$x_1$')
        ax2.set_ylabel('h')
        ax2.set_zlabel('f')
        ax2.set_title('exploration')
                
        
        #autoencoder_path = autoencoder.predict(x_test)
        #z_auto = np.multiply(np.transpose(x_test), norms).transpose() + mean
        #z_exp = exploration.predict(exp_grid)


        ax1.plot_surface(x_grid_2d, h_grid_2d, z_mds, cmap = cm.coolwarm,
                 rstride = 1, cstride = 1, antialiased = True)
        
        ax2.plot_surface(x_grid_2d, h_grid_2d, z_mds, cmap = cm.coolwarm,
                 rstride = 1, cstride = 1, antialiased = True)

        # Calculate the function exploration model on the grids(x_grid_2d, h_grid_2d)

        
        save_name = os.getcwd() + '/compare_image/' + name
        #plt.legend(loc=0, ncol=2)
        #plt.savefig(save_name)
        
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('h_mds')
        ax.set_ylabel('h_auto')
        ax.plot(np.sort(h), np.sort(h_auto))
        '''
        
        plt.show()
        h_show = h_exp
        return h_show
    
    def exp_stage2(self, x, h, exp_grid, exp_grid_stage2, mds_index, stage2_index, exploration, bottle_neck_dim = 1, name = 'exp_sin4_3d.jpg'):
        x_discrete = self.dx1 * (self.ub[mds_index] - self.lb[mds_index]) + self.lb[mds_index]
        x_discrete_stage2 = self.dx1 * (self.ub[stage2_index] - self.lb[stage2_index]) + self.lb[stage2_index]
        x_grid_2d, h_grid_2d = np.meshgrid(x_discrete, h)
        x_grid_2d_stage2, h_grid_2d_stage2 = np.meshgrid(x_discrete_stage2, h)
        #plt.axis([0,1,-1,1])
        fig = plt.figure()
            
        ######################################################
        #   plot MDS
        ######################################################
        
        ax1 = fig.add_subplot(111, projection = '3d')
        ax1.set_xlabel('$x_1$')
        ax1.set_ylabel('h')
        ax1.set_zlabel('f')
        ax1.set_title('MDS')
        for i in range(len(h)): 
            path = np.reshape(np.array(self.obj(np.insert(np.repeat(np.matrix(x[i, :]),\
                                                        self.N1, axis = 0), mds_index, values = self.dx1, axis = 1))), [1, -1])
            if i == 0:
                z_mds = path
            else:
                z_mds = np.concatenate((z_mds, path), axis = 0)
        
        ######################################################
        #   plot auto-encoder
        ###################################################### 
        
        x_exp = exp_grid[:,mds_index]
        h1_exp, h2_exp = self.exploration_encode_stage2(exploration, exp_grid_stage2)
        x_exp = np.reshape(x_exp, x_grid_2d.shape)
        h1_exp = np.reshape(h1_exp, h_grid_2d.shape)
        h2_exp = np.reshape(h2_exp, h_grid_2d.shape)
        
        h_exp,h_exp_tmp = self.exploration_encode_stage2(exploration, exp_grid,)
        h_exp = np.reshape(h_exp, h_grid_2d.shape)
        
        #z_exp = self.exploration_decode(exploration, x_grid_2d.reshape((-1, 1)), h_grid_2d.reshape((-1, bottle_neck_dim)))

        #z_exp = np.reshape(z_exp, x_grid_2d.shape)
        #ax2 = fig.add_subplot(122, projection = '3d')
        fig = plt.figure()
        ax2 = fig.add_subplot(111, projection = '3d')
        ax2.set_xlabel('$x_{j2}$')
        ax2.set_ylabel('$h_2$')
        ax2.set_zlabel('$h_1$', rotation=90)
        ax2.set_title('exploration stage 2')
        
        fig = plt.figure()
        ax3 = fig.add_subplot(111, projection = '3d')
        ax3.set_xlabel('$x_{j1}$')
        ax3.set_ylabel('$h_1$')
        ax3.set_zlabel('$f$')
        ax3.set_title('exploration stage 1')
                
        
        #autoencoder_path = autoencoder.predict(x_test)
        #z_auto = np.multiply(np.transpose(x_test), norms).transpose() + mean
        #z_exp = exploration.predict(exp_grid)
        '''
        ax1.scatter3D(x_grid_2d, h_grid_2d, z_mds, color = 'green')
        ax2.scatter3D(x_grid_2d_stage2, h1_exp, color = 'green')
        ax3.scatter3D(x_grid_2d, h_exp, z_mds, color = 'green')
        '''
        ax1.plot_surface(x_grid_2d, h_grid_2d, z_mds, cmap = cm.coolwarm,
                 rstride = 1, cstride = 1, antialiased = True)
        
        ax2.plot_surface(x_grid_2d_stage2, h2_exp, h1_exp, cmap = cm.coolwarm,
                 rstride = 1, cstride = 1, antialiased = True)  
        
        ax3.plot_surface(x_grid_2d, h_exp, z_mds, cmap = cm.coolwarm,
                 rstride = 1, cstride = 1, antialiased = True)
        
        # Calculate the function exploration model on the grids(x_grid_2d, h_grid_2d)

        
        save_name = os.getcwd() + '/compare_image/' + name
        #plt.legend(loc=0, ncol=2)
        #plt.savefig(save_name)
        
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('h_mds')
        ax.set_ylabel('h_auto')
        ax.plot(np.sort(h), np.sort(h_auto))
        '''
        
        plt.show()
        h_show = h_exp
        return h_show

    def sparse_plot(self, x, h, exp_grid, exp_grid_stage2, x_plot, x_plot_s2, \
                    mds_index, stage2_index, exploration):
        x_discrete = self.dx1 * (self.ub[mds_index] - self.lb[mds_index]) + self.lb[mds_index]
        x_discrete_stage2 = self.dx1 * (self.ub[stage2_index] - self.lb[stage2_index]) + self.lb[stage2_index]
        x_grid_2d, h_grid_2d = np.meshgrid(x_discrete, h)
        x_grid_2d_stage2, h_grid_2d_stage2 = np.meshgrid(x_discrete_stage2, h)
        z_exp = self.obj(exp_grid)
        
        ######################################################
        #   plot MDS
        ######################################################
        
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection = '3d')
        ax1.set_xlabel('$x_1$')
        ax1.set_ylabel('h')
        ax1.set_zlabel('f')
        ax1.set_title('MDS')
        for i in range(len(h)): 
            path = np.reshape(np.array(self.obj(np.insert(np.repeat(np.matrix(x[i, :]),\
                                                        self.N1, axis = 0), mds_index, values = self.dx1, axis = 1))), [1, -1])
            if i == 0:
                z_mds = path
            else:
                z_mds = np.concatenate((z_mds, path), axis = 0)
        ######################################################
        #   plot sparse grid exploration method.
        ###################################################### 
        
        x_exp = exp_grid[:,mds_index]
        h1_exp, h2_exp = self.exploration_encode_stage2(exploration, exp_grid_stage2)
        h_exp, h_exp_tmp = self.exploration_encode_stage2(exploration, exp_grid,)

        ### generate grid points for smooth surface.
        h1_exp_grid, h2_exp_grid = self.exploration_encode_stage2(exploration, x_plot_s2)
        h1_exp_grid = np.reshape(h1_exp_grid, h_grid_2d_stage2.shape)
        h2_exp_grid = np.reshape(h2_exp_grid, h_grid_2d_stage2.shape)
        
        h_exp_grid, h_exp2_grid = self.exploration_encode_stage2(exploration, x_plot,)
        h_exp_grid = np.reshape(h_exp_grid, h_grid_2d.shape)
        
        fig = plt.figure()
        ax2 = fig.add_subplot(111, projection = '3d')
        ax2.set_xlabel('$x_{j2}$')
        ax2.set_ylabel('$h_2$')
        ax2.set_zlabel('$h_1$', rotation=90)
        ax2.set_title('exploration stage 2')
        
        fig = plt.figure()
        ax3 = fig.add_subplot(111, projection = '3d')
        ax3.set_xlabel('$x_{j1}$')
        ax3.set_ylabel('$h_1$')
        ax3.set_zlabel('$f$')
        ax3.set_title('exploration stage 1')
        if 0:
            #################################################################
            ### Gaussian Process Regression (GPR)
            #################################################################
            '''
            gp_s2 = sklearn.gaussian_process.GaussianProcessRegressor(kernel=RBF(10, (1e-2, 1e2)))
            Data_s2 = np.concatenate((exp_grid_stage2[:,1], h2_exp, h1_exp), axis = 1)
            gp_s2.fit(Data_s2[:,0:2], Data_s2[:,2])  
            mesh_grid_s2 = np.concatenate((np.reshape(x_grid_2d_stage2, (-1, 1)), np.reshape(h2_exp_grid, (-1,1))), axis = 1)
            z_pred_s2 = gp_s2.predict(mesh_grid_s2)
            '''
            Data_s2 = np.concatenate((exp_grid_stage2[:,1], h2_exp, h1_exp), axis = 1)
            z_pred_s2 = griddata(Data_s2[:,0:2], Data_s2[:,2], (x_grid_2d_stage2, h2_exp_grid), method = 'linear')
            
            z_pred_s2 = np.reshape(z_pred_s2, h_grid_2d_stage2.shape)
            ax2.plot_surface(x_grid_2d_stage2, h2_exp_grid, z_pred_s2, cmap = cm.coolwarm,
                    rstride = 1, cstride = 1, antialiased = True)  
            '''
            gp = sklearn.gaussian_process.GaussianProcessRegressor(kernel=RBF(10, (1e-2, 1e2)))
            Data = np.concatenate((exp_grid[:,0], h_exp, z_exp), axis = 1)
            gp.fit(Data[:,0:2], Data[:,2])  
            mesh_grid = np.concatenate((np.reshape(x_grid_2d, (-1, 1)), np.reshape(h_exp_grid, (-1,1))), axis = 1)
            z_pred = gp_s2.predict(mesh_grid)
            '''
            Data = np.concatenate((exp_grid[:,0], h_exp, z_exp), axis = 1)
            z_pred = griddata(Data[:,0:2], Data[:,2], (x_grid_2d, h_exp_grid), method = 'linear')
            z_pred = np.reshape(z_pred, h_exp_grid.shape)
            ax3.plot_surface(x_grid_2d, h_exp_grid, z_pred, cmap = cm.coolwarm,
                    rstride = 1, cstride = 1, antialiased = True)


            #ax2.scatter3D(exp_grid_stage2[:,1], h2_exp, h1_exp, color = 'green')
            #ax3.scatter3D(exp_grid[:,0], h_exp, z_exp, color = 'green')
        else:
            
            ax1.plot_surface(x_grid_2d, h_grid_2d, z_mds, cmap = cm.coolwarm,
                    rstride = 1, cstride = 1, antialiased = True)
            
            ax2.plot_surface(x_grid_2d_stage2, h2_exp_grid, h1_exp_grid, cmap = cm.coolwarm,
                    rstride = 1, cstride = 1, antialiased = True)  
            #plt.savefig(os.getcwd() + '/history_image/sin_uncommon_2.png')
            
            ax3.plot_surface(x_grid_2d, h_exp_grid, z_mds, cmap = cm.coolwarm,
                    rstride = 1, cstride = 1, antialiased = True)
            #plt.savefig(os.getcwd() + '/history_image/sin_uncommon_1.png')
            

            #ax2.scatter3D(exp_grid_stage2[:,1], h2_exp, h1_exp, color = 'green')
            #ax3.scatter3D(exp_grid[:,0], h_exp, z_exp, color = 'green')
        # Calculate the function exploration model on the grids(x_grid_2d, h_grid_2d)
    
    def stage_plot(self, stage, x_plot, y_plot, index, var_index, name, exploration, nonadd_id):
        '''
            Make the 3D visualization plot for specific stage using the function:
            exploration_encode functionhzn.
        '''
        x_discrete = self.dx1 * (self.ub[index] - self.lb[index]) + self.lb[index]
        
        dense, cat = self.exploration_get_layer(exploration)
        h_mid, h_out = self.exploration_encode(exploration, x_plot, stage, dense, cat)

        h_mid = np.reshape(h_mid, (-1, self.N1))
        h_out = np.reshape(h_out, (-1, self.N1))

        h = h_mid[:,0]
        if stage==self.d-1:
            h_mid =  h_mid * (self.ub[nonadd_id[var_index[-1]]] - self.lb[nonadd_id[var_index[-1]]]) \
                + self.lb[nonadd_id[var_index[-1]]]

        x_grid_2d, h_grid_2d = np.meshgrid(x_discrete, h)
        ######################################################
        #   plot Neural Network at given stage
        ######################################################
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection = '3d')
        ax1.set_xlabel('$x$'+str(nonadd_id[var_index[stage-1]]+1))
        ax1.set_ylabel('h'+str(stage))
        if stage == 1:
            ax1.set_zlabel('f')
        else:
            ax1.set_zlabel('h'+str(stage-1))
        if stage == self.d - 1:
            ax1.set_ylabel('$x$'+str(nonadd_id[var_index[stage]]+1))

        ax1.set_title('Visualization for level %d' % stage)

        azimuth = 135+180
        altitude = 45
        ax1.view_init(altitude, azimuth)
        cmap = matplotlib.colors.ListedColormap("red") 

        if stage == 1:
            z_mds = np.reshape(y_plot, h_mid.shape)
            #ax1.plot_surface(x_grid_2d, h_mid, z_mds, cmap = cm.coolwarm,\
            #        rstride = 1, cstride = 1, antialiased = True)
            ax1.plot_trisurf(x_grid_2d.flatten(), h_mid.flatten(), z_mds.flatten(), cmap = cm.coolwarm)
            #ax1.scatter3D(x_grid_2d, h_mid, z_mds)
            #ax1.scatter3D(x_grid_2d, h_mid, h_out)
        else:
            h_mid = np.array(h_mid)
            ax1.plot_trisurf(x_grid_2d.flatten(), h_mid.flatten(), h_out.flatten(), cmap = cm.coolwarm, antialiased = True)
            #ax1.scatter3D(x_grid_2d, h_mid, h_out)
        # Store the image
        if 1:
            file_path = os.path.join(os.getcwd(), 'image')
            file_path = os.path.join(file_path, 'tmp')
            save_name = os.path.join(file_path, name + str(stage)+'.png')
            print(save_name)
            plt.savefig(save_name, dpi=100)
            plt.show()
            plt.close()

    def backward_additive_test(self, stage, x_plot, y_plot, disjoint_group, name, exploration,\
         num_x, num_y, nonadd_id, add_eta = 1e-6):
        '''
            test the additivity of neural network at given stage.
            x_plot: indexed by its original order
        '''
        if stage <= 1:
            raise ValueError('Additive Test Stage should be greater than 1')
        var_index = sum(disjoint_group, [])
        rm_index = [i for i in range(self.d) if i not in sum(disjoint_group[:stage-1], [])]

        _, _, _, func_h = self.backward_encode(exploration, np.zeros((1, self.d)), stage, disjoint_group)
        def backward_obj(x):
            #   Get the objective function from fitted neural network at specific stage.
            l = np.shape(x)[0]
            x_ = np.zeros((l, self.d)) # fill to the original dimension
            x_[:,rm_index] = x
            x_reorder = x_[:, var_index]
            h_out = func_h.predict(x_reorder)
            return h_out.reshape((-1,1))

        grid_add_stage = lhs(len(rm_index), samples=200, criterion='m')
        add_id = additive(backward_obj, grid_add_stage, N_j = 20, grid_length = 0.01, eta=add_eta)
        print('add_id is ')
        print(add_id)
        if len(add_id) > 0 and add_id[0] == 0:
            return True
        else:
            self.backward_stage_plot(stage, x_plot_reorder, y_plot, disjoint_group,\
                 name, exploration, num_x, num_y, nonadd_id)
            return False


    def backward_stage_plot(self, stage, x_plot, y_plot, disjoint_group, name, exploration,\
         num_x, num_y, nonadd_id):
        '''
            Make the 3D visualization plot for specific stage using the function:
            exploration_encode functionhzn.
        '''
        var_index = sum(disjoint_group, [])
        num_group = len(disjoint_group)

        v, h_mid, h_out, _ = self.backward_encode(exploration, x_plot, stage, disjoint_group)
        v = np.reshape(v, (-1, num_x))
        h_mid = np.reshape(h_mid, (-1, num_x))
        h_out = np.reshape(h_out, (-1, num_x)) 
        if stage == 1:
            l = len(h_mid)
            h1_median = np.sort(h_mid[:,0])[int(l/2)]
            ind = np.where(h_mid[:,0] == h1_median)
            h1_path = h_mid[ind].flatten()
            v_path = v[ind].flatten()
            f_path = h_out[ind].flatten()

        
        if len(disjoint_group[stage-1]) == 1:
            index = disjoint_group[stage-1][0]
            v = v * (self.ub[index] - self.lb[index]) + self.lb[index]
        
        if stage == num_group-1:
            if len(disjoint_group[stage]) == 1:
                index = disjoint_group[stage][0]
                h_mid = h_mid * (self.ub[index] - self.lb[index]) + self.lb[index]

        ######################################################
        #   plot Neural Network at given stage
        ######################################################
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection = '3d')
        if len(disjoint_group[stage-1]) == 1:
            index = disjoint_group[stage-1][0]
            index = nonadd_id[index]
            ax1.set_xlabel('$x$'+str(index+1))
        else:
            ax1.set_xlabel('$v$'+str(stage))
        ax1.set_ylabel('h'+str(stage))
        if stage == 1:
            ax1.set_zlabel('f')
        else:
            ax1.set_zlabel('h'+str(stage-1))
        if stage == num_group - 1:
            if len(disjoint_group[stage]) == 1:
                index = disjoint_group[stage][0]
                index = nonadd_id[index]
                ax1.set_ylabel('$x$'+str(index+1))
            else:
                ax1.set_ylabel('$v$'+str(stage+1))

        ax1.set_title('Visualization for level %d' % stage)

        rgb = np.ones((h_out.shape[0], h_out.shape[1], 3))
        light = LightSource(90, 0)
        illuminated_surface = light.shade_rgb(rgb, h_out)
        green = np.array([0,1.0,0])
        green_surface = light.shade_rgb(rgb * green, h_out)

        # Set view parameters for all subplots.
        azimuth = 135+180
        altitude = 60
        ax1.view_init(altitude, azimuth)
        cmap = matplotlib.colors.ListedColormap("red")     
        if stage == 1:
            z_mds = np.reshape(y_plot, h_mid.shape)
            ax1.plot_trisurf(v.flatten(), h_mid.flatten(), z_mds.flatten(), \
                cmap = cm.coolwarm, linewidth=0, antialiased=True)
            # Add a line on the surface
            if 0:
                ind = np.argsort(v_path)
                v_path = v_path[ind]
                f_path = f_path[ind]+0.05
                line = ax1.plot3D(v_path, h1_path, f_path, color = 'k', zorder = 400)
            #ax1.scatter3D(v, h_mid, h_out)
        else:
            h_mid = np.array(h_mid)
            ax1.plot_trisurf(v.flatten(), h_mid.flatten(), h_out.flatten(), \
                cmap = cm.coolwarm, linewidth=0, antialiased=True)
            #ax1.scatter3D(v, h_mid, h_out)
        # Store the image
        #plt.show()
        
        if 1:
            file_path = os.getcwd() + '\\image\\'
            save_name = file_path + name + str(stage)+'.png'
            print(save_name)
            plt.savefig(save_name, dpi=100)
            plt.show()
            plt.close()


    def exp_plot_in_one(self, name, row = 2, column = 3):
        file_path = os.path.join(os.getcwd(), 'image')
        pic_path = os.path.join(file_path, 'tmp')
        count = 1
        pic = Image.open(os.path.join(pic_path,name+str(count)+'.png'))
        width, height = pic.size
        toImage = Image.new('RGBA', (column*width, row*height))
        for i in range(row):
            for j in range(column):
                pic = Image.open(os.path.join(pic_path,name+str(count)+'.png'))
                #print(file_path+name+'_auto_'+str(count)+'.png')
                toImage.paste(pic, (j*width, i*height))
                count += 1
                if count >= self.d:
                    break
            if count >= self.d:
                break
        toImage.save(os.path.join(file_path, name+'.png'))

    def backward_plot_in_one(self, name, disjoint_group, row = 2, column = 3):
        num_group = len(disjoint_group)
        file_path = os.getcwd() + '\\image\\'
        count = 1
        pic = Image.open(file_path+name+str(count)+'.png')
        width, height = pic.size
        toImage = Image.new('RGBA', (column*width, row*height))
        for i in range(row):
            pic = Image.open(file_path+name+str(count)+'.png')
            toImage.paste(pic, (0*width, i*height))
            if column > 1:
                pic_linear = Image.open(file_path+name+str(count)+'_linear.png')
                toImage.paste(pic_linear, (1*width, i*height))
            count += 1
            if count >= num_group:
                break
        toImage.save(file_path+name+'.png')
        
