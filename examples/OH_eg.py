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
from function_class import function 
import sklearn.preprocessing as prep
from var_selection import variable_selection, active_subspace
from numpy import linalg as LA
from test_function import test

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


if __name__ == '__main__':   
    test = test()
    dim = 5

    lb, ub = bound(dim)
    func_dict = {'func_obj': func_obj, 'lb': lb, 'ub': ub}
    func_dict['dim'] = dim
    func_dict['original dimension'] = dim
    func_dict['additive_index'] = []
    func= function(func_dict)

    name = 'OH_eg_'

    #func.func_name = 'borehole'
    func.N1 = 50
    func.n = 200

    isomap_flag = 1

    bottle_neck_dim = 1

    #Variable Selection
    grid_var = lhs(dim, samples=100, criterion='m')
    var_index = variable_selection(func.obj, grid_var, N_j = 50, grid_length = 0.01)
    print(var_index)
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
    grid = lhs(dim, samples=np.int(1000), criterion = 'm')
    #grid = np.random.uniform(0, 1, size = [int(1e5), dim])

    #grid_NN = permute_index(grid, var_index)

    #y_LHD = func.obj(grid)                  




    # LHD for test set

    #N_test = 10
    # x_test = np.array([(x1, x2, x3, x4, x5) for x1 in range(N_test) for x2 in range(N_test) \
    #            for x3 in range(N_test) for x4 in range(N_test) for x5 in range(N_test) ])

    x_val = np.random.uniform(0, 1, size = [int(1e3), dim])
    y_val = func.obj(x_val)

    x_test = np.random.uniform(0, 1, size = [int(1e6), dim])
    y_test = func.obj(x_test)

    # LHD total for training set and set boundary to the extreme points.
    train_x = func.sparse_grid(grid)
    train_y = func.obj(train_x)

    '''
    train_x = np.multiply(train_x, np.array(func.ub)-np.array(func.lb)) + np.array(func.lb)
    x_val = np.multiply(x_val, np.array(func.ub)-np.array(func.lb)) + np.array(func.lb)
    x_test = np.multiply(x_test, np.array(func.ub)-np.array(func.lb)) + np.array(func.lb)
    '''

    train_x = train_x[:, var_index]
    x_val = x_val[:,var_index]
    x_test = x_test[:,var_index]


    #test_grid = np.repeat(.5, func.d-1).reshape((1,-1))
    #test_x = func.input_data(test_grid, 0, 1, func.id)

    # Construct the exploration neural network model 
    def copyModel2Model(model_source,model_target,certain_layer=""):        
        for l_tg,l_sr in zip(model_target.layers,model_source.layers):
            wk0=l_sr.get_weights()
            l_tg.set_weights(wk0)
            if l_tg.name==certain_layer:
                break
        print("model source was copied into model target") 

    """
    exploration_stage2_encode = func.exploration_model_stage2( bottle_neck_dim = bottle_neck_dim,\
                                        regularization_rate = regularization_rate,\
                                        activation = 'relu', trainable = True)

    exploration_stage2_decode = func.exploration_model_stage2( bottle_neck_dim = bottle_neck_dim,\
                                        regularization_rate = regularization_rate,\
                                        activation = 'relu', trainable = False)


    # Train the neural network for stage 2
    exploration_stage2_encode = func.exploration_train(exploration_stage2_encode, x_train_s2_NN, y_train_s2, x_test, y_test,\
                                        BATCH_SIZE = 5, lr = 1e-3,\
                                        EPOCHS = 50)
    copyModel2Model(exploration_stage2_encode,exploration_stage2_decode,)

    '''
    layer1 = exploration_stage2_encode.layers[3]
    print(layer1.get_weights())

    layer2 = exploration_stage2_decode.layers[3]
    print(layer2.get_weights())
    '''

    # Train the neural network for stage 1
    exploration_stage2 = func.exploration_train(exploration_stage2_decode, x_train_NN, y_train, x_test, y_test,\
                                        BATCH_SIZE = 5, lr = 1e-3,\
                                        EPOCHS = 50)
    """
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

    regularization=1e-8
    #xploration = func.active_subspace_model(regularization_rate = regularization,\
    #                                 activation = 'sigmoid', node = 16, skip = False)
    if 0:
        exploration = func.exploration_model_additive(regularization_rate = regularization,\
                                        activation = 'sigmoid', base_node = 16)
    else:
        exploration = func.exploration_model(regularization_rate = regularization,\
                                        activation = 'relu', base_node = 16)
    exploration = func.exploration_train(exploration, train_x, train_y, x_val, y_val,\
                                        BATCH_SIZE = 32, lr = 5e-2,\
                                        EPOCHS = 200)

    """
    layer = exploration.get_layer('dense_1')
    weight = layer.get_weights()[0]
    for i in range(np.shape(weight)[1]):
        for j in range(2):
            if np.abs(weight[j,i]) < 1e-1:
                weight[j,i] = 0
        '''
        if np.abs(weight[0, i]) > np.abs(weight[1,i]):
            weight[1, i] = 0
        else:
            weight[0,i] = 0
        '''
    """

    y_pred = exploration.predict(x_test).flatten()
    y_test = y_test.flatten()
    r2 = 1 - LA.norm(y_pred - y_test, 2)**2 / (LA.norm(y_test)**2)
    print('the test r^2 is', r2)


    #evaulation#
    num_y = func.n
    num_x = func.N1
    grid_mds = func.generate_grid(dim-1, 50)
    disjoint_group = [[var_index[i]] for i in range(dim)]
    grid_mds = func.generate_grid(dim-1, func.n)
    nonadd_id = list(range(dim))
    for stage in np.arange(1, dim):

        # 2d path grids for first stage visualization 
        x_plot = func.input_path_grid(grid_mds, 0, func.n,  var_index[stage-1])
        #x_plot = np.multiply(x_plot, np.array(func.ub)-np.array(func.lb)) + np.array(func.lb)
        y_plot = func.obj(x_plot)
        x_plot = x_plot[:, var_index]

        h_show = func.stage_plot(stage, x_plot, y_plot, var_index[stage-1], var_index, name, exploration, nonadd_id)

        # h_show = func.backward_stage_plot(stage, x_plot, y_plot, disjoint_group,\
        #          name, exploration, num_x, num_y, list(range(dim)))
    #func.compound_img(x,  h, index, autoencoder, x_test = x_train, h_num = 5, name = 'sin4.jpg', plot = 2)

    func.exp_plot_in_one(name, 2, 2)

    # r^2 0.9994931951672645

