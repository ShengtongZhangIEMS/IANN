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
from var_selection_greedy import greedy_disjoint_subspace
from var_selection import variable_selection, active_subspace, disjoint_subspace, disjoint_selection, additive
from numpy import linalg as LA
from test_function import test
import os
from pandas.plotting import table
import pandas as pd
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LinearRegression

from scipy.optimize import linprog

from pyDOE import *

def bound(dim):
    x = np.zeros((1, dim))
    if method == 0:
        _, lb, ub = test.OTL(x)
    if method == 1:
        _, lb, ub = test.piston(x)
    if method == 2:
        _, lb, ub = test.borehole(x)
    return lb, ub 

def quadratic(x):
    f1 = np.multiply(np.square(x[:,0] - 0.5), np.square(np.sum(x[:, 1:4], axis = 1)))
    #f1 =  np.square(np.sum(x[:, :3], axis = 1))
    f = np.multiply(f1, (np.sum(x[:, 4:6], axis = 1))) + np.square(x[:,6] - 0.5) + np.power(x[:,7], 3)
    return f

def func_obj(x):
    if method == 0:
        f, _, _ = test.OTL(x)
    if method == 1:
        f, _, _ = test.piston(x)
    if method == 2:
        f, _, _ = test.borehole(x)
    return f.reshape((-1, 1))

def Sensitivity(obj, dim):    
    # Include Sensitivity Analysis from SALib
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    problem = {
        'num_vars': dim,
        'names': ['x'+str(i+1) for i in range(dim)],
        'bounds': [[lb[i], ub[i]] for i in range(dim)]
    }
    param_values = saltelli.sample(problem, 10000)
    Y = np.zeros([param_values.shape[0]])

    for i, X in enumerate(param_values):
        Y[i] = obj(X.reshape((1,-1)))
    Si = sobol.analyze(problem, Y)
    Si_diff = Si['ST'] - Si['S1']
    # First-order indices: measures the contribution to the output variance by a single model input alone.
    print(Si['S1']) 
    print(Si['ST']) 
    return Si
    
if __name__ == '__main__':   
    test = test()
    method = 2
    if method == 0:
        dim = 6
        lb, ub = bound(dim)
        fig_name = 'OTL_'
    if method == 1:
        dim = 7
        lb, ub = bound(dim)
        fig_name = 'piston_'
    if method == 2:
        dim = 8
        lb, ub = bound(dim)
        fig_name = 'borehole_'

    func_dict = {'func_obj': func_obj, 'lb': lb, 'ub': ub}
    
    
    # Sensitivity Analysis
    Si = Sensitivity(func_obj, dim)
    Si_diff = abs(Si['ST'] - Si['S1'])
    print("The total order indexes from GSA:", np.around(Si['ST'], decimals = 3))

    # # pick out the variable with additive effect
    # add_eta = 1e-2
    # add_id = [i for i in range(dim) if Si_diff[i] < add_eta]
    
    # Seperate Variables with Additive Effect
    #grid_add = lhs(dim, samples=500, criterion='m')
    #add_eta = 1e-4
    #add_id = additive(func_obj, grid_add, grid_length = 0.01, eta=add_eta)
    add_id = []

    func_d = dim - len(add_id)
    func_dict['dim'] = func_d
    func_dict['original dimension'] = dim
    func_dict['additive_index'] = add_id

    func= function(func_dict)

    #func.func_name = 'borehole'
    func.N1 = 50
    func.n = 200

    test_r2 = []

    bottle_neck_dim = 1

    # plot the additive effect predictors
    if 1:
        h = 1e-2
        x_mid = np.array([(func.lb[i] + func.ub[i])/2 for i in range(dim)])
        for i in add_id:
            x_discrete = np.arange(func.lb[i], func.ub[i], h)
            l = len(x_discrete)
            x_plot = np.repeat([x_mid], l, 0)
            x_plot[:,i] = x_discrete
            y_plot = func.func_obj(x_plot)
            ax = plt.subplot(111)
            ax.plot(x_discrete, y_plot)
            ax.set_xlabel('x'+str(i+1))
            ax.set_ylabel('f')
            ax.set_title('visualization for additive variable')
            plt.show()


    #Variable Selection
    eta = 1e-4
    #disjoint_group = greedy_disjoint_subspace(func.obj, dim, 100, init_index=[5], grid_length = 0.01, eta=1e-4)
    #print([[i+1 for i in l] for l in disjoint_group])
    grid_var = lhs(func_d, samples=500, criterion = 'm')

    dg = disjoint_subspace(func.obj, func_d, grid_var, grid_length = 0.01, eta=eta)
    num_group = len(dg)

    # base of samples
    grid = lhs(func_d, samples=np.int(1000), criterion = 'm')

    for ind in range(num_group):
        disjoint_group = disjoint_selection(func.obj, dg, grid_var, init_index=[dg[ind][0]], grid_length = 0.01)
        print('The group index is %s' % disjoint_group)
        print([[i+1 for i in l] for l in disjoint_group])

        x_val = np.random.uniform(0, 1, size = [int(1e4), func_d])
        #v_val = backward_trans(func.obj, x_val, disjoint_group, eig_vec)
        y_val = func.obj(x_val)

        x_test = np.random.uniform(0, 1, size = [int(1e6), func_d])
        #v_test = backward_trans(func.obj, x_test, disjoint_group, eig_vec)
        y_test = func.obj(x_test)

        # LHD total for training set and set boundary to the extreme points.
        train_x = func.sparse_grid(grid)
        #v_train = backward_trans(func.obj, train_x, disjoint_group, eig_vec)
        train_y = func.obj(train_x)


        '''
        train_x = np.multiply(train_x, np.array(func.ub)-np.array(func.lb)) + np.array(func.lb)
        x_val = np.multiply(x_val, np.array(func.ub)-np.array(func.lb)) + np.array(func.lb)
        x_test = np.multiply(x_test, np.array(func.ub)-np.array(func.lb)) + np.array(func.lb)
        '''
        var_index = sum(disjoint_group, [])

        train_x = train_x[:, var_index]
        x_val = x_val[:,var_index]
        x_test = x_test[:,var_index]

        #regularization_rate = [1e-3*np.power(.1, n) for n in range(6)]
        """
        regularization_rate = [ 1e-3, 1e-5, 1e-7, 1e-10, 1e-12, 0]
        train_score = []
        test_score = []
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

        regularization=0.
        #exploration = func.exploration_model(regularization_rate = regularization,\
        #                                 activation = 'relu', base_node = 16)
        exploration = func.backward_subspace_model(disjoint_group, regularization_rate = regularization,\
                                        activation = 'relu', base_node = 64)
        exploration = func.exploration_train(exploration, train_x, train_y, x_val, y_val,\
                                            BATCH_SIZE = 32, lr = 1e-2,\
                                            EPOCHS = 200)

        # Fit a linear regression model

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

        # Get weights in linear transformation
        weights = []
        for i in range(num_group):
            layer = exploration.get_layer('linear_'+str(i))
            weights += [layer.get_weights()]

        print(weights)

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

        v_min = []; v_max = []
        for i in range(num_group):
            if len(weights[i]) > 0:
                Min, Max = MinMax(disjoint_group[i], weights[i][0])
                v_min += [Min]
                v_max += [Max]
            else:
                index = disjoint_group[i][0]
                v_min += [lb[index]]
                v_max += [ub[index]]


        #evaulation#
        num_y = func.n
        num_x = func.N1
        name = fig_name + str(ind)
        nonadd_id = [i for i in range(dim) if i not in func.add_id]
        for stage in np.arange(1, num_group): # Plot (x,y,z)

            # 2d path grids for first stage visualization 
            grid_y=func.generate_grid(func_d-len(disjoint_group[stage-1]), num_y) # Take LHD in the y variable's space
            grid_x=func.generate_grid(len(disjoint_group[stage-1]), num_x) # Take LHD in the x variable's space
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
            length = len(disjoint_group[stage-1])
            row_labels = ['x' + str(nonadd_id[e]+1) for e in disjoint_group[stage-1]]
            df = pd.DataFrame({'coeff': [None for i in range(length)],
                            'lower bound': [None for i in range(length)],
                            'upper bound': [None for i in range(length)]},
                            index = row_labels)
            if length == 1:
                i = disjoint_group[stage-1][0]
                table_i = nonadd_id[i]
                df.iloc[0] = [1, func.lb[table_i], func.ub[table_i]] 
            else:
                for i in range(length):
                    index = disjoint_group[stage-1][i]
                    table_index = nonadd_id[index]
                    coeff = weights[stage-1][0].flatten()
                    df.iloc[i] = [coeff[i].round(decimals = 3), func.lb[table_index], func.ub[table_index]]

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
            if stage == num_group-1:
                ax1 = plt.subplot(211, frame_on=False)
                ax1.xaxis.set_visible(False)  # hide the x axis
                ax1.yaxis.set_visible(False)  # hide the y axis
                ax1.table(cellText = df.values, rowLabels = df.index, colLabels = df.columns,loc = "center")
                ax1.set_title('v'+str(stage))
                ax2 = plt.subplot(212, frame_on=False)
                ax2.xaxis.set_visible(False)  # hide the x axis
                ax2.yaxis.set_visible(False)  # hide the y axis

                length2 = len(disjoint_group[stage])
                row_labels = ['x' + str(nonadd_id[e]+1) for e in disjoint_group[stage]]
                df = pd.DataFrame({'coeff': [None for i in range(length2)],
                                'lower bound': [None for i in range(length2)],
                                'upper bound': [None for i in range(length2)]},
                                index = row_labels)
                length2 = len(disjoint_group[stage])
                if length2 == 1:
                    i = disjoint_group[stage][0]
                    table_i = nonadd_id[i]
                    df.iloc[0] = [1, func.lb[table_i], func.ub[table_i]] 
                else:
                    for i in range(length2):
                        index = disjoint_group[stage][i]
                        table_index = nonadd_id[index]
                        coeff = weights[stage][0].flatten()
                        df.iloc[i] = [coeff[i].round(decimals = 3), func.lb[table_index], func.ub[table_index]]
                ax2.table(cellText = df.values, rowLabels = df.index, colLabels = df.columns,loc = "center") # where df is your data frame
                ax2.set_title('v'+str(stage+1))
            else:
                ax = plt.subplot(111, frame_on=False) # no visible frame
                ax.xaxis.set_visible(False)  # hide the x axis
                ax.yaxis.set_visible(False)  # hide the y axis
                ax.set_title('v'+str(stage))
                ax.table(cellText = df.values, rowLabels = df.index, colLabels = df.columns,loc = "center")

            file_path = os.getcwd() + '\\image\\'
            plt.savefig(file_path+name+str(stage)+'_linear.png', dpi = 100)
            # if stage > 1:
            #     func.backward_additive_test(stage, x_plot, y_plot, disjoint_group,\
            #         name, exploration, num_x, num_y, nonadd_id)

            x_plot_reorder = x_plot[:, var_index]

            h_show = func.backward_stage_plot(stage, x_plot_reorder, y_plot, disjoint_group,\
                 name, exploration, num_x, num_y, nonadd_id)
        #func.compound_img(x,  h, index, autoencoder, x_test = x_train, h_num = 5, name = 'sin4.jpg', plot = 2)
        if num_group == 1:
            h = 1e-2
            weight = weights[0][0]
            lower_bound, upper_bound = 0, 0
            for i in range(func_d):
                if weight[i] < 0:
                    lower_bound += weight[i]*func.ub[nonadd_id[i]]
                    upper_bound += weight[i]*func.lb[nonadd_id[i]]
                else:
                    lower_bound += weight[i]*func.lb[nonadd_id[i]]
                    upper_bound += weight[i]*func.ub[nonadd_id[i]]
            vp = np.arange(lower_bound, upper_bound, h)
            yp = func.exploration_decode_single_group(exploration, vp)
            row_labels = ['x' + str(nonadd_id[e]+1) for e in disjoint_group[stage]]
            df = pd.DataFrame({'coeff': [weight[i][0] for i in range(func_d)],
                            'lower bound': [func.lb[nonadd_id[i]] for i in range(func_d)],
                            'upper bound': [func.ub[nonadd_id[i]] for i in range(func_d)]},
                            index = row_labels)
            df['coeff'].round(decimals = 3)
            ax1 = plt.subplot(221)
            ax1.plot(vp, yp)
            ax1.set_xlabel('v1')
            ax1.set_ylabel('f')
            ax1.set_title('visualization')

            ax2 = plt.subplot(222, frame_on=False) # no visible frame
            ax2.xaxis.set_visible(False)  # hide the x axis
            ax2.yaxis.set_visible(False)  # hide the y axis
            ax2.set_title('v1')
            ax2.table(cellText = df.values, rowLabels = df.index, colLabels = df.columns,loc = "center")
            plt.show()
            
        else:      
            func.backward_plot_in_one(name, disjoint_group, row = num_group-1, column = 2)

    print(['{:.2%}'.format(i) for i in test_r2])
