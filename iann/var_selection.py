'''
This code implements gradient projection algorithm in variable selection.

Written by Shengtong Zhang
11/5/2019
'''

import numpy as np
import matplotlib.pyplot as plt
from pyDOE import *
from scipy.stats import multivariate_normal as mvn
import math
from numpy import linalg as LA
import scipy.sparse as sp
import time
from scipy.sparse.linalg import eigs
from test_function import test

def h_func(x, n, cov = 0.05):
    y = np.array([mvn.pdf(xi, mean=[0.25]*n, cov = cov)+mvn.pdf(xi, mean=[0.75]*n, cov=cov) for xi in x])
    return y
    
def quadratic(x):
    f1 = np.multiply(np.square(x[:,0] - 0.5), np.square(np.sum(x[:, 1:4], axis = 1)))
    #f1 =  np.square(np.sum(x[:, :3], axis = 1))
    f = np.multiply(f1, (np.sum(x[:, 4:6], axis = 1))) + np.square(x[:,6] - 0.5) + np.power(x[:,7], 3)
    return f

def obj(x):
    #return np.sin(np.sum(x, axis = 1))
    #return np.sum(np.square(x - 0.5), axis = 1)
    m, n = np.shape(x)   
    t = test()
    #y = np.multiply(x[:,1]-x[:,2], x[:, 0]) + np.square(x[:,0])
    #f = np.square(np.sum(x[:,:3],axis=1)) + np.square(np.sum(x[:,3:], axis = 1))
    #f1 = np.multiply(np.square(x[:,0] - 0.5), np.square(np.sum(x[:, 1:3], axis = 1)))
    #f1 =  np.square(np.sum(x[:, :3], axis = 1))
    #f = np.multiply(f1, (np.sum(x[:, 3:], axis = 1)))
    xs = np.square(x)
    #f = np.multiply(xs[:,0]+ 0.01*xs[:,1], xs[:,2] + xs[:,3])
    f = quadratic(x)
    return np.reshape(f,(-1,1))
    #return np.reshape(np.multiply(np.square(x[:,0]-0.5), np.sum(x[:, 1:], axis = 1)) , (-1, 1))

def grad(obj, x0, index, xj, grid_length):
    '''
    Calculate the gradient of f with respect to x_\j by finite difference.
    The domain is [0,1]^n.
    
    x0: sample in the x_\j space, where we calculate the gradient at x0.
    index: index for j
    xj: xj value
    grid_length: length of grid for finite difference
    '''
    def insert(x, index, y):
        # x and y are two 1d vectors
        # insert y into x based on the total index in len(x) + len(y)
        if index == []:
            return np.array(x)
        vec = np.concatenate((x, y), axis = 0)
        rm_index = [i for i in range(len(vec)) if i not in index]
        vec[index] = y
        vec[rm_index] = x
        return np.array(vec)
    dim = len(x0)
    gradient = np.zeros((dim, 1))
    f0 = obj(np.reshape(insert(x0, index, xj), (1,-1)))
    for i in range(dim):
        x_tmp = np.array(x0)
        if x0[i] + grid_length < 1:
            x_tmp[i] = x0[i] + grid_length
            f_tmp = obj(np.reshape(insert(x_tmp, index, xj), (1,-1)))
            gradient[i] = (f_tmp - f0) / grid_length
        elif x0[i] - grid_length > 0:
            x_tmp[i] = x0[i] - grid_length
            f_tmp = obj(np.reshape(insert(x_tmp, index, xj), (1,-1)))
            gradient[i] = -(f_tmp - f0) / grid_length
        else:
            print('reduce the spacing in the finite difference')
    return gradient

def indexProj(obj, sample_num, dim, grid, N_j = 10, grid_length = 0.05):
    '''
    Automatically detect the critical index to decompose the original function
    sample_num: # samples
    grid: grids in the x_\j space
    dim: dimension of x_\j space
    N_j: number of points in the x_j axis
    grid_length: length of grid for finite difference
    '''

    dx = (np.arange(N_j))/N_j
    max_proj = 0
    crit_index = 0
    E = []
    for index in range(dim+1):
        proj = np.zeros(sample_num)
        div = 0
        for l in range(sample_num):
            pt = grid[l, :]
            gradient = np.zeros((N_j, dim))
            norm = np.zeros(N_j)
            for i in range(N_j):
                gradient[i, :] = grad(pt, index, dx[i], grid_length).transpose()
                norm[i] = LA.norm(gradient[i, :], 2)
            for i in range(N_j):
                for j in range(i+1, N_j):
                    proj[l] += np.abs(np.dot(gradient[i, :], gradient[j,:].transpose()))
                    div += norm[i] * norm[j]
        avg_proj = sum(proj)/div
        E += [1-avg_proj]
        if avg_proj > max_proj:
            max_proj = avg_proj
            crit_index = index
    print(E)
    return crit_index
    

def power_iteration(A, niter = 100):
    tol = 10**(-9)
    Ashape = np.shape(A)
    eigvec = np.random.rand(Ashape[1])    
    for i in range(niter):
        s = np.zeros(Ashape[1])
        # calculate the matrix-by-vector product Ab
        for row in A:
            s += np.dot(row, eigvec.transpose()) * row
        eigvec1 = s
        #eigenvalue
        eigval = eigvec.dot(np.transpose(s))
        if LA.norm(eigval * eigvec - s) < tol:
                break
        # calculate the norm
        eigvec1_norm = np.linalg.norm(eigvec1)
        # re normalize the vector
        eigvec = eigvec1 / eigvec1_norm
    return eigval, eigvec

"""
# For one variable
def indexPCA(obj, sample_num, dim_bsj, grid, N_j = 10, grid_length = 0.05):
    '''
    Automatically detect the critical index to decompose the original function
    sample_num: # samples
    grid: grids in the x_\j space
    dim_bsj: dimension of x_\j space
    N_j: number of points in the x_j axis
    grid_length: length of grid for finite difference
    '''

    dx = (np.arange(N_j))/N_j

    E = []
    for index in range(dim_bsj+1):
        err = np.zeros(sample_num)
        for l in range(sample_num):
            pt = grid[l, :]
            gradient = np.zeros((N_j, dim_bsj))
            for i in range(N_j):
                gradient[i, :] = grad(obj, pt, index, dx[i], grid_length).transpose()
            eigval, eigvec = power_iteration(gradient)
            # calculate PCA error
            for i in range(N_j):
                orth = gradient[i, :] - gradient[i, :].dot(np.transpose(eigvec)) \
                       * eigvec
                err[l] += orth.dot(np.transpose(orth))
        E += [np.mean(err)]
        index = np.argmin(E)
    return index
"""

def indexPCA(obj, grid, rm_index, prev_index = [], N_j = 50, grid_length = 0.05):
    '''
    Automatically detect the critical index to decompose the original function
    sample_num: # samples
    grid: grids in the x_\j space
    rm_index: remaining index of variables in x_\j space.
    prev_index: previous index of variables already selected
    N_j: number of points in the {x_j} space
    grid_length: length of grid for finite difference
    grid_flag: whether delete the last column of grid
    '''
    grid = grid[:,:-1]
    dim_bsj = len(rm_index)
    dim_j = len(prev_index)+1

    #dx = (np.arange(N_j))/N_j
    #if dim_bsj == 1: # only one variable left
    #    return [], prev_index + rm_index
    sample_num, sample_dim = np.shape(grid) # sample_dim = dim_bsj-1

    xj = lhs(dim_j, samples=N_j, criterion = 'm')

    E = []
    for ind in rm_index:
        err = np.zeros(sample_num)
        grad_norm = 0 # Normalization parameter for the mean of summation of gradient vectors.
        for l in range(sample_num):
            pt = grid[l, :]
            gradient = np.zeros((N_j, sample_dim))
            ind_j = prev_index + [ind]
            ind_j.sort()
            for i in range(N_j):
                gradient[i, :] = grad(obj, pt, ind_j, xj[i,:], grid_length).transpose()
            eigval, eigvec = power_iteration(np.dot(gradient.T,gradient))
            # calculate PCA error
            for i in range(N_j):
                orth = gradient[i, :] - gradient[i, :].dot(np.transpose(eigvec)) \
                       * eigvec
                err[l] += orth.dot(np.transpose(orth))
            grad_norm += LA.norm(gradient)**2 # Calculate the gradients L^2 norm.
        grad_norm /= sample_num # Take the mean
        E += [np.mean(err)/grad_norm]
    #print(E)
    index = np.argmin(E)
    prev_index += [rm_index[index]]
    rm_index.pop(index)
    return rm_index, prev_index, E

def group_indexPCA(obj, grid, rm_index, prev_index = [], N_j = 50, grid_length = 0.05):
    '''
    Automatically detect the critical index to decompose the original function
    sample_num: # samples
    grid: grids in the x_\j space
    rm_index: remaining index of variables in x_\j space.
    prev_index: previous index of variables already selected
    N_j: number of points in the {x_j} space
    grid_length: length of grid for finite difference
    '''
    dim_bsj = len(rm_index)
    dim_j = len(prev_index)

    #dx = (np.arange(N_j))/N_j
    #if dim_bsj == 1: # only one variable left
    #    return [], prev_index + rm_index
    sample_num, sample_dim = np.shape(grid) # sample_dim = dim_bsj-1

    xj = lhs(dim_j, samples=N_j, criterion = 'm')

    E = []
    for ind in rm_index:
        err = np.zeros(sample_num)
        for l in range(sample_num):
            pt = grid[l, :]
            gradient = np.zeros((N_j, sample_dim))
            ind_j = prev_index
            ind_j.sort()
            for i in range(N_j):
                gradient[i, :] = grad(obj, pt, ind_j, xj[i,:], grid_length).transpose()
            eigval, eigvec = power_iteration(np.dot(gradient.T,gradient))
            # calculate PCA error
            for i in range(N_j):
                orth = gradient[i, :] - gradient[i, :].dot(np.transpose(eigvec)) \
                       * eigvec
                err[l] += orth.dot(np.transpose(orth))
        E += [np.mean(err)]
    #print(E)
    index = np.argmin(E)
    prev_index += [rm_index[index]]
    rm_index.pop(index)
    return rm_index, prev_index, E

def greedy_indexPCA(obj, rm_index, prev_index = [], N_j = 50, grid_length = 0.05):
    '''
    Add one more variable to the prev_index in a greedy way. Add two projection errors to get the greedy error.
    sample_num: # samples
    rm_index: remaining index of variables in x_\j space.
    prev_index: previous index of variables already selected
    N_j: number of points in the {x_j} space
    grid_length: length of grid for finite difference
    grid_flag: whether delete the last column of grid
    '''
    #grid = grid[:,:-1]
    dim_bsj = len(rm_index)-1
    dim_j = len(prev_index)+1

    #dx = (np.arange(N_j))/N_j
    #if dim_bsj == 1: # only one variable left
    #    return [], prev_index + rm_index
    #sample_num, sample_dim = np.shape(grid) # sample_dim = dim_bsj-1

    xj_prev = lhs(dim_j, samples=N_j, criterion = 'm')
    xj_rm = lhs(dim_bsj, samples=N_j, criterion = 'm')

    E = []
    for ind in rm_index:
        ind_j = prev_index + [ind]
        ind_j.sort()
        err = np.zeros(N_j)
        for l in range(N_j): # fix xj_rm
            pt = xj_rm[l, :]
            gradient = np.zeros((N_j, dim_bsj))
            for i in range(N_j):
                gradient[i, :] = grad(obj, pt, ind_j, xj_prev[i,:], grid_length).transpose()
            eigval, eigvec = power_iteration(np.dot(gradient.T,gradient))
            # calculate PCA error
            for i in range(N_j):
                orth = gradient[i, :] - gradient[i, :].dot(np.transpose(eigvec)) \
                       * eigvec
                err[l] += orth.dot(np.transpose(orth))

        err_rm = np.zeros(N_j)
        ind_bsj = list.copy(rm_index)
        ind_bsj.remove(ind)
        ind_bsj.sort()
        for l in range(N_j): # fix xj_prev
            pt = xj_prev[l, :]
            gradient = np.zeros((N_j, dim_j))
            for i in range(N_j):
                gradient[i, :] = grad(obj, pt, ind_bsj, xj_rm[i,:], grid_length).transpose()
            eigval, eigvec = power_iteration(np.dot(gradient.T,gradient))
            # calculate PCA error
            for i in range(N_j):
                orth = gradient[i, :] - gradient[i, :].dot(np.transpose(eigvec)) \
                       * eigvec
                err_rm[l] += orth.dot(np.transpose(orth))
        E += [np.mean(err)+np.min(err_rm)]
    print(E)
    index = np.argmin(E)
    prev_index += [rm_index[index]]
    rm_index.pop(index)
    return rm_index, prev_index, E
  
    
def greedy_selection(obj, dim, N_j = 50, grid_length = 0.05):
    '''
    Group the variables into two subgroups using greedy algorithm. [prev_index, rm_index]
    N_j: number of points in the {x_j} space
    grid_length: length of grid for finite difference
    '''
    rm_index = list(range(dim))
    prev_index = []
    eta = 1e-8

    error = []
    while len(rm_index)>0:
        rm_index, prev_index, E = greedy_indexPCA(obj, rm_index, prev_index = prev_index, N_j = N_j, grid_length = grid_length)
        error += [min(E)]
        print([rm_index, prev_index])
        if min(E) < eta:
            break
    error = ["%.3e" % ele for ele in error]
    print(error)
    return prev_index
      
def active_subspace(obj, dim, N_j = 500, grid_length = 0.01, eta = 1e-8):
    '''
    Detect the two components in the active subspace method.
    sample_num: # samples
    N_j: number of points in the {x_j} space
    dim: # predictors
    grid_length: length of grid for finite difference
    grid_flag: whether delete the last column of grid
    '''
    def right_multiplication(mat, eta=eta):
            # right multiply a square matrix to transform mat into a block diagonal matrix.
        # mat is (d-1) by 2 matrix
        # eta: threshold for zero
        mat = np.array(mat)
        row, column = np.shape(mat)

        # Switch the first non-zero element to the first column
        for i in range(row):
            if np.abs(mat[i, 0]) < eta and np.abs(mat[i, 1]) >= eta:
                mat[i, 0], mat[i, 1] = mat[i, 1], mat[i, 0]
            elif np.abs(mat[i, 0]) >= eta:
                break
        
        for i in range(row):
            if np.abs(mat[i, 0]) >= eta:
                mat[:, 1] -= mat[:, 0] * mat[i, 1] / mat[i, 0]
                break
        for i in range(row):
            if np.abs(mat[i, 1]) >= eta:
                mat[:, 0] -= mat[:, 1] * mat[i, 0] / mat[i, 1]
                break  
        return mat
        
    def binary_mat(mat, eta=eta):
        # Transform a matrix into a binary matrix based on the threshold eta
        # flag: indicate whether mat can be transformed into a block diagonal matrix.
        r, l = np.shape(mat)
        #if l > 2:
        #    print('the matrix should have the same component with the eigenvectors')
        binary_mat = np.zeros((r, l))
        for i in range(r):
            for j in range(l):
                if np.abs(mat[i, j]) >= eta:
                    binary_mat[i, j] = 1
        return binary_mat
    
    if dim == 1: # only one variable left
        return [1]
    #sample_num, sample_dim = np.shape(grid) # sample_dim = dim_bsj-1

    sample = lhs(dim, samples=N_j, criterion = 'm')

    ind = []
    err = 0
    err2 = 0
    k = 2 # components

    gradient = np.zeros((N_j, dim))
    for i in range(N_j):
        gradient[i, :] = grad(obj, sample[i, :], ind, [0], grid_length).transpose()
    cov_mat = np.dot(gradient.T,gradient)
    #eigval, eigvec = LA.eig(np.dot(gradient.T,gradient))
    eigval, eigvec = LA.eig(cov_mat)
    idx = eigval.argsort()[::-1]   
    eigval = np.real(eigval[idx[:k]])
    eigvec = np.real(eigvec[:,idx[:k]])
    print(eigvec)
    print(right_multiplication(eigvec))

    mean_vec = binary_mat(right_multiplication(eigvec), eta)
    
    # calculate PCA error
    for i in range(N_j):
        orth = gradient[i, :] - gradient[i, :].dot(np.transpose(eigvec[:,0])) * eigvec[:,0]
        orth2 = gradient[i, :] - gradient[i, :].dot(np.transpose(eigvec[:,0])) * eigvec[:,0] \
                - gradient[i, :].dot(np.transpose(eigvec[:,1])) * eigvec[:,1]
        err += orth.dot(np.transpose(orth))
        err2 += orth2.dot(np.transpose(orth2))

    E = np.mean(err)
    E2 = np.mean(err2)
    print([E,E2])

    flag = True
    first_index = []
    second_index = []
    if 1:
        print(mean_vec)
    '''
    for i in range(sample_dim):
        if np.abs(mean_vec[i, 0]) >= zero_threshold:
            mean_vec[i, 0] = 1
            if i < ind:
                first_index += [i]
            else:
                first_index += [i+1]
        if np.abs(mean_vec[i, 1]) >= zero_threshold:
            mean_vec[i, 1] = 1
            if i < ind:
                second_index += [i]
            else:
                second_index += [i+1]
        if mean_vec[i, 0] + mean_vec[i, 1] != 1:
            flag = False
    '''
    return first_index, second_index, [E, E2]

def indexPCA_comp2(obj, grid, ind, rm_index, prev_index = [], N_j = 50, grid_length = 0.05,eta=1e-8):
    '''
    Automatically detect the critical index to decompose the original function
    sample_num: # samples
    grid: grids in the x_\j space
    ind: index for j_1(scalar).
    rm_index: remaining index of variables in x_\j space.
    prev_index: previous index of variables already selected.
    N_j: number of points in the {x_j} space
    grid_length: length of grid for finite difference
    eta: threshold for zero
    '''

    def right_multiplication(mat, eta=eta):
        # right multiply a square matrix to transform mat into a block diagonal matrix.
        # mat is (d-1) by 2 matrix
        # eta: threshold for zero
        mat = np.array(mat)
        row, column = np.shape(mat)

        # Switch the first non-zero element to the first column
        for i in range(row):
            if np.abs(mat[i, 0]) < eta and np.abs(mat[i, 1]) >= eta:
                mat[i, 0], mat[i, 1] = mat[i, 1], mat[i, 0]
            elif np.abs(mat[i, 0]) >= eta:
                break
        
        for i in range(row):
            if np.abs(mat[i, 0]) >= eta:
                mat[:, 1] -= mat[:, 0] * mat[i, 1] / mat[i, 0]
                break
        for i in range(row):
            if np.abs(mat[i, 1]) >= eta:
                mat[:, 0] -= mat[:, 1] * mat[i, 0] / mat[i, 1]
                break  
        return mat
    
    def binary_mat(mat, eta=eta):
        # Transform a matrix into a binary matrix based on the threshold eta
        # flag: indicate whether mat can be transformed into a block diagonal matrix.
        r, l = np.shape(mat)
        #if l > 2:
        #    print('the matrix should have the same component with the eigenvectors')
        binary_mat = np.zeros((r, l))
        for i in range(r):
            for j in range(l):
                if np.abs(mat[i, j]) >= eta:
                    binary_mat[i, j] = 1
        return binary_mat

    k = 2
    scalar_index = -1
    grid = grid[:,:-1]
    #dx = (np.arange(N_j))/N_j
    dim_bsj = len(rm_index)
    dim_j = len(prev_index) + 1
    sample_num, sample_dim = np.shape(grid) # sample_dim = dim_bsj-1

    xj = lhs(dim_j, samples=N_j, criterion = 'm')

    E = []
    E2 = []
    E3 = []

    err = np.zeros(sample_num)
    err2 = np.zeros(sample_num)
    mean_vec = np.zeros((sample_dim, k))
    mean_val = np.zeros(k)
    for l in range(sample_num):
        pt = grid[l, :]
        gradient = np.zeros((N_j, sample_dim))
        ind_j = prev_index + [ind]
        ind_j.sort()

        mean_vec = np.zeros((sample_dim, k))
        for i in range(N_j):
            gradient[i, :] = grad(obj, pt, ind_j, xj[i,:], grid_length).transpose()
        
        # Test: Whether the second component is a constant
        cov_mat = np.dot(gradient.T,gradient)
        #eigval, eigvec = LA.eig(np.dot(gradient.T,gradient))
        eigval, eigvec = LA.eig(cov_mat)
        idx = eigval.argsort()[::-1]   
        eigval = np.real(eigval[idx[:k]])
        eigvec = np.real(eigvec[:,idx[:k]])
        #print(right_multiplication(eigvec))
        mean_vec += binary_mat(right_multiplication(eigvec), eta)
        mean_val += eigval
        
        # calculate PCA error
        for i in range(N_j):
            orth = gradient[i, :] - gradient[i, :].dot(np.transpose(eigvec[:,0])) * eigvec[:,0]
            orth2 = gradient[i, :] - gradient[i, :].dot(np.transpose(eigvec[:,0])) * eigvec[:,0] \
                    - gradient[i, :].dot(np.transpose(eigvec[:,1])) * eigvec[:,1]
            err[l] += orth.dot(np.transpose(orth))
            err2[l] += orth2.dot(np.transpose(orth2))

    E += [np.mean(err)]
    E2 += [np.mean(err2)]
    #print('E = %s' % E)
    #print('E2 = %s' % E2)
    if E[0] < eta: # case 1: (j, \j) 
        rm_index.pop(ind)
        status = 1
        return  [ind, rm_index], status, E, E2
    elif E2[0] < eta:
        # if block diagonal eigenvector
            # return [ind, first_index, second_index], status = 2
        flag = True
        zero_threshold = 5/sample_dim
        first_index = []
        second_index = []
        if 0:
            print(mean_vec)
        for i in range(sample_dim):
            if np.abs(mean_vec[i, 0]) >= zero_threshold:
                mean_vec[i, 0] = 1
                if i < ind:
                    first_index += [i]
                else:
                    first_index += [i+1]
            if np.abs(mean_vec[i, 1]) >= zero_threshold:
                mean_vec[i, 1] = 1
                if i < ind:
                    second_index += [i]
                else:
                    second_index += [i+1]
            if mean_vec[i, 0] + mean_vec[i, 1] != 1:
                flag = False
        if flag:
            status = 2
            return [ind, first_index, second_index], status, E, E2
        '''
        # Else if PCA with 2 components and one constant vector
        err_center = []
        mean_vec = np.zeros((sample_dim, 1))
        for l in range(sample_num):
            pt = grid[l, :]
            gradient = np.zeros((N_j, sample_dim))
            ind_j = prev_index + [ind]
            ind_j.sort()

            mean_vec = np.zeros((sample_dim, k))
            for i in range(N_j):
                gradient[i, :] = grad(obj, pt, ind_j, xj[i,:], grid_length).transpose()
            cov_mat = np.dot(np.cov(gradient.T))
            eigval, eigvec = LA.eig(cov_mat)
            idx = eigval.argsort()[::-1]   
            eigval = np.real(eigval[idx[:1]])
            eigvec = np.real(eigvec[:,idx[:1]])
            mean_vec += eigvec

            for i in range(N_j):
                orth = gradient[i, :] - gradient[i, :].dot(np.transpose(eigvec[:,0])) * eigvec[:,0]
                err_center[l] += orth.dot(np.transpose(orth))
        
        E3 += [np.mean(err_center)]
        mean_vec /= sample_num
        mean_vec = binary_mat(sample_num)
        print(mean_vec)
        first_index = []
        second_index = []

        if E3[0] < eta:
            status = 3
            return [ind, first_index, second_index], status
        '''    
    return [ind], 4, E, E2

    '''
    # Summarize the nonzero elements of two principal components.
    mean_vec /= sample_num # The mean_vec is not very robust.
    mean_val /= sample_num 
    ind_vec = np.zeros((sample_dim, k)) # 0 means that index is zero
    first_index = []; second_index = []
    flag = True
    for i in range(sample_dim):
        if np.abs(mean_vec[i, 0]) >= eta:
            ind_vec[i, 0] = 1
            if i < ind:
                first_index += [i]
            else:
                first_index += [i+1]
        if np.abs(mean_vec[i, 1]) >= eta:
            ind_vec[i, 1] = 1
            if i < ind:
                second_index += [i]
            else:
                second_index += [i+1]
        if ind_vec[i, 0] + ind_vec[i, 1] != 1:
            flag = False
    '''

    
def variable_selection(obj, grid, N_j = 50, grid_length = 0.05):
    '''
    rank the indices of variables based on the gradient PCA method
    grid: grids in the R^{dim} space
    N_j: number of points in the {x_j} space
    grid_length: length of grid for finite difference
    '''
    sample_num, dim = np.shape(grid) # sample_dim = dim-1
    rm_index = list(range(dim))
    prev_index = []

    error = []
    while len(rm_index) > 0:
        rm_index, prev_index, E = indexPCA(obj, grid[:, rm_index], rm_index, prev_index = prev_index, N_j = N_j, grid_length = grid_length)
        error += [min(E)]
    error = ["%.3e" % ele for ele in error]
    print(error)
    return prev_index


def group_comp2(obj, grid, N_j = 50, grid_length = 0.05):
    '''
    Determine whether the scalar index belongs to the first_index or the second_index
    '''
    sample_num, dim = np.shape(grid)
    rm_index = list(range(dim))
    for ind in rm_index:
        scalar_index, first_index, second_index, flag = indexPCA_comp2(obj, grid, ind, rm_index, N_j = N_j, grid_length = grid_length)
        if flag:
            prev_index = list(second_index)
            rm, prev, E = group_indexPCA(obj, grid[:, first_index+[scalar_index]], [scalar_index], prev_index = prev_index, N_j = N_j, grid_length = grid_length)
            print(E)
            if E[0] < 1e-12:
                return [[scalar_index, first_index], second_index]
            else:
                first_index, second_index = second_index, first_index
                prev_index = list(second_index)
                rm_index, prev_index, E = group_indexPCA(obj, grid[:, first_index+[scalar_index]], [scalar_index], prev_index = prev_index, N_j = N_j, grid_length = grid_length)
                print(E)
                if E[0] < 1e-12:
                    return [[scalar_index, first_index], second_index]
                else:
                    continue
    print('the function cannot be decomposed into two distinct subgroups')

def variable_selection_alg(obj, grid, N_j = 50, grid_length = 0.05, eta= 1e-8):
    # Total algorithm to select the variable
    sample_num, dim = np.shape(grid)
    rm_index = list(range(dim))
    error1 = []
    error2 = []
    for ind in rm_index:
        index, status, E, E2 = indexPCA_comp2(obj, grid, ind, rm_index, N_j = N_j, grid_length = grid_length, eta = eta)
        #print(index)
        error1 += E
        error2 += E2
        if status == 1:
            print('the variable x_%d satisfy the decomposition (j, \j).' % (ind+1))
        if status == 2:
            prev_index = list(index[2])
            first_index = index[1]
            second_index = index[2]
            rm, prev, E = group_indexPCA(obj, grid[:, first_index+[ind]], [ind], prev_index = prev_index, N_j = N_j, grid_length = grid_length)
            if E[0] < eta:
                index = [[ind, first_index], second_index]
                print('the variable x_%d satisfy the decomposition (j, J_2, J_3).' % (ind+1))
                print('the index is %s' % index)
            else:
                first_index, second_index = second_index, first_index
                prev_index = list(second_index)
                rm_index, prev_index, E = group_indexPCA(obj, grid[:, first_index+[ind]], [ind], prev_index = prev_index, N_j = N_j, grid_length = grid_length)
                if E[0] < 1e-12:
                    index = [[ind, first_index], second_index]
                    print('the variable x_%d satisfy the decomposition (j, J_2, J_3).' % (ind+1))
                    print('the index is %s' % index)
    
    error1 = ["%.3e" % ele for ele in error1]
    error2 = ["%.3e" % ele for ele in error2]
    print('error_1 is %s' % error1)
    print('error_2 is %s' % error2)

def backward_indexPCA(obj, grid, grad_index, nongrad_index, candidate_index, N_j = 50, grid_length = 0.05):
    '''
    Automatically detect the critical index to decompose the original function;
    each time take an index from nongrad_index into grad_index and calculate the projection error.
    sample_num: # samples
    grid: grids in the x_{nongrad_index} space
    grad_index: calculate the gradients for these index
    nongrad_index: do not calculate the gradients for these index
    candidate_index: add the variable in the candidate index into grad_index
    N_j: number of points in the {x_j} space
    grid_length: length of grid for finite difference
    grid_flag: whether delete the last column of grid
    '''
    dim_grad = len(grad_index)+1
    dim_nongrad = len(nongrad_index)-1

    #dx = (np.arange(N_j))/N_j
    #if dim_bsj == 1: # only one variable left
    #    return [], prev_index + rm_index
    sample_num, dim = np.shape(grid) # sample_dim = dim_nongrad-1

    #x_grad = lhs(dim_grad, samples=N_j, criterion = 'm')

    E = []
    k = 1
    for ind in candidate_index:
        err = 0
        grad_norm = 0 # Normalization parameter for the mean of summation of gradient vectors.
        gradient = np.zeros((sample_num, dim_grad))
        nongrad_ind = list.copy(nongrad_index)
        nongrad_ind.remove(ind)
        nongrad_ind.sort()
        grad_ind = grad_index + [ind]
        grad_ind.sort()
        for i in range(sample_num):
            gradient[i, :] = grad(obj, grid[i,grad_ind], nongrad_ind, grid[i, nongrad_ind], grid_length).T
        cov_mat = np.dot(gradient.T,gradient)
        eigval, eigvec = LA.eig(cov_mat)
        idx = eigval.argsort()[::-1]   
        eigval = np.real(eigval[idx[:k]])
        eigvec = np.real(eigvec[:,idx[:k]])
        # calculate PCA error
        for i in range(sample_num):
            orth = gradient[i, :] - gradient[i, :].dot(np.transpose(eigvec.flatten())) \
                    * eigvec.flatten()
            err += orth.dot(np.transpose(orth))
        grad_norm += LA.norm(gradient)**2 # Calculate the gradients L^2 norm.
        E += [np.mean(err)/grad_norm]
    #print(E)
    index = np.argmin(E)
    grad_index += [candidate_index[index]]
    nongrad_index.remove(candidate_index[index])
    candidate_index.pop(index)
    return grad_index, nongrad_index, E

def disjoint_subspace(obj, dim, grid, N_j = 50, grid_length = 0.01, eta = 1e-4):
    """
    Calculate the disjoint subgroups of indexes.
    """
    index = list(range(dim))
    total_index = list(range(dim))
    tmp_index = []
    disjoint_group = []
    while len(index)>0:
        grad_index = [index[0]]
        candidate_index = list.copy(index)
        candidate_index.remove(index[0])
        nongrad_index = [element for element in total_index if element not in grad_index]
        while len(candidate_index) > 0:
            grad_, nongrad_, E = backward_indexPCA(obj, grid, grad_index, nongrad_index, candidate_index, N_j = N_j, grid_length=grid_length)
            print(E)
            if np.min(E) > eta:
                nongrad_index += [grad_index[-1]]
                grad_index.pop(-1)
                print(grad_index)
                print(nongrad_index)
                break
        index = [element for element in index if element not in grad_index]
        disjoint_group += [grad_index]
    return disjoint_group

def disjoint_indexPCA(obj, disjoint_group, grid, rm_index, prev_index = [], N_j = 50, grid_length = 0.05):
    '''
    Automatically detect the critical index of subgroup to decompose the disjoint_group
    sample_num: # samples
    grid: grids in the x_\j space
    rm_index: remaining index of variables in x_\j space.
    prev_index: previous index of variables already selected
    N_j: number of points in the {x_j} space
    grid_length: length of grid for finite difference
    grid_flag: whether delete the last column of grid
    '''
    num_group = len(disjoint_group)
    rm_list = sum(rm_index, [])
    prev_list = sum(prev_index, [])

    dim_bsj = len(rm_list)
    dim_j = len(prev_list)
    dim = dim_j+dim_bsj

    #dx = (np.arange(N_j))/N_j
    #if dim_bsj == 1: # only one variable left
    #    return [], prev_index + rm_index
    sample_num, sample_dim = np.shape(grid) # sample_dim = dim_bsj-1

    xj = lhs(dim, samples=N_j, criterion = 'm')

    E = []
    k = 1
    for ind in rm_index:
        err = np.zeros(sample_num)
        grad_norm = 0 # Normalization parameter for the mean of summation of gradient vectors.
        for l in range(sample_num):
            ind_j = prev_list + ind
            ind_j.sort()
            ind_bsj = [e for e in range(dim) if e not in ind_j]
            pt = grid[l, ind_bsj]
            gradient = np.zeros((N_j, len(ind_bsj)))
            for i in range(N_j):
                gradient[i, :] = grad(obj, pt, ind_j, xj[i,ind_j], grid_length).transpose() 
            cov_mat = np.dot(gradient.T,gradient)
            eigval, eigvec = LA.eig(cov_mat)
            idx = eigval.argsort()[::-1]   
            eigval = np.real(eigval[idx[:k]])
            eigvec = np.real(eigvec[:,idx[:k]])
            # calculate PCA error
            for i in range(N_j):
                orth = gradient[i, :] - gradient[i, :].dot(np.transpose(eigvec.flatten())) \
                       * eigvec.flatten()
                err[l] += orth.dot(np.transpose(orth))
            grad_norm += LA.norm(gradient)**2 # Calculate the gradients L^2 norm.
        grad_norm /= sample_num # Take the mean
        E += [np.mean(err)/grad_norm]
    #print(E)
    index = np.argmin(E)
    prev_index += [rm_index[index]]
    rm_index.pop(index)
    return rm_index, prev_index, E

def disjoint_selection(obj, disjoint_group, grid, init_index = [], N_j = 50, grid_length = 0.05):
    # Reorder the subgroups in "disjoint_group" by minimizing the projection error
    num_group = len(disjoint_group)
    if len(init_index)>0:
        prev_index = [i for i in disjoint_group if init_index[0] in i]
        rm_index = [i for i in disjoint_group if i not in prev_index]
    else:
        prev_index = []
        rm_index = disjoint_group

    error = []
    while len(rm_index) > 0:
        rm_index, prev_index, E = disjoint_indexPCA(obj, disjoint_group, grid, rm_index, prev_index = prev_index, N_j = N_j, grid_length = grid_length)
        error += [min(E)]
        print(E)
        print(rm_index)
    error = ["%.3e" % ele for ele in error]
    print(error)
    return prev_index

def additive(obj, grid, N_j = 50, grid_length = 0.01, eta = 1e-4):
    '''
    Seperate Additive Variable
    grid: grids in the R^d space
    N_j: number of points in the {x_j} space
    grid_length: length of grid for finite difference
    eta: threshold
    '''
    grid = np.array(grid)
    sample_num, sample_dim = np.shape(grid) # sample_dim = dim_bsj-1

    xj = (np.arange(N_j))/N_j

    E = []
    add_id = []
    for ind in range(sample_dim):
        err = np.zeros(sample_num)
        gradient = np.zeros((N_j, sample_num))
        ind_bsj = [e for e in range(sample_dim) if e != ind]
        for i in range(N_j):
            for l in range(sample_num):
                pt = grid[l, ind_bsj]
                gradient[i, l] = grad(obj, np.array([xj[i]]), ind_bsj, pt, grid_length)[0][0]
        var = np.var(gradient, axis = 1)
        var = np.mean(var)
        grad_norm = LA.norm(np.mean(gradient, axis = 1))  # Calculate the gradients L^2 norm.
        E += [var/grad_norm]
        if var/grad_norm < eta:
            add_id += [ind]
    print(E)
    return add_id

    
if __name__ == '__main__':

    dim = 8
    samples = 10
    grid = lhs(dim, samples = samples, criterion= 'm')
    #x0 = lhs(dim, samples=10, criterion = 'm')
    #x = CCS2(samples,dim, 0.05, 0, x0=x0)
    #grid = lhs(dim, samples = samples, criterion = 'm')

    #rm_index = list(range(dim))
    #scalar_index, first_index, second_index, mean_vec = indexPCA_comp2(obj, x, rm_index)
    #dg = disjoint_subspace(obj, dim, grid, eta=1e-4)
    #dg = disjoint_selection(obj, dg, grid, init_index = [0])
    add_id = additive(obj, grid)
    print(add_id)

"""
start1 = time.process_time()
print(indexProj(samples, dim, x, N_j = 20))
elapsed1 = (time.process_time() - start1)

start2 = time.process_time()
print(indexPCA(samples, dim, x, N_j = 20))
elapsed2 = (time.process_time() - start2)
print('Time used for projection:', elapsed1)
print('Time used for PCA:', elapsed2)
'''
xk = np.arange(50)/50
plot_x, plot_y = np.meshgrid(xk, xk)
#plot_z = (plot_x-0.5)**2 + (plot_y-0.5)**2
plot_z = np.reshape(h_func(np.column_stack((plot_x.flatten(), plot_y.flatten())), dim), plot_x.shape)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.contour(plot_x, plot_y, plot_z, 10)
ax.plot(x[:,0], x[:,1], '.r')
plt.title('Contour plot of h with step size %.2f' % 0.05)
plt.xlabel('(x_{\j})^1')
plt.ylabel('(x_{\j})^2')
#plt.savefig('CCD_linear_50.jpg')

plt.show()
'''

"""