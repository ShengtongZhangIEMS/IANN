'''
This code implements gradient projection algorithm in variable selection.

Written by Shengtong Zhang
11/5/2019
'''

import numpy as np
import matplotlib.pyplot as plt
from pyDOE import *
import tensorflow as tf
from numpy import linalg as LA
from func.test_function import test

class TreeNode:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
    def isLeaf(self):
        if (self.left == None and self.right == None):
            return True
        else:
            return False
        
def h_func(x, n, cov = 0.05):
    y = np.array([mvn.pdf(xi, mean=[0.25]*n, cov = cov)+mvn.pdf(xi, mean=[0.75]*n, cov=cov) for xi in x])
    return y
    
def quadratic(x):
    f1 = np.multiply(np.square(x[:,0] - 0.5), np.square(np.sum(x[:, 1:4], axis = 1)))
    #f1 =  np.square(np.sum(x[:, :3], axis = 1))
    f = np.multiply(f1, (np.sum(x[:, 4:6], axis = 1))) + np.square(x[:,6] - 0.5) + np.power(x[:,7], 3)
    return f

# def obj(x):
#     #return np.sin(np.sum(x, axis = 1))
#     #return np.sum(np.square(x - 0.5), axis = 1)
#     m, n = np.shape(x)   
#     t = test()
#     #y = np.multiply(x[:,1]-x[:,2], x[:, 0]) + np.square(x[:,0])
#     #f = np.square(np.sum(x[:,:3],axis=1)) + np.square(np.sum(x[:,3:], axis = 1))
#     #f1 = np.multiply(np.square(x[:,0] - 0.5), np.square(np.sum(x[:, 1:3], axis = 1)))
#     #f1 =  np.square(np.sum(x[:, :3], axis = 1))
#     #f = np.multiply(f1, (np.sum(x[:, 3:], axis = 1)))
#     xs = np.square(x)
#     #f = np.multiply(xs[:,0]+ 0.01*xs[:,1], xs[:,2] + xs[:,3])
#     f = quadratic(x)
#     return np.reshape(f,(-1,1))
#     #return np.reshape(np.multiply(np.square(x[:,0]-0.5), np.sum(x[:, 1:], axis = 1)) , (-1, 1))

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

def grad_mat(obj, x0, index, xj, grid_length):
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
        assert x.shape[0] == y.shape[0]
        vec = np.concatenate((x, y), axis = 1)
        rm_index = [i for i in range(vec.shape[1]) if i not in index]
        vec[:, index] = y
        vec[:, rm_index] = x
        return np.array(vec)
    dim = len(x0)
    n = xj.shape[0]
    gradient = np.zeros((n, dim))
    x0 = np.repeat(x0.reshape((1, -1)), n, axis = 0) # origin of the gradient
    x0_grad = insert(x0, index, xj)
    f0 = obj(x0_grad)
    
    for i in range(dim):
        x_tmp = np.array(x0)
        x_tmp[:,i] = np.minimum(x_tmp[:,i]+grid_length, 1 + 0*x_tmp[:,i]) # Finite difference cannot exceed the boundary 1.
        f_tmp = obj(insert(x_tmp, index, xj))
        gradient[:,i] = ((f_tmp - f0) / grid_length).ravel()
    
    return gradient

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
        grad_norm = 0. # Normalization parameter for the mean of summation of gradient vectors.
        for l in range(sample_num):
            pt = grid[l, :]
            gradient = np.zeros((N_j, sample_dim))
            ind_j = prev_index + [ind]
            ind_j.sort()
            # for i in range(N_j):
            #     gradient[i, :] = grad(obj, pt, ind_j, xj[i,:], grid_length).transpose()
            
            gradient = grad_mat(obj, pt, ind_j, xj, grid_length)
            
            eigval, eigvec = power_iteration(np.dot(gradient.T,gradient))
            # calculate PCA error
            for i in range(N_j):
                orth = gradient[i, :] - gradient[i, :].dot(np.transpose(eigvec)) \
                       * eigvec
                err[l] += orth.dot(np.transpose(orth))
            grad_norm += LA.norm(gradient)**2 # Calculate the gradients L^2 norm.
        grad_norm /= sample_num # Take the mean
        grad_norm = np.max((grad_norm, 1e-6))
        E += [np.mean(err)/grad_norm]
    print(E)
    index = np.argmin(E)
    prev_index += [rm_index[index]]
    rm_index.pop(index)
    return rm_index, prev_index, E

def Grad(obj, grid, grid_length, lb, ub):
    sample_num, dim = grid.shape
    grad = np.zeros((sample_num, dim))
    f_0 = obj(grid)
    for d in range(dim):
        grid_tmp = np.copy(grid)
        grid_tmp[:, d] = grid[:, d] + grid_length
        f_tmp = obj(grid_tmp)
        diff = ub[d]-lb[d] if ub[d] > lb[d] else np.float('inf')
        grad[:,d] = np.ravel(f_tmp - f_0) / grid_length / diff
    return grad

def Hessian(obj, grid, grid_length, lb, ub):
    sample_num, dim = grid.shape
    hess = np.zeros((sample_num, dim, dim))
    f_0 = obj(grid)
    for i in range(dim):
        for j in range(dim):
            if j < i:
                hess[:, i, j] = hess[:, j, i]
                
            grid_x, grid_y, grid_xy = np.copy(grid), np.copy(grid), np.copy(grid)
            grid_x[:,i] = grid[:, i] + grid_length
            grid_y[:,j] = grid[:, j] + grid_length
            grid_xy[:,[i,j]] = grid[:, [i,j]] + grid_length
            diff_i = ub[i] - lb[i] if ub[i] > lb[i] else np.float('inf')
            diff_j = ub[j] - lb[j] if ub[j] > lb[j] else np.float('inf')
                
            hess[:, i, j] = np.ravel(obj(grid_xy) - obj(grid_x) - obj(grid_y) + f_0) / (grid_length**2 * diff_i * diff_j)
    
    return hess

def HessRegErr(obj, grid, grad, hess, rm_index, prev_index):
    '''
    Caclulate the Regression Error in the Hessian Matrix
    grad: gradient vector for all grids. (grid.shape[0], d)
    hess: Hessian matrix for all grids. (grid.shape[0], d, d)
    '''

    sample_num, sample_dim = np.shape(grid) # sample_dim = dim_bsj-1

    E = []
    for ind in rm_index:
        group_idx = [i for i in rm_index if i != ind]
        out_group_idx = prev_index + [ind]
        err = np.zeros(sample_num)
        for l in range(sample_num):
            vec_norm = np.sum(np.square(grad[l, group_idx]))
            if vec_norm > 1e-8:
                for rm in out_group_idx:
                    err[l] += np.sum(np.square(hess[l, group_idx, rm])) - \
                        np.square(np.dot(hess[l, group_idx, rm], grad[l, group_idx].reshape((-1,1)))) / vec_norm                
                
        E += [np.mean(err)]
    print(E)
    index = np.argmin(E)
    prev_index += [rm_index[index]]
    rm_index.pop(index)
    return rm_index, prev_index, E
    
def variable_selection_hessian(obj, grid, lb, ub, grid_length = 0.01, init_index = []):
    '''
    rank the indices of variables based on the gradient PCA method
    grid: grids in the R^{dim} space
    grid_length: length of grid for finite difference
    '''
    grad = Grad(obj, grid, grid_length, lb, ub)
    hess = Hessian(obj, grid, grid_length, lb, ub)
    sample_num, dim = np.shape(grid) # sample_dim = dim-1
    rm_index = list(range(dim))
    prev_index = []
    if init_index:
        prev_index = init_index
        rm_index = [e for e in rm_index if e not in init_index]
        

    error = []
    while len(rm_index) > 1:
        print(rm_index)
        rm_index, prev_index, E = HessRegErr(obj, grid, grad, hess, rm_index, prev_index = prev_index)
        error += [min(E)]
    error = ["%.3e" % ele for ele in error]
    print(error)
    prev_index = prev_index + rm_index
    return prev_index

def TreeHessRegErr(obj, grid, grad, hess, groups, reg, overlap):
    '''
    Caclulate the Regression Error for each pair of the remaining p groups
    grad: gradient vector for all grids. (grid.shape[0], d)
    hess: Hessian matrix for all grids. (grid.shape[0], d, d)
    
    return:
        group_index [j1, j2]
        groups: new group with p-1 groups
    '''

    sample_num, sample_dim = np.shape(grid) # sample_dim = dim_bsj-1
    p = len(groups) 
    
    E = dict()
    for i in range(p):
        for j in range(i+1, p):
            group_idx = groups[i]+groups[j]
            if not any([ele in overlap for ele in group_idx]):
                regularization = reg*len(group_idx)
                
                out_group_idx = [i for i in range(sample_dim) if i not in group_idx]
                err = np.zeros(sample_num)
                for l in range(sample_num):
                    vec_norm = np.sum(np.square(grad[l, group_idx]))
                    if vec_norm > 1e-8:
                        for rm in out_group_idx:
                            err[l] += np.sum(np.square(hess[l, group_idx, rm])) - \
                                np.square(np.dot(hess[l, group_idx, rm], grad[l, group_idx].reshape((-1,1)))) / vec_norm
                            
                # Normalize by the number of elements in the Hessian block
                num_blocks = len(group_idx)*len(out_group_idx) if (len(group_idx) > 0 and len(out_group_idx) > 0) else 1
                E[(i,j)] = abs(err.mean()*regularization / num_blocks)
    print(E)
    group_index, min_error = min(E.items(), key=lambda x: x[1])
    group_index = list(group_index)
    # remove the two groups, combine them and put at the end.
    groups.append(groups[group_index[0]]+groups[group_index[1]])
    groups = [g for i, g in enumerate(groups) if i not in group_index]
    return group_index, groups
    
def variable_selection_tree(obj, grid, lb, ub, grid_length = 0.01, reg = 1, overlap = []):
    '''
    rank the indices of variables based on the gradient PCA method
    grid: grids in the R^{dim} space
    grid_length: length of grid for finite difference
    reg: reglularization coefficient to balance the tree
    '''
    grad = Grad(obj, grid, grid_length, lb, ub)
    hess = Hessian(obj, grid, grid_length, lb, ub)
    sample_num, dim = np.shape(grid) # sample_dim = dim-1
    
    groups = [[i] for i in range(dim)]

    # error = []
    group_indices = []
    while len(groups)-len(overlap) > 1:
        group_index, groups = TreeHessRegErr(obj, grid, grad, hess, groups, reg, overlap)
        print(groups[-1])
        group_indices.append(group_index)
        # rm_index, prev_index, E = HessRegErr(obj, grid, grad, hess, rm_index, prev_index = prev_index, grid_length = grid_length)
        # error += [min(E)]
    return group_indices

def index_by_level(prev_index, group_index):
    # n - 1 levels in total, each level group two variables
    # prev_index: index from the previous level
    # group_index: [j1 ,j2] two index forms the group of this level
    if max(group_index) > len(prev_index):
        print('Wrong index to group \n')
    index = [ele for i, ele in enumerate(prev_index) if i not in group_index]
    group_node = TreeNode((prev_index[group_index[0]].data, prev_index[group_index[1]].data))
    group_node.left = prev_index[group_index[0]]
    group_node.right = prev_index[group_index[1]]
    index.append(group_node)
    return index
    
def index_tree(dim, group_indices):
    prev_index = [TreeNode(i) for i in range(dim)]
    for i in range(len(group_indices)):
        prev_index = index_by_level(prev_index, group_indices[i])
    tree_data = [i.data for i in prev_index]
    print(tree_data)
    return prev_index[0], tree_data

def HessRegErrSingle(obj, grid, grad, hess, group_i, group_j):
    '''
    Caclulate the Regression Error for H_ij with scalar j
    grad: gradient vector for all grids. (grid.shape[0], d)
    hess: Hessian matrix for all grids. (grid.shape[0], d, d)
    group_i: multiple variables
    group_j: scalar variable
    
    return:
        err: Hessian Regression Error for H_ij
    '''

    sample_num, sample_dim = np.shape(grid) # sample_dim = dim_bsj-1
    
    err = np.zeros(sample_num)
    for l in range(sample_num):
        vec_norm = np.sum(np.square(grad[l, group_i]))
        if vec_norm > 1e-8:
            err[l] += np.sum(np.square(hess[l, group_i, group_j])) - \
                np.square(np.dot(hess[l, group_i, group_j], grad[l, group_i].reshape((-1,1)))) / vec_norm
                
    # Normalize by the number of elements in the Hessian block
    num_blocks = len(group_i)
    err = abs(err.mean() / num_blocks)
    
    return err

def HessRegErrGroup(grid, grad, hess, group_i, group_j):
    '''
    Caclulate the Regression Error for H_ij with both i, j as groups
    grad: gradient vector for all grids. (grid.shape[0], d)
    hess: Hessian matrix for all grids. (grid.shape[0], d, d)
    group_i: multiple variables
    group_j: multiple variables
    
    return:
        err: Hessian Regression Error for H_ij
    '''
    if min(len(group_i), len(group_j)) < 1:
        return 0
    
    sample_num, sample_dim = np.shape(grid) # sample_dim = dim_bsj-1
    
    err = np.zeros(sample_num)
    for l in range(sample_num):
        vec_norm_i = np.sum(np.square(grad[l, group_i]))
        vec_norm_j = np.sum(np.square(grad[l, group_j]))
        grad_i = grad[l, group_i].reshape((-1,1))
        grad_j = grad[l, group_j].reshape((-1,1))
        hess_block = hess[l][np.ix_(group_i, group_j)]
        if min(vec_norm_i, vec_norm_j) > 1e-8:
            err[l] += np.sum(np.square(hess_block)) - \
                np.square(np.sum(np.multiply(hess_block, grad_i.dot(grad_j.T)))) / (vec_norm_i * vec_norm_j) 
                
    # Normalize by the number of elements in the Hessian block
    num_blocks = len(group_i) * len(group_j)
    err = abs(err.mean() / num_blocks)
    
    return err

def two_groups_greedy(grid, grad, hess, vars, reg):
    '''
    Find the two groups J1 and J2 satisfy the structure f(x) = g(h_1(x_0, x_J1), h_2(x_0, x_J2))
    Function works in a greedy way.
    return:
        regularized error and two groups J1 and J2.
    '''
    
    groups_err = dict()
    first_group = []
    second_group = list(vars)
    cnt = 0
    normalization = 0
    while len(second_group) > 1:
        err = []
        for idx in second_group:
            group_i = first_group + [idx]
            group_j = [i for i in second_group if i != idx]
            group_err = HessRegErrGroup(grid, grad, hess, group_i, group_j) 
            err.append(group_err)
        min_idx = np.argmin(err)
        # print(second_group[min_idx])
        first_group += [second_group[min_idx]]
        second_group.pop(min_idx)        
        reg_coef = np.exp(reg*(abs(len(second_group) - len(first_group))))
        normalization += reg_coef
        min_err = reg_coef*min(err)
        groups_err[cnt] = (min_err, list(first_group), list(second_group))
        # print(err, list(first_group), list(second_group))
        cnt += 1
    
    opt_err = [groups_err[k][0]/normalization for k in groups_err.keys()]
    print(opt_err)
    key = list(groups_err.keys())[np.argmin(opt_err)]    
    
    return np.min(opt_err), groups_err[key][1], groups_err[key][2]

def overlapping_stepwise(obj, grid, lb, ub, grid_length = 0.01, reg = 1, thres = 1e-6):
    '''
        Find the overlappig structure f(x) = g(h_1(x_0, x_J1), h_2(x_0, x_J2)) in a stepwise manner
    '''
    grad = Grad(obj, grid, grid_length, lb, ub)
    hess = Hessian(obj, grid, grid_length, lb, ub)
    
    _, dim = np.shape(grid) # sample_dim = dim-1
    
    # No overlapping
    hess_err, J1, J2 = two_groups_greedy(grid, grad, hess, [i for i in range(dim)], reg)
    print('Non-onverlapping error is ', hess_err)
    if hess_err < thres:
        print('No overlapping')
        print(J1, J2, hess_err)
        return [], J1, J2
    
    groups = [[i] for i in range(dim)]
    overlap = []
    overlap_err = float('inf')
    
    while overlap_err >= thres:
        # 1. Fix each variable, use binary tree structure to determine the group of remaining variables.
        structure = dict()
        for ovl in groups:
            vars = [ele[0] for ele in groups if ele != ovl]
            hess_err, J1, J2 = two_groups_greedy(grid, grad, hess, vars, reg)
            
            structure[ovl[0]] = (hess_err, J1, J2)
        print(structure)
        err = [structure[k][0] for k in structure.keys()]
        key = list(structure.keys())[np.argmin(err)]
        
        overlap += [groups[np.argmin(err)][0]]
        overlap_err = np.min(err)
        print('the overlapping error is: ', overlap_err)
        if overlap_err < thres:
            J1, J2 = structure[key][1], structure[key][2]
            break
            # return overlap, group_i, group_j
        # 2. If the error is greater than the stopping threshold, fix more variables.
        else:
            groups = [ele for ele in groups if ele[0] not in overlap]
    
    return overlap, J1, J2


from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import PartialDependenceDisplay      
from sklearn.utils.validation import check_is_fitted
    
class Estimator(BaseEstimator, RegressorMixin):
    def __init__(self, obj):
        self.obj = obj

    def fit(self, X, y):
        # Check that X and y have correct shape
        # Return the classifier
        return self

    def predict(self, X):
        check_is_fitted(self)
        # Check is fit had been called
        Y = self.obj(X)
        return Y
    def score(self, X, y, sample_weight=None):
        return 1.
    def __sklearn_is_fitted__(self):
        return True

def ICEPlot(obj, grid, cols):
    model = Estimator(obj)
    model.fit(grid, obj(grid))   
    
    PartialDependenceDisplay.from_estimator(model, grid, cols,                                            
        ice_lines_kw={"color": "tab:blue", "alpha": 0.2, "linewidth": 0.5, "label": "ICE"},
        pd_line_kw={"color": "tab:orange", "linestyle": "--", "label": "PD plot"},
        kind='both')  
    
    # model = Estimator(func.obj)
    # model.fit(grid_var, func.obj(grid_var)) 
    
    # X = pd.DataFrame(grid_var, columns=['$x_'+str(i+1)+'$' for i in range(5)])  
    
    # ax = PartialDependenceDisplay.from_estimator(model, X, ['$x_1$'],                                            
    #     ice_lines_kw={"color": "tab:blue", "alpha": 0.2, "linewidth": 0.5, "label": "ICE"},
    #     pd_line_kw={"color": "tab:orange", "linestyle": "--", "label": "PD plot"},
    #     kind='both')  
    ax.tick_params(axis='both', which='major', labelsize=12)
    
def overlapping_stepwise_tf(obj, grid, reg = 1, thres = 1e-6):
    '''
        Use autodiff in Tensorflow to compute grad and hessian.
    '''
    tf.keras.backend.set_floatx('float64')
    x = tf.Variable(grid)
    with tf.GradientTape() as tape:
        with tf.GradientTape() as tape2:
            y = obj(x)
        grad = tape2.gradient(y,x)
    hess = tape.jacobian(grad,x)
    hess = tf.reduce_sum(hess, axis=2)
    grad = grad.numpy()
    hess = hess.numpy()
    
    _, dim = np.shape(grid) # sample_dim = dim-1
    
    # No overlapping
    hess_err, J1, J2 = two_groups_greedy(grid, grad, hess, [i for i in range(dim)], reg)
    print('Non-onverlapping error is ', hess_err)
    if hess_err < thres:
        print('No overlapping')
        print(J1, J2)
        return [], J1, J2
    
    groups = [[i] for i in range(dim)]
    overlap = []
    overlap_err = float('inf')
    
    while overlap_err >= thres:
        # 1. Fix each variable, use binary tree structure to determine the group of remaining variables.
        structure = dict()
        for ovl in groups:
            vars = [ele[0] for ele in groups if ele != ovl]
            hess_err, J1, J2 = two_groups_greedy(grid, grad, hess, vars, reg)
            
            structure[ovl[0]] = (hess_err, J1, J2)
        print(structure)
        err = [structure[k][0] for k in structure.keys()]
        key = list(structure.keys())[np.argmin(err)]
        
        overlap += [groups[np.argmin(err)][0]]
        overlap_err = np.min(err)
        print('the overlapping error is: ', overlap_err)
        if overlap_err < thres:
            J1, J2 = structure[key][1], structure[key][2]
            break
            # return overlap, group_i, group_j
        # 2. If the error is greater than the stopping threshold, fix more variables.
        else:
            groups = [ele for ele in groups if ele[0] not in overlap]
    
    return overlap, J1, J2    
    
def overlapping(obj, grid, lb, ub, grid_length = 0.01, reg = 1, thres = 1e-6):
    '''
        Find the overlappig structure f(x) = g(h_1(x_0, x_J1), h_2(x_0, x_J2))
    '''
    
    grad = Grad(obj, grid, grid_length, lb, ub)
    hess = Hessian(obj, grid, grid_length, lb, ub)
    _, dim = np.shape(grid) # sample_dim = dim-1
    
    groups = [[i] for i in range(dim)]
    
    # 1. Find the two initial indices (group_index) minimize the hessian regression error, put them into J1.
    group_index, _ = TreeHessRegErr(obj, grid, grad, hess, groups, reg, [])
    
    # Thinking: encourage to expand J1 first.
    
    # 2. Use (group_index) to determine J2
    rm_indices = [[i] for i in range(dim) if i not in group_index]
    hess_err = dict()
    for group_j in rm_indices:
        hess_err[group_j[0]] = HessRegErrSingle(obj, grid, grad, hess, group_index, group_j)
    j2 = [j for j in rm_indices if hess_err[j[0]] < thres]
    
    # 3. Use J2 to determine J1.
    rm_indices = [[i] for i in range(dim) if [i] not in j2]
    hess_err = dict()
    group_i = [i[0] for i in j2]
    for group_j in rm_indices:
        hess_err[group_j[0]] = HessRegErrSingle(obj, grid, grad, hess, group_i, group_j)
    j1 = [j for j in rm_indices if hess_err[j[0]] < thres]
    
    # The remaining is J0 (overlapping variables)
    j0 = [[i] for i in range(dim) if [i] not in j1 + j2]
    
    # Balance J1 and J2 if possible
    if len(j1) < len(j2):
        j1, j2 = j2, j1
    
    err = 0.
    while len(j1) > len(j2):
        for l in j1:
            group_i = [i[0] for i in j2]+l
            rm_indices = [ele for ele in j1 if ele != l]
            hess_err = 0
            for group_j in rm_indices:
                hess_err += HessRegErrSingle(obj, grid, grad, hess, group_i, group_j)
            err = hess_err/len(rm_indices)
        if err < thres:
            j2 = [[i] for i in group_i]
            j1 = rm_indices
        else:
            break
        
    return j0, j1, j2
    
def overlapping_fixed(obj, grid, lb, ub, grid_length = 0.01, reg = 1, thres = 1e-6):
    '''
        Find the overlappig structure f(x) = g(h_1(x_0, x_J1), h_2(x_0, x_J2)) by gradually fixing more input variables
    ''' 
    def flatten(object):
        for item in object:
            if isinstance(item, (tuple, set)):
                yield from flatten(item)
            else:
                yield item
            
    grad = Grad(obj, grid, grid_length, lb, ub)
    hess = Hessian(obj, grid, grid_length, lb, ub)
    sample_num, dim = np.shape(grid) # sample_dim = dim-1
    
    groups = [[i] for i in range(dim)]
    overlap = []
    overlap_err = float('inf')
    
    while overlap_err >= thres:
        # 1. Fix each variable, use binary tree structure to determine the group of remaining variables.
        structure = dict()
        for ovl in groups:
            group_indices = variable_selection_tree(obj, grid, lb, ub, grid_length, reg, overlap = overlap + ovl)
            index_root, tree_data = index_tree(dim, group_indices)
            group_i = list(flatten(tree_data[-1][-2]))
            group_j = list(flatten(tree_data[-1][-1]))
            
            # Calculate the Group Hessian Regression Error
            hess_err = HessRegErrGroup(obj, grid, grad, hess, group_i, group_j)
            # Regularize based on the difference of number of variables between two groups
            hess_err *= reg * (abs(len(group_i) - len(group_j))+1)
            structure[ovl[0]] = (hess_err, group_i, group_j)

        err = [structure[k][0] for k in structure.keys()]
        print(structure)
        key = list(structure.keys())[np.argmin(err)]
        
        overlap += [groups[np.argmin(err)][0]]
        overlap_err = np.min(err)
        print('the overlapping error is: ', overlap_err)
        if overlap_err < thres:
            group_i, group_j = structure[key][1], structure[key][2]
            break
            # return overlap, group_i, group_j
        # 2. If the error is greater than the stopping threshold, fix more variables.
        else:
            groups = [ele for ele in groups if ele[0] not in overlap]
    
    return overlap, group_i, group_j
    
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
