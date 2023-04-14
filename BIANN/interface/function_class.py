# """
# Created on Mon Oct 22 15:29:15 2018

# @author: Shengtong Zhang
# """

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from copy import copy
import random
from pyDOE import *
from matplotlib import cm
import os
import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Lambda
from keras.models import Model
import keras
from keras import regularizers
from PIL import Image
from matplotlib.colors import LightSource
import matplotlib.gridspec as gridspec

def fmt(x):
    s = f"{x:.1e}"
    if s.endswith("0"):
        s = f"{x:.0e}"
    return rf"{s}" if plt.rcParams["text.usetex"] else f"{s}"


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

def convertTreeAux(node, newick):
    if node.isLeaf():
        newick.append(node.data)
    if node.left != None:
        newick.append("(")
        convertTreeAux(node.left, newick)
        newick.append(",")
    if node.right != None:
        convertTreeAux(node.right, newick)
        newick.append(")")
    
    # newick.append(")")
        
def Depth(node):
    if node is None:
        return 0
  
    else:
  
        # Compute the depth of each subtree
        lDepth = Depth(node.left)
        rDepth = Depth(node.right)
  
        # Use the larger one
        if (lDepth > rDepth):
            return lDepth+1
        else:
            return rDepth+1
        
def Leaves(root):
    if root is None:
        return []
    if root.left is None and root.right is None:
        return [root.data]
    leaves = []
    if root.left:
        leaves += Leaves(root.left)
    if root.right:
        leaves += Leaves(root.right)
    return leaves

def Connect(root, level):
    # connect all nodes at a specific level
    res = []
    if root is None:
        return []
    if level <= 0:
        raise ValueError('level should be positive')
    
    depth = Depth(root)
    if level > depth:
        return Connect(root, depth)
    if level == 1:
        return [root]
    if root.left:
        res += Connect(root.left, level-1)
    if root.right:
        res += Connect(root.right, level-1)
    return res

def CompleteTree(root):
    # Complete a tree s.t. all branch have the same length. Used in IANN_tree_model
    nodes = [(root, 1)]
    if root is None:
        return []
    depth = Depth(root)
    while nodes != []:
        x, l = nodes.pop(0)
        if l < depth:
            if x.left is None and x.right is None:
                x.left = x
            if x.left:
                nodes.append((x.left, l+1))
            if x.right:
                nodes.append((x.right, l+1))
            
    return root

def printTree(node, level=0):
    if node != None:
        printTree(node.left, level + 1)
        print(' ' * 4 * level + '------> ' + str(node.data))
        printTree(node.right, level + 1)
    

def ConnectOriginal(root, level):
    # connect all nodes at a specific level
    res = []
    nodes = []
    nodes.append((root,1))
    if root is None:
        return []
    if level > Depth(root):
        print('The level is larger than the depth of the tree. Check the tree structure!')
        level = Depth(root)
    while nodes != []:
        x, l = nodes.pop(0)
        if l == level:
            res.append(x)
            continue
        if x.left:
            nodes.append((x.left, l+1))
        if x.right:
            nodes.append((x.right, l+1))
    return res

        
def traverse_names(root, fig_name):
    depth = Depth(root)
    nodes_by_level = dict()
    for l in range(1, depth+1):
        nodes_by_level[l] = ConnectOriginal(root, l)
    
    node_names = []
    nodes = [(root, 1)]
    while nodes != []:
        x, l = nodes.pop(0)
        if (x.left and x.right is None) or (x.right and x.left is None):
            raise ValueError()('This is not a full binary tree! Check the tree structure.')
        if l < depth and (x.left or x.right):
            pos = nodes_by_level[l].index(x)
            name = fig_name+'_' + str(l) + '_' + str(pos)
            node_names.append(name)
            
            nodes.append((x.left, l+1))
            nodes.append((x.right, l+1))
    return node_names

def treeIANNPlot(root, tree_data, fig_name):
    # organize the IANN plots in a tree structure
    # Packages from http://etetoolkit.org/docs/latest/index.html
    from ete3 import Tree, NodeStyle, TreeStyle, TextFace, faces
    
    # Creates my own layout function. I will use all previously created
    # faces and will set different node styles depending on the type of
    # node.
    def mylayout(node):
        # If node is a leaf, add the nodes name and a its scientific
        # name

        # Add the corresponding face to the node
        nodeFace = faces.ImgFace(os.path.join(img_path, node.name+'.png'))
        faces.add_face_to_node(nodeFace, node, column=0)

        # Modifies this node's style
        node.img_style["size"] = 16
        node.img_style["shape"] = "sphere"
        node.img_style["fgcolor"] = "#AA0000" 
            
    t = Tree(str(tree_data[0]) + ";")
    for node in t.traverse():
        if node.is_leaf():
            node.detach()
    # Rotate the tree
    # ts.rotation = 90

    img_path = os.path.join(os.getcwd(), 'image')

    nameFace = faces.AttrFace("name", fsize=20, fgcolor="#009000")

    node_names = traverse_names(root, fig_name)
    cnt = 0
    for node in t.traverse():
        node.name = node_names[cnt]
        cnt += 1
        
    # Basic tree style
    ts = TreeStyle()
    ts.show_leaf_name = False
    ts.show_scale = False
    ts.layout_fn = mylayout
    t.render(fig_name + "IANN_tree.png", w=6000, tree_style = ts)
    t.show(tree_style=ts)       


def treeIANNPlotOverlap(tree_data, fig_name, node_names):
    # organize the IANN plots in a tree structure
    # Packages from http://etetoolkit.org/docs/latest/index.html
    from ete3 import Tree, NodeStyle, TreeStyle, TextFace, faces
    
    # Creates my own layout function. I will use all previously created
    # faces and will set different node styles depending on the type of
    # node.
    def mylayout(node):
        # If node is a leaf, add the nodes name and a its scientific
        # name

        # Add the corresponding face to the node
        nodeFace = faces.ImgFace(os.path.join(img_path, node.name+'.png'))
        faces.add_face_to_node(nodeFace, node, column=0)

        # Modifies this node's style
        node.img_style["size"] = 16
        node.img_style["shape"] = "sphere"
        node.img_style["fgcolor"] = "#AA0000" 
            
    t = Tree(str(tree_data[0]) + ";")
    for node in t.traverse():
        if node.is_leaf():
            node.detach()
    # Rotate the tree
    # ts.rotation = 90

    img_path = os.path.join(os.getcwd(), 'image')

    nameFace = faces.AttrFace("name", fsize=20, fgcolor="#009000")

    # node_names = traverse_names(root, fig_name)
    cnt = 0
    for node in t.traverse():
        node.name = node_names[cnt]
        cnt += 1
        
    # Basic tree style
    ts = TreeStyle()
    ts.show_leaf_name = False
    ts.show_scale = False
    ts.layout_fn = mylayout
    t.render(fig_name + "IANN_tree.png", w=6000, tree_style = ts)
    t.show(tree_style=ts)  


class function():
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
    
    def generate_grid(self, dim, samples):
        grid = lhs(dim, samples=samples, criterion = 'm')
        return grid
         
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
        exploration = Model(inputs=input_path, outputs=decoded)

        return exploration
    
    def IANN_model(self, disjoint_group, plot_level, regularization_rate = 0.,\
                        activation = 'sigmoid', base_node = 8):
        # plot_level: How many levels should IANN plot

        def slice(x, st, ed):
            return x[:,st:ed]
        #     Construct auto-encoder with multiple layers
        n_input = self.d
        num_group = len(disjoint_group)
        input_path = Input(shape=(n_input,))
        
        if plot_level == 0:
            bottle_neck = input_path
        else:
            if num_group == 1:
                linear_node = Dense(1, activation='linear', use_bias = False, kernel_initializer = 'random_normal',
                                kernel_regularizer=regularizers.l2(0.), name = 'linear_0')(input_path)
                hidden=Dense(base_node, activation=activation, kernel_initializer = 'random_normal',
                                    kernel_regularizer=regularizers.l2(regularization_rate), name = 'dense_0')(linear_node)
                output = Dense(1, activation='linear', kernel_initializer = 'random_normal',
                            kernel_regularizer=regularizers.l2(regularization_rate))(hidden)
                exploration = Model(inputs=input_path, outputs=output)
                return exploration

            input_node = {}
            tot_len = 0
            for i in range(plot_level):
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
            input_rs = Lambda(slice,output_shape=(length,),arguments={'st':tot_len, 'ed':n_input})(input_path)
            input0 = Reshape((n_input - tot_len,))(input_rs)
            bottle_neck = input_rs
            for i in range(plot_level):
                encoded = Dense(base_node*(i+1), activation=activation, kernel_initializer = 'random_normal',
                                kernel_regularizer=regularizers.l2(regularization_rate), name = 'dense_hidden1_level_'+str(plot_level - i))(bottle_neck)
                # encoded = Dense(base_node*(i+1), activation=activation, kernel_initializer = 'random_normal',
                #                 kernel_regularizer=regularizers.l2(regularization_rate), name = 'dense_hidden2_level_'+str(plot_level - i))(encoded)
                encoded = Dense(1, activation='linear', kernel_initializer = 'random_normal',
                                kernel_regularizer=regularizers.l2(regularization_rate), name = 'dense_out_level_'+str(plot_level - i))(encoded)
                bottle_neck = keras.layers.concatenate([input_node['group_'+str(plot_level-i-1)], encoded],\
                    axis = 1, name = 'con_' + str(i+1))

            
        decoded = Dense(base_node*(plot_level+1), activation=activation, kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate), name = 'dense_hidden_level_0')(bottle_neck) 
        decoded = Dense(1, activation='linear', kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate), name = 'dense_out_level_0')(decoded)
        exploration = Model(inputs=input_path, outputs=decoded)

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
            exploration = Model(inputs=input_path, outputs=output)
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
            stage = num_group-i-1
            encoded = Dense(base_node*(i+1), activation=activation, kernel_initializer = 'random_normal',
                            kernel_regularizer=regularizers.l2(regularization_rate), name = 'dense_encode_'+str(stage))(bottle_neck)
            encoded = Dense(1, activation='linear', kernel_initializer = 'random_normal',
                            kernel_regularizer=regularizers.l2(regularization_rate), name = 'dense_decode_'+str(stage))(encoded)
            bottle_neck = keras.layers.concatenate([input_node['group_'+str(num_group-i-3)], encoded],\
                 axis = 1, name = 'con_' + str(i+1))

            
        decoded = Dense(base_node*(num_group-1), activation=activation, kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate), name = 'dense_encode_1')(bottle_neck)
        decoded = Dense(1, activation='linear', kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate), name = 'dense_decode_1')(decoded)
        exploration = Model(inputs=input_path, outputs=decoded)

        return exploration

    def IANN_tree_model(self, index_root, regularization_rate = 0.,\
                        activation = 'sigmoid', base_node = 8):

        def slice(x, st, ed):
            return x[:,st:ed]
        #     Construct auto-encoder with multiple layers
        n_input = self.d
        input_path = Input(shape=(n_input,))
        
        group_index = Leaves(index_root)
        depth = Depth(index_root)
        inputs = []
        for i in range(n_input):
            input0 = Lambda(slice,output_shape=(1,),arguments={'st':i, 'ed':i+1})(input_path)
            #input_rs = Lambda(slice,output_shape=(n_input-i-1,),arguments={'st':0, 'ed':n_input-i-1})(input_path)
            input0 = Reshape((1,))(input0)
            #input_rs = Reshape((n_input-i-1,))(input_rs)
            inputs.append(input0)
        # inputs = inputs[group_index]
        
        hidden = inputs
        for level in range(depth-1,0,-1):
            nodes_by_level = Connect(index_root, level)
            # nodes_by_level_ori = ConnectOriginal(index_root, level)
            idx = 0
            hidden_tmp = []
            for i, ele in enumerate(nodes_by_level):
                # if len(ele.data) > 2:
                #     raise('Wrong Tree Structure')
                if isinstance(ele.data, int):
                    hidden_tmp.append(hidden[idx])
                    idx += 1
                elif len(ele.data) == 2:
                    node_input = keras.layers.concatenate([hidden[idx], hidden[idx+1]], axis = 1)
                    encoded = Dense(base_node*(level+1), activation=activation, kernel_initializer = 'random_normal',
                                    kernel_regularizer=regularizers.l2(regularization_rate), name = 'h_'+str(level)+'_'+str(i))(node_input)
                    # encoded = Dense(base_node*(level+1), activation=activation, kernel_initializer = 'random_normal',
                    #                 kernel_regularizer=regularizers.l2(regularization_rate), name = 'h2_'+str(level)+'_'+str(i))(encoded)
                    encoded = Dense(1, activation='linear', kernel_initializer = 'random_normal',
                                    kernel_regularizer=regularizers.l2(regularization_rate), name = 'o_'+str(level)+'_'+str(i))(encoded)
                    
                    hidden_tmp.append(encoded)
                    idx += 2
            hidden = hidden_tmp
            # print(hidden)
        
        output = hidden[0]
        tree = Model(inputs=input_path, outputs=output)

        return tree
    
    def IANN_overlapping(self, n_input, J0, J1, J2, regularization_rate = 0.,\
                        activation = 'sigmoid', base_node = 8):
        def slice(x, st, ed):
            return x[:,st:ed]
        #     Construct auto-encoder with multiple layers
        # n_input = self.d
        input_path = Input(shape=(n_input,))
        
        inputs = []
        for i in range(n_input):
            input0 = Lambda(slice,output_shape=(1,),arguments={'st':i, 'ed':i+1})(input_path)
            #input_rs = Lambda(slice,output_shape=(n_input-i-1,),arguments={'st':0, 'ed':n_input-i-1})(input_path)
            input0 = Reshape((1,))(input0)
            #input_rs = Reshape((n_input-i-1,))(input_rs)
            inputs.append(input0)
        # inputs = inputs[group_index]
        
        if len(J0+J1)>1:
            group_1 = keras.layers.concatenate([inputs[i] for i in range(n_input) if i in J0+J1], axis = 1)
            encoded_1 = Dense(base_node, activation=activation, kernel_initializer = 'random_normal',
                            kernel_regularizer=regularizers.l2(regularization_rate), name = 'h1_1')(group_1)
            encoded_1 = Dense(base_node, activation=activation, kernel_initializer = 'random_normal',
                            kernel_regularizer=regularizers.l2(regularization_rate), name = 'h2_1')(encoded_1)
            # encoded_1 = Dense(base_node, activation=activation, kernel_initializer = 'random_normal',
            #                 kernel_regularizer=regularizers.l2(regularization_rate), name = 'h3_1')(encoded_1)
            h1 = Dense(1, activation='linear', kernel_initializer = 'random_normal',
                            kernel_regularizer=regularizers.l2(regularization_rate), name = 'o_1')(encoded_1)
        else:
            J = J0+J1
            h1 = inputs[J[0]]

        if len(J0+J2)>1:
            group_2 = keras.layers.concatenate([inputs[i] for i in range(n_input) if i in J0+J2], axis = 1)
            encoded_2 = Dense(base_node, activation=activation, kernel_initializer = 'random_normal',
                            kernel_regularizer=regularizers.l2(regularization_rate), name = 'h1_2')(group_2)
            encoded_2 = Dense(base_node, activation=activation, kernel_initializer = 'random_normal',
                            kernel_regularizer=regularizers.l2(regularization_rate), name = 'h2_2')(encoded_2)
            # encoded_2 = Dense(base_node, activation=activation, kernel_initializer = 'random_normal',
            #                 kernel_regularizer=regularizers.l2(regularization_rate), name = 'h3_2')(encoded_2)
            h2 = Dense(1, activation='linear', kernel_initializer = 'random_normal',
                            kernel_regularizer=regularizers.l2(regularization_rate), name = 'o_2')(encoded_2)
        else:
            J = J0+J2
            h2 = inputs[J[0]]
        
        bottleneck = keras.layers.concatenate([h1, h2], axis = 1)
        encoded = Dense(base_node, activation=activation, kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate), name = 'h1_3')(bottleneck)
        encoded = Dense(base_node, activation=activation, kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate), name = 'h2_3')(encoded)
        # encoded = Dense(base_node, activation=activation, kernel_initializer = 'random_normal',
        #                 kernel_regularizer=regularizers.l2(regularization_rate), name = 'h3_3')(encoded)
        output = Dense(1, activation='linear', kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate), name = 'o_3')(encoded)
            
        model = Model(inputs=input_path, outputs=output)

        return model
    
    def overlapping_models(self, model, J0, J1, J2):
        # return two latent functions h1 and h2, and decoder model_dc for further analysis.
        input_1 = Input(shape=(len(J0) + len(J1),))
        input_2 = Input(shape=(len(J0) + len(J2),))
        input_dc = Input(shape = (2,))
        
        if len(J0+J1)>1:
            hidden_1 = model.get_layer('h1_1')(input_1)
            hidden_1 = model.get_layer('h2_1')(hidden_1)
            # hidden_1 = model.get_layer('h3_1')(hidden_1)
            output_1 = model.get_layer('o_1')(hidden_1)
            model_1 = Model(inputs = input_1, outputs = output_1)  
        else:
            model_1 = Model(inputs = input_1, outputs = input_1)  
            
        if len(J0+J2)>1:
            hidden_2 = model.get_layer('h1_2')(input_2)
            hidden_2 = model.get_layer('h2_2')(hidden_2)
            # hidden_2 = model.get_layer('h3_2')(hidden_2)
            output_2 = model.get_layer('o_2')(hidden_2)
            model_2 = Model(inputs = input_2, outputs = output_2)
        else:
            model_2 = Model(inputs = input_2, outputs = input_2)
        
        hidden_dc = model.get_layer('h1_3')(input_dc)
        hidden_dc = model.get_layer('h2_3')(hidden_dc)
        # hidden_dc = model.get_layer('h3_3')(hidden_dc)
        output_dc = model.get_layer('o_3')(hidden_dc)
        
        model_dc = Model(inputs = input_dc, outputs = output_dc)
        
        return model_1, model_2, model_dc

    def exploration_train(self, exploration, x_train, y_train, x_test, y_test, lr = 0.01, BATCH_SIZE = 5, EPOCHS = 50):
        adam = keras.optimizers.Adam(lr=lr) #lr := learning rate
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
        checkpoint_filepath = os.getcwd()+'\\checkpoint\\'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            save_best_only=True)
        history = exploration.fit(x_train,y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        verbose = 2,
                        validation_data=(x_test,y_test),
                        callbacks=[model_checkpoint_callback])
        if 1:
            plt.plot(history.history['loss'][5:])
            plt.plot(history.history['val_loss'][5:])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.show()
        exploration.load_weights(checkpoint_filepath)
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
    
    def backward_encode_heatmap(self, exploration, v, h_mid, stage, disjoint_group, base_node = 8, activation = 'relu', regularization_rate = 0.):
        def slice(x, st, ed):
            return x[:,st:ed]
        #     Construct auto-encoder with multiple layers
        num_group = len(disjoint_group)
        input_v = Input(shape=(1,))
        input_h = Input(shape=(1,))
        bottle_neck = keras.layers.concatenate([input_v, input_h], axis = 1, name = 'con_'+str(num_group-stage-1))
        i = num_group - stage
            
        decoded = Dense(base_node*(i), activation=activation, kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate), name = 'dense_encode_'+str(stage))(bottle_neck)
        decoded = Dense(1, activation='linear', kernel_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(regularization_rate), name = 'dense_decode_'+str(stage))(decoded)
        v_val = Model(inputs = [input_v, input_h], outputs = decoded)
        v_val.get_layer('dense_encode_'+str(stage)).set_weights(exploration.get_layer('dense_encode_'+str(stage)).get_weights())
        v_val.get_layer('dense_decode_'+str(stage)).set_weights(exploration.get_layer('dense_decode_'+str(stage)).get_weights())
        
        h_out = v_val.predict([v.ravel(), h_mid.ravel()])
        
        return h_out 
    
    def stage_plot(self, stage, x_plot, y_plot, index, var_index, name, exploration, nonadd_id, heatmap = None):
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
        if heatmap:
            ind = np.array(np.argsort(h_mid, axis = 0))[:,0].flatten()
            h_mid = h_mid[ind,]
            ax1 = fig.add_subplot(111)
            ax1.set_xlabel('$x$'+str(nonadd_id[var_index[stage-1]]+1))
            ax1.set_ylabel('h'+str(stage))
            ax1.set_title('Visualization for level %d' % stage)
            if stage == 1:
                z_mds = np.reshape(y_plot, h_mid.shape)
                z_mds = z_mds[ind,]
                CS = ax1.contour(x_grid_2d, h_mid, z_mds, 5, colors = 'k')
                ax1.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)
                CP = ax1.contourf(x_grid_2d, h_mid, z_mds, 10, cmap = cm.coolwarm, extend='both')
                plt.colorbar(CP)
            else:
                h_mid = np.reshape(h_mid, x_grid_2d.shape)
                h_out = np.reshape(h_out, x_grid_2d.shape)
                h_out = h_out[ind,]
                CS = ax1.contour(x_grid_2d, h_mid, h_out, 5, colors = 'k')
                ax1.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)
                CP = ax1.contourf(x_grid_2d, h_mid, h_out, cmap = cm.coolwarm)
                plt.colorbar(CP)
        else:
            ax1 = fig.add_subplot(111, projection = '3d')
            ax1.set_xlabel('$x$'+str(nonadd_id[var_index[stage-1]]+1))
            ax1.set_ylabel('h'+str(stage))
            ax1.set_title('Visualization for level %d' % stage)
            azimuth = 135+180
            altitude = 45
            ax1.view_init(altitude, azimuth)
            cmap = matplotlib.colors.ListedColormap("red") 
            if stage == 1:
                ax1.set_zlabel('f')
            else:
                ax1.set_zlabel('h'+str(stage-1))
            if stage == 1:
                z_mds = np.reshape(y_plot, h_mid.shape)
                ax1.plot_trisurf(x_grid_2d.flatten(), h_mid.flatten(), z_mds.flatten(), cmap = cm.coolwarm)
            else:
                h_mid = np.array(h_mid)
                ax1.plot_trisurf(x_grid_2d.flatten(), h_mid.flatten(), h_out.flatten(), cmap = cm.coolwarm, antialiased = True)
        if stage == self.d - 1:
            ax1.set_ylabel('$x$'+str(nonadd_id[var_index[stage]]+1))

        # Store the image
        if 1:
            file_path = os.getcwd() + '\\image\\'
            save_name = file_path + name + str(stage)+'.png'
            plt.savefig(save_name, dpi=100)
            plt.show()
            plt.close()         

    def backward_stage_plot(self, stage, x_plot, y_plot, disjoint_group, name, exploration,\
            num_x, num_y, nonadd_id, base_node = 8, activation = 'relu', regularization_rate = 0.,\
            xlabel = [], pad = 20, heatmap = None,):
        '''
            Make the 3D visualization plot for specific stage using the function:
            exploration_encode functionhzn.
        '''
        def grid(x, y, z, resX=100, resY=100, base_node = base_node, activation = activation, regularization_rate = regularization_rate):
            "Convert 3 column data to matplotlib grid"
            xi = np.linspace(np.min(x), np.max(x), resX)
            yi = np.linspace(np.min(y), np.max(y), resY)
            X, Y = np.meshgrid(xi, yi)
            Z = self.backward_encode_heatmap(exploration, X, Y, stage, disjoint_group, base_node = base_node, \
                activation = activation, regularization_rate = regularization_rate)
            Z = np.reshape(Z, X.shape)
            return X, Y, Z
        
        var_index = sum(disjoint_group, [])
        num_group = len(disjoint_group)

        v, h_mid, h_out, _ = self.backward_encode(exploration, x_plot, stage, disjoint_group)
        v = np.reshape(v, (-1, num_x))
        h_mid = np.reshape(h_mid, (-1, num_x))
        h_out = np.reshape(h_out, (-1, num_x))         
        if len(disjoint_group[stage-1]) == 1:
            index = disjoint_group[stage-1][0]
            v = v 
        
        if stage == num_group-1:
            if len(disjoint_group[stage]) == 1:
                index = disjoint_group[stage][0]
                h_mid = h_mid

        ######################################################
        #   plot Neural Network at given stage
        ######################################################
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        if heatmap:
            ax1 = fig.add_subplot(111)
        else:
            ax1 = fig.add_subplot(111, projection = '3d')
        if len(xlabel) > 0:
            ax1.set_xlabel(xlabel[stage-1], labelpad = pad)
            if stage == 1:
                ax1.set_zlabel('f', labelpad = pad)
            else:
                ax1.set_zlabel('h'+str(stage-1), labelpad = pad)
        else:
            if len(disjoint_group[stage-1]) == 1:
                index = disjoint_group[stage-1][0]
                index = nonadd_id[index]
                ax1.set_xlabel('$x$'+str(index+1), labelpad = pad)
            else:
                ax1.set_xlabel('$v$'+str(stage), labelpad = pad)
        ax1.set_ylabel('h'+str(stage), labelpad = pad)
        if stage == num_group - 1:
            if len(disjoint_group[stage]) == 1:
                index = disjoint_group[stage][0]
                index = nonadd_id[index]
                ax1.set_ylabel('$x$'+str(index+1), labelpad = pad)
            else:
                ax1.set_ylabel('$v$'+str(stage+1), labelpad = pad)
            if len(xlabel) > 0:
                ax1.set_ylabel(xlabel[stage], labelpad = pad)

        ax1.set_title('Visualization for level %d' % stage)

        if heatmap:    
            if stage == 1:
                z_mds = np.reshape(y_plot, h_mid.shape)
                X, Y, Z = grid(v, h_mid, z_mds)
                CS = ax1.contour(X, Y, Z, 5, colors = 'k')
                ax1.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)
                CP = ax1.contourf(X, Y, Z, cmap = cm.coolwarm)
                plt.colorbar(CP)
            else:
                h_mid = np.array(h_mid)
                h_out = np.reshape(h_out, h_mid.shape)
                X, Y, Z = grid(v, h_mid, h_out)
                CS = ax1.contour(X, Y, Z, 5, colors = 'k')
                ax1.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)
                CP = ax1.contourf(X, Y, Z, cmap = cm.coolwarm)
                plt.colorbar(CP) 
        else:
            rgb = np.ones((h_out.shape[0], h_out.shape[1], 3))
            light = LightSource(90, 0)
            illuminated_surface = light.shade_rgb(rgb, h_out)
            green = np.array([0,1.0,0])
            green_surface = light.shade_rgb(rgb * green, h_out)

            # Set view parameters for all subplots.
            azimuth = 135+180
            altitude = 45
            ax1.view_init(altitude, azimuth)
            # cmap = matplotlib.colors.ListedColormap("red") 
            if stage == 1:
                z_mds = np.reshape(y_plot, h_mid.shape)
                ax1.plot_trisurf(v.flatten(), h_mid.flatten(), z_mds.flatten(), \
                    cmap = cm.coolwarm, linewidth=0, antialiased=True)
            else:
                h_mid = np.array(h_mid)
                ax1.plot_trisurf(v.flatten(), h_mid.flatten(), h_out.flatten(), \
                    cmap = cm.coolwarm, linewidth=0, antialiased=True)
        if 1:
            file_path = os.getcwd() + '\\image\\'
            save_name = file_path + name + str(stage)+'.png'
            print(save_name)
            plt.savefig(save_name, dpi=100)
            plt.show()
            plt.close()
        return v, h_mid, h_out
    
    def overlap_plot(self, model_1, model_2, model_dc, x_plot, J0, J1, J2, num_x, labels, name, heatmap = False):
        # out = model.predict(x_plot[:, J0+J1+J2])
        h1 = model_1.predict(x_plot[:, J0 + J1])
        h2 = model_2.predict(x_plot[:, J0 + J2])
        
        h1_min, h2_min = np.percentile(h1, 10), np.percentile(h2, 10)
        h1_max, h2_max = np.percentile(h1, 90), np.percentile(h2, 90)
        
        dx = np.linspace(h1_min, h1_max, num_x)
        dy = np.linspace(h2_min, h2_max, num_x)
        xx, yy = np.meshgrid(dx, dy)        
        zz = model_dc.predict(np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1)))))
                
        xlabel, ylabel, zlabel = labels[0], labels[1], labels[2]
    
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        if heatmap:
            ind = np.array(np.argsort(y, axis = 0))[:,0].flatten()
            y = y[ind,]
            ax1 = fig.add_subplot(111)
            ax1.set_xlabel(xlabel)
            ax1.set_ylabel(ylabel)
            # ax1.set_title('Visualization for level %d' % stage)
            CS = ax1.contour(xx,yy,zz, 5, colors = 'k')
            ax1.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=12)
            CP = ax1.contourf(xx,yy,zz, 10, cmap = cm.coolwarm, extend='both')
            plt.colorbar(CP)
        else:
            ax1 = fig.add_subplot(111, projection = '3d')
            ax1.tick_params(axis='both', which='major', labelsize=12)
            ax1.set_xlabel(xlabel, fontsize = 10)
            ax1.set_ylabel(ylabel, fontsize = 10)
            ax1.set_zlabel(zlabel, fontsize = 10)
            # ax1.set_title('Visualization for level %d' % stage)
            azimuth = 135+180
            altitude = 45
            ax1.view_init(altitude, azimuth)
            cmap = matplotlib.colors.ListedColormap("red") 
            ax1.plot_trisurf(xx.flatten(), yy.flatten(), zz.flatten(), cmap = cm.coolwarm)   
            # ax1.scatter(x.flatten(), y.flatten(), z.flatten())    
              
        if 1:
            file_path = os.getcwd() + '\\image\\'
            save_name = file_path + name
            plt.savefig(save_name, dpi=100)
            plt.show()
            plt.close()

    def overlap_plot_leaf(self, model, J1, J2, num_x, labels, name, heatmap = False):
        # out = model.predict(x_plot[:, J0+J1+J2])
        
        # x1_min, x2_min = np.percentile(x_plot[:,J1], 10), np.percentile(x_plot[:,J2], 10)
        # x1_max, x2_max = np.percentile(x_plot[:,J1], 90), np.percentile(x_plot[:,J2], 90)
        
        dx = np.linspace(0, 1, num_x)
        dy = np.linspace(0, 1, num_x)
        xx, yy = np.meshgrid(dx, dy)        
        zz = model.predict(np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1)))))
        xx = xx * (self.ub[J1] - self.lb[J1]) + self.lb[J1]
        yy = yy * (self.ub[J2] - self.lb[J2]) + self.lb[J2]
                
        xlabel, ylabel, zlabel = labels[0], labels[1], labels[2]
    
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        if heatmap:
            ind = np.array(np.argsort(y, axis = 0))[:,0].flatten()
            y = y[ind,]
            ax1 = fig.add_subplot(111)
            ax1.set_xlabel(xlabel)
            ax1.set_ylabel(ylabel)
            # ax1.set_title('Visualization for level %d' % stage)
            CS = ax1.contour(xx,yy,zz, 5, colors = 'k')
            ax1.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=12)
            CP = ax1.contourf(xx,yy,zz, 10, cmap = cm.coolwarm, extend='both')
            plt.colorbar(CP)
        else:
            ax1 = fig.add_subplot(111, projection = '3d')
            ax1.tick_params(axis='both', which='major', labelsize=15)
            ax1.set_xlabel(xlabel, fontsize = 20, labelpad = 15)
            ax1.set_ylabel(ylabel, fontsize = 20, labelpad = 15)
            ax1.set_zlabel(zlabel, fontsize = 20, labelpad = 15, rotation = 0)
            # ax1.set_title('Visualization for level %d' % stage)
            azimuth = 135+180
            altitude = 45
            ax1.view_init(altitude, azimuth)
            cmap = matplotlib.colors.ListedColormap("red") 
            ax1.plot_trisurf(xx.flatten(), yy.flatten(), zz.flatten(), cmap = cm.coolwarm)   
            # ax1.scatter(x.flatten(), y.flatten(), z.flatten())    
              
        if 1:
            file_path = os.getcwd() + '\\image\\'
            save_name = file_path + name
            plt.savefig(save_name, dpi=100)
            plt.show()
            plt.close()
            
    def overlap_plot(self, model_1, model_2, model_dc, x_plot, J0, J1, J2, num_x, labels, name, heatmap = False):
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        
        h1 = model_1.predict(x_plot[:, J0 + J1])
        h2 = model_2.predict(x_plot[:, J0 + J2])
              
        h1_min, h2_min = np.percentile(h1, 5), np.percentile(h2, 5)
        h1_max, h2_max = np.percentile(h1, 95), np.percentile(h2, 95)
        # h1_min, h2_min = np.min(h1), np.min(h2)
        # h1_max, h2_max = np.max(h1), np.max(h2)
        
        dx = np.linspace(h1_min, h1_max, num_x)
        dy = np.linspace(h2_min, h2_max, num_x)
        xx, yy = np.meshgrid(dx, dy)        
        zz = model_dc.predict(np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1)))))
                
        xlabel, ylabel, zlabel = labels[0], labels[1], labels[2]
    
        if heatmap:
            ind = np.array(np.argsort(y, axis = 0))[:,0].flatten()
            y = y[ind,]
            ax1 = fig.add_subplot(111)
            ax1.set_xlabel(xlabel)
            ax1.set_ylabel(ylabel)
            # ax1.set_title('Visualization for level %d' % stage)
            CS = ax1.contour(xx,yy,zz, 5, colors = 'k')
            ax1.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=15)
            CP = ax1.contourf(xx,yy,zz, 10, cmap = cm.coolwarm, extend='both')
            plt.colorbar(CP)
        else:
            ax1 = fig.add_subplot(111, projection = '3d')
            ax1.tick_params(axis='both', which='major', labelsize=15)
            ax1.set_xlabel(xlabel, fontsize = 20, labelpad = 15)
            ax1.set_ylabel(ylabel, fontsize = 20, labelpad = 15)
            ax1.set_zlabel(zlabel, fontsize = 20, labelpad = 15, rotation = 0)
            # ax1.set_title('Visualization for level %d' % stage)
            azimuth = 135+180
            altitude = 45
            ax1.view_init(altitude, azimuth)
            cmap = matplotlib.colors.ListedColormap("red") 
            ax1.plot_trisurf(xx.flatten(), yy.flatten(), zz.flatten(), cmap = cm.coolwarm)   
            # ax1.scatter(x.flatten(), y.flatten(), z.flatten())  
              
        if 1:
            file_path = os.getcwd() + '\\image\\'
            save_name = file_path + name
            plt.savefig(save_name, dpi=100)
            plt.show()
            plt.close()    

        return xx, yy, zz

    def overlap_plot_density(self, model_1, model_2, model_dc, x_plot, J0, J1, J2, num_x, labels, name, heatmap = False):
        # out = model.predict(x_plot[:, J0+J1+J2])
        from scipy.stats import kde
        fig = plt.figure(figsize=(7, 10))
        fig.patch.set_facecolor('white')
        
        spec = gridspec.GridSpec(ncols=1, nrows=2,
                         height_ratios=[1, 1], hspace=0.3)
                
        h1 = model_1.predict(x_plot[:, J0 + J1])
        h2 = model_2.predict(x_plot[:, J0 + J2])
              
        h1_min, h2_min = np.percentile(h1, 5), np.percentile(h2, 5)
        h1_max, h2_max = np.percentile(h1, 95), np.percentile(h2, 95)
        # h1_min, h2_min = np.min(h1), np.min(h2)
        # h1_max, h2_max = np.max(h1), np.max(h2)
        
        dx = np.linspace(h1_min, h1_max, num_x)
        dy = np.linspace(h2_min, h2_max, num_x)
        xx, yy = np.meshgrid(dx, dy)        
        zz = model_dc.predict(np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1)))))
                
        xlabel, ylabel, zlabel = labels[0], labels[1], labels[2]
    
        if heatmap:
            ind = np.array(np.argsort(y, axis = 0))[:,0].flatten()
            y = y[ind,]
            ax1 = fig.add_subplot(211)
            ax1.set_xlabel(xlabel)
            ax1.set_ylabel(ylabel)
            # ax1.set_title('Visualization for level %d' % stage)
            CS = ax1.contour(xx,yy,zz, 5, colors = 'k')
            ax1.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=15)
            CP = ax1.contourf(xx,yy,zz, 10, cmap = cm.coolwarm, extend='both')
            plt.colorbar(CP)
        else:
            ax1 = fig.add_subplot(spec[0], projection = '3d')
            ax1.tick_params(axis='both', which='major', labelsize=15)
            ax1.set_xlabel(xlabel, fontsize = 20, labelpad = 15)
            ax1.set_ylabel(ylabel, fontsize = 20, labelpad = 15)
            ax1.set_zlabel(zlabel, fontsize = 20, labelpad = 15, rotation = 0)
            # ax1.set_title('Visualization for level %d' % stage)
            azimuth = 135+180
            altitude = 45
            ax1.view_init(altitude, azimuth)
            cmap = matplotlib.colors.ListedColormap("red") 
            ax1.plot_trisurf(xx.flatten(), yy.flatten(), zz.flatten(), cmap = cm.coolwarm)   
            # ax1.scatter(x.flatten(), y.flatten(), z.flatten())  
              
        # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
        nbins=50
        # x = np.clip(h1.ravel(), h1_min, h1_max)
        # y = np.clip(h2.ravel(), h2_min, h2_max)
        k = kde.gaussian_kde([h1.ravel(),h2.ravel()])
        xi, yi = np.mgrid[h1_min:h1_max:nbins*1j, h2_min:h2_max:nbins*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        # 2D density plot of the training samples
        ax2 = fig.add_subplot(spec[1])
        ax2.tick_params(axis='both', which='major', labelsize=15)
        f2 = ax2.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
        
        CS = ax2.contour(xi, yi, zi.reshape(xi.shape), 5, colors = 'k')
        ax2.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=20)
        CP = ax2.contourf(xi, yi, zi.reshape(xi.shape), 10, cmap = cm.coolwarm, extend='both')
        plt.colorbar(CP)
        
        ax2.set_title('Training sample density plot', fontsize = 15)
        ax2.set_xlabel(xlabel, fontsize = 25)
        ax2.set_ylabel(ylabel, fontsize = 25)
        ax2.set_aspect(1./ax2.get_data_ratio())
        # fig.subplots_adjust(right = 0.9)
        # cbar = plt.colorbar(f2, ax = ax2, pad = 0.05)
        # cbar.ax.tick_params(labelsize=20)
        
              
        if 1:
            file_path = os.getcwd() + '\\image\\'
            save_name = file_path + name
            plt.savefig(save_name, dpi=100)
            plt.show()
            plt.close()    

        return xx, yy, zz
    
    def tree_plot(self, index_root, x_plot, y_plot, name, model, labels, num_x = 20, heatmap = None):
        '''
            Make the 3D visualization plot for tree IANN structure at all stages
        '''
        group_index = Leaves(index_root)
        depth = Depth(index_root)
        
        def slice(x, st, ed):
            return x[:,st:ed]
        
        def tree_encode(model, x_plot):
            tree_outputs = {}
            
            n_input = x_plot.shape[-1]
            input_path = Input(n_input)
            inputs = []
            for i in range(n_input):
                input0 = Lambda(slice,output_shape=(1,),arguments={'st':i, 'ed':i+1})(input_path)
                input0 = Reshape((1,))(input0)
                inputs.append(input0)
                
            hidden = inputs
            for level in range(depth-1,0,-1):
                nodes_by_level = Connect(index_root, level)
                idx = 0
                hidden_tmp = []
                for i, ele in enumerate(nodes_by_level):
                    # if len(ele.data) > 2:
                    #     raise('Wrong Tree Structure')
                    if isinstance(ele.data, int):
                        hidden_tmp.append(hidden[idx])
                        idx += 1
                    elif len(ele.data) == 2:
                        node_input = keras.layers.concatenate([hidden[idx], hidden[idx+1]], axis = 1)
                        encoded = model.get_layer('h_'+str(level)+'_'+str(i))(node_input)
                        encoded = model.get_layer('o_'+str(level)+'_'+str(i))(encoded)
                        
                        hidden_tmp.append(encoded)
                        idx += 2
                        
                        model_tmp = Model(inputs = input_path, outputs = encoded)
                        tree_outputs[ele.data] = model_tmp.predict(x_plot)
                hidden = hidden_tmp

            return tree_outputs

        def plot_encode(model, xx, yy, level, pos):
            inputs = np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1))))
            # print(inputs.shape)
            input_path = Input(shape = (2,))
            encoded = model.get_layer('h_'+str(level)+'_'+str(pos))(input_path)
            encoded = model.get_layer('o_'+str(level)+'_'+str(pos))(encoded)
            model_tmp = Model(inputs = input_path, outputs = encoded)
            output = model_tmp.predict(inputs)
            # if level == 1:
            #     output = np.exp(output)-1
            return output

        def IANN_plot3d(x, y, z, x_label, y_label, z_label, heatmap = heatmap):
            fig = plt.figure()
            fig.patch.set_facecolor('white')
            if heatmap:
                ind = np.array(np.argsort(y, axis = 0))[:,0].flatten()
                y = y[ind,]
                ax1 = fig.add_subplot(111)
                ax1.set_xlabel(xlabel)
                ax1.set_ylabel(ylabel)
                # ax1.set_title('Visualization for level %d' % stage)
                CS = ax1.contour(x, y, z, 5, colors = 'k')
                ax1.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=12)
                CP = ax1.contourf(x, y, z, 10, cmap = cm.coolwarm, extend='both')
                plt.colorbar(CP)
            else:
                ax1 = fig.add_subplot(111, projection = '3d')
                ax1.tick_params(axis='both', which='major', labelsize=12)
                ax1.set_xlabel(xlabel, fontsize = 20)
                ax1.set_ylabel(ylabel, fontsize = 20)
                ax1.set_zlabel(zlabel, fontsize = 20)
                # ax1.set_title('Visualization for level %d' % stage)
                azimuth = 135+180
                altitude = 45
                ax1.view_init(altitude, azimuth)
                cmap = matplotlib.colors.ListedColormap("red") 
                ax1.plot_trisurf(x.flatten(), y.flatten(), z.flatten(), cmap = cm.coolwarm)   
                # ax1.scatter(x.flatten(), y.flatten(), z.flatten())         
        
        tree_outputs = tree_encode(model, x_plot)
        tree_ranges = dict()
        for key in tree_outputs:
            tree_ranges[key] = [tree_outputs[key].min(), tree_outputs[key].max()]  
            # tree_ranges[key] = [np.percentile(tree_outputs[key], 5), np.percentile(tree_outputs[key], 95)]    
        
        nodes_by_level = dict()
        nodes_by_level_ori = dict()
        for l in range(1, depth+1):
            nodes_by_level[l] = Connect(index_root, l)
            nodes_by_level_ori[l] = ConnectOriginal(index_root, l)
            
        nodes = [(index_root, 1)]
        while nodes != []:
            # print([i[0].data for i in nodes])
            x, l = nodes.pop(0)
            xleaf, yleaf = False, False
            if x.left is None or x.right is None:
                continue
            if x.left:
                # print(x.left.data)
                nodes.append((x.left, l+1))
                pos = nodes_by_level_ori[l+1].index(x.left)
                if isinstance(x.left.data, int): 
                    # Original variable
                    index = x.left.data
                    dx = np.linspace(0, 1, num_x)
                    xleaf = True
                    xlabel = labels[index]
                else:
                    # xx = tree_outputs[x.left.data]
                    dx = np.linspace(*tree_ranges[x.left.data], num_x)
                    xlabel = '$h_{'+str(l)+', '+ str(pos+1) + '}$'
            if x.right:
                nodes.append((x.right, l+1))
                pos = nodes_by_level_ori[l+1].index(x.right)
                if isinstance(x.right.data, int): 
                    # Original variable
                    index = x.right.data
                    dy = np.linspace(0, 1, num_x)
                    yleaf = True
                    # yy = x_plot[:,pos] * (self.ub[index] - self.lb[index]) + self.lb[index]
                    ylabel = labels[index]
                else:
                    dy = np.linspace(*tree_ranges[x.right.data], num_x)
                    ylabel = '$h_{'+str(l)+', '+ str(pos+1) + '}$'
            
            xx, yy = np.meshgrid(dx, dy)
            pos = nodes_by_level_ori[l].index(x)
            zz = plot_encode(model, xx, yy, l, nodes_by_level[l].index(x))
            if l == 1:
                zlabel = '$f$'
                # zz = np.exp(zz)-1
            else:
                zlabel = '$h_{'+str(l-1)+', '+ str(pos+1) + '}$'
            
            if xleaf:
                index = x.left.data
                xx = xx * (self.ub[index] - self.lb[index]) + self.lb[index]
            if yleaf:
                index = x.right.data
                yy = yy * (self.ub[index] - self.lb[index]) + self.lb[index]
            
            IANN_plot3d(xx, yy, zz, xlabel, ylabel, zlabel)
    
            # Store the image
            if 1:
                file_path = os.getcwd() + '\\image\\'
                save_name = file_path + name + '_' + str(l)+'_'+ str(pos)
                plt.savefig(save_name, dpi=100)
                plt.show()
                plt.close()
                
            if l == 1:
                fig = plt.figure()
                fig.patch.set_facecolor('white')
                ax1 = fig.add_subplot(111, projection = '3d')
                ax1.tick_params(axis='both', which='major', labelsize=12)
                ax1.set_xlabel('$h_1$', fontsize = 20)
                ax1.set_ylabel('$h_2$', fontsize = 20)
                ax1.set_zlabel('$f$', fontsize = 20)
                # ax1.set_title('Visualization for level %d' % stage)
                azimuth = 135+180
                altitude = 45
                ax1.view_init(altitude, azimuth)
                cmap = matplotlib.colors.ListedColormap("red") 
                ax1.plot_trisurf(xx.flatten(), yy.flatten(), zz.flatten(), cmap = cm.coolwarm)   
                
                file_path = os.getcwd() + '\\image\\'
                save_name = file_path + name
                plt.savefig(save_name, dpi=400)
                plt.show()
                plt.close()

    def IANN_encode(self, IANN, x_train, plot_level, level):

        '''
            generate h_i and h_{i+1} from the IANN
        '''
        v_val = Model(inputs = IANN.input, outputs = IANN.get_layer('linear_'+str(level-1)).output)
        v = v_val.predict(x_train)

        h_mid_val = Model(inputs = IANN.input, outputs = IANN.get_layer('dense_out_level_'+str(level)).output)
        h_mid = h_mid_val.predict(x_train)

        h_out_val = Model(inputs = IANN.input, outputs = IANN.get_layer('dense_out_level_'+str(level-1)).output)
        h_out = h_out_val.predict(x_train)
        
        return v, h_mid, h_out

    def first_stage_plot(self, exploration, plot_level, stage, x_plot, y_plot, disjoint_group, name,\
         num_x, num_y, nonadd_id, plot_var, r2):
        '''
            Make the 3D visualization plot for specific stage using the function:
            exploration_encode functionhzn.
        '''
        font = 15
        var_index = sum(disjoint_group, [])
        num_group = len(disjoint_group)

        v, h_mid, h_out = self.IANN_encode(exploration, x_plot, plot_level, stage)
        # print(h_mid)
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
        ax1.set_title("Test r2: %.2f%%" % (r2*100.0))
        if len(disjoint_group[stage-1]) == 1:
            index = disjoint_group[stage-1][0]
            index = nonadd_id[index]
            ax1.set_xlabel(plot_var[index], fontsize = font)
            # ax1.set_xlabel('$x$'+str(index+1), font = font)
        else:
            ax1.set_xlabel('$v$'+str(stage), fontsize = font)
        ax1.set_ylabel('h', fontsize = font)
        if stage == 1:
            ax1.set_zlabel('f', fontsize = font)
        else:
            ax1.set_zlabel('h'+str(stage-1))
        if stage == num_group - 1:
            if len(disjoint_group[stage]) == 1:
                index = disjoint_group[stage][0]
                index = nonadd_id[index]
                ax1.set_ylabel('$x$'+str(index+1), fontsize = font)
            else:
                ax1.set_ylabel('$v$'+str(stage+1), fontsize = font)

        # ax1.set_title('Visualization for level %d' % stage)

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
        if 1:
            file_path = os.getcwd() + '\\image\\'
            save_name = file_path + name +'.png'
            print(save_name)
            plt.savefig(save_name, dpi=400)
            plt.show()
            plt.close()

    def exp_plot_in_one(self, name, row = 2, column = 3, file_path = None, save_path = ''):
        if file_path is None:
            file_path = os.path.dirname(os.getcwd()) + '\\image\\'
        count = 1
        pic = Image.open(os.path.join(file_path,name+str(count)+'.png'))
        width, height = pic.size
        toImage = Image.new('RGBA', (column*width, row*height))
        for i in range(row):
            for j in range(column):
                pic = Image.open(os.path.join(file_path,name+str(count)+'.png'))
                #print(file_path+name+'_auto_'+str(count)+'.png')
                toImage.paste(pic, (j*width, i*height))
                count += 1
                if count > self.d:
                    break
            if count > self.d:
                break
        toImage.save(os.path.join(os.path.join(file_path,save_path), name[:-1]+'.png'))

    def backward_plot_in_one(self, name, disjoint_group, row = 2, column = 3, save_path = ''):
        num_group = len(disjoint_group)
        file_path = os.getcwd() + '\\image\\'
        count = 1
        pic = Image.open(file_path+name+str(count)+'.png')
        pic_table = Image.open(file_path+name+str(count)+'_linear.png')
        width, height = pic.size
        width_table, height_table = pic_table.size
        resize_width = int(height/height_table*width_table)
        toImage = Image.new('RGBA', (resize_width+width, row*height))
        for i in range(row):
            pic = Image.open(file_path+name+str(count)+'.png')
            toImage.paste(pic, (0*width, i*height))
            if column > 1:
                pic_linear = Image.open(file_path+name+str(count)+'_linear.png')
                pic_linear = pic_linear.resize((resize_width, height))
                toImage.paste(pic_linear, (1*width, i*height))
            count += 1
            if count >= num_group:
                break
        toImage.save(os.path.join(os.path.join(file_path,save_path), name+'.png'))
        