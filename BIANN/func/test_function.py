# -*- coding: utf-8 -*-
"""
Created on Mon 04/22/2019

@author: Shengtong Zhang
"""

"""
This code generates the test functions in 
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy as sp

class test():
    '''    
    The following functions are chosen from the paper
    "Lifted Brownian Kriging Models", Section 6
    '''
    #def __init__(self):
        
    def rescale(self, x, lb, ub):
        # rescale x from [0, 1] to [lb, ub] using linear transformation
        # lb: lower bound; ub: upper bound
        n, m = np.array(x).shape
        for i in range(m):
            x[:, i] = x[:, i] * (ub[i] - lb[i]) + lb[i]
        return np.array(x)
        
    def OTL(self, x_input):
        '''
        otl circuit function
        
        x_input = (R_b1, R_b2, R_f, R_c1, R_c2, B)
        R_b1 \in [50, 150]
        R_b2 \in [25, 70]
        R_f \in [0.5, 3.0]
        R_c1 \in [1.2, 2.5]
        R_c2 \in [0.25, 1.20]
        B \in [50, 300]
        '''
        lb = [50, 25, 0.5, 1.2, 0.25, 50]
        ub = [150, 70, 3.0, 2.5, 1.20, 300]
        x = self.rescale(np.copy(x_input), lb, ub)
        #print(x[:,5])
        V = 12 * x[:,1] / (x[:,0] + x[:, 1])
        y = x[:,5]*(x[:,4] + 9) + x[:, 2]
        f = (x[:,5]*(V+0.74)*(x[:,4]+9) + 11.35*x[:,2] +\
             0.74 * x[:,5] * x[:,2] * (x[:,4] + 9)/x[:,3])/y
        return f, lb, ub

    def piston(self, x_input):
        '''
        piston simulation function

        x_input = (M, S, V_0, k, P_0, T_a, T_0)
        '''
        lb = [30, 0.005, 0.002, 1000, 90000, 290, 340]
        ub = [60, 0.02, 0.01, 5000, 110000, 296, 360]
        x = self.rescale(np.copy(x_input), lb, ub)
        A = x[:,4] * x[:,1] + 19.62 * x[:,0] - x[:,3] * x[:,2]/x[:,1]
        V = x[:,1] / (2 * x[:,3]) * (np.sqrt(np.square(A) \
                        + 4 * x[:,3]*x[:,4]*x[:,5]*x[:,2] / x[:,6]) - A)
        f = 2 * np.pi * np.sqrt(x[:,0] / (x[:,3] + \
                np.square(x[:,1]) * (x[:,4] * x[:, 2] / np.square(V)) * (x[:, 5] / x[:, 6])))
        return f, lb, ub

    def borehole(self,x_input):
        '''
        x = (T_u, r, r_w, H_u, T_l, H_l, L, K_w)
        '''
        lb = [63070, 100, 0.05, 990, 63.1, 700, 1120, 9855]
        ub = [115600, 50000, 0.15, 1110, 116, 820, 1680, 12045]
        x = self.rescale(np.copy(x_input), lb, ub)
        f = 2 * np.pi*x[:,0]*(x[:,3] - x[:,5]) / (np.log(x[:,1]/x[:,2])*\
                                                  (1+2*x[:,6]*x[:,0]/(np.log(x[:,1]/x[:,2])*\
                                                    np.square(x[:,2])*x[:,7])+\
                                                   x[:,0]/x[:,4]))
        return f, lb, ub
    
    def wing_weight(self,x_input):
        '''
        x = (T_u, r, r_w, H_u, T_l, H_l, L, K_w)
        '''
        lb = [150, 220, 6, -10, 16, 0.5, 0.08, 2.5, 1700, 0.025]
        ub = [200, 300, 10, 10, 45, 1, 0.18, 6, 2500, 0.08]
        x = self.rescale(np.copy(x_input), lb, ub)
        f = 0.036*np.power(x[:,0], 0.758)*np.power(x[:,1], 0.0035)*np.power(x[:,2]/np.power(np.cos(x[:,3]*np.pi/180), 2), 0.6) \
                * np.power(x[:,4], 0.006) * np.power(x[:,5], 0.04) * np.power(100*x[:,6]/np.cos(x[:,3]*np.pi/180), -0.3) \
                * np.power(x[:,7] * x[:,8], 0.49) + x[:,0]*x[:,9]
                    
        return f, lb, ub    
    
    def trid(self,x_input):
        lb = [-4]*5
        ub = [4]*5
        x = self.rescale(np.copy(x_input), lb, ub)
        f = np.sum(np.square(x-1), axis = 1) - np.sum(x[:,1:]*x[:,:-1], axis = 1)
        # f = np.square(np.sum(x[:,:5], axis = 1)) + np.square(np.sum(x[:,3:], axis = 1))
                    
        return f, lb, ub, 'trid_' 
    
    def physical(self,x_input):
        eqn = 1
        if eqn == 0:
            fig_name = 'rutherford_scattering_'
            lb = [0.5]*8 + [np.pi/2]
            ub = [1.5]*8 + [3*np.pi/2]
            x = self.rescale(np.copy(x_input), lb, ub)
            f = x[:,0] * x[:, 1] * x[:, 2] * np.square(x[:, 3]*x[:, 5]) * np.power(x[:, 4], 4) / \
                    (4*np.square(x[:, 6] * x[:, 7]) * np.power(np.sin(x[:,8]/2), 4))
        if eqn == 1:
            # x = [m1, m2, G, c, r]; the rate of orbital decay, https://en.wikipedia.org/wiki/Gravitational_wave
            fig_name = 'radiated_grav_'
            lb = [0, 0, 0, 1, 1] 
            ub = [1, 1, 1, 2, 2]
            x = self.rescale(np.copy(x_input), lb, ub)
            f = - 64/5 * np.power(x[:,2], 3) / np.power(x[:,3], 5) * (x[:,0]*x[:,1]) * (x[:,0] + x[:,1]) / np.power(x[:,4], 3)
                    
        return f, lb, ub, fig_name
    
    def plot_path(self,x):
        q=50
        x0 = np.reshape(np.arange(q) / q, (q, 1))
        x = np.repeat(np.reshape(x, (1, -1)), q, axis = 0)
        print(x.shape)
        print(x0.shape)
        x = np.hstack((x[:,:3], x0, x[:, 3:]))
        y, lb, ub = self.piston(x)
        plt.plot(x0, y)
        plt.show()

