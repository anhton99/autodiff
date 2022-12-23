#!/usr/bin/env python3
import numpy as np
import autodiff.trig as tr
from autodiff.dual import Dual
from autodiff.reverse import Node 


class ForwardDiff: 
    def __init__(self, f):
        self.f = f 

    def derivative(self, x, p=[1]):
        """ 
        Parameters
        ==========
        x : constant associated with each component of vector x
        p : direction at which the direcitonal derivative is evaluated 

        Returns
        =======
        the dual part of a Dualnumber

        Example: 
        =======
        z_i = Dual(x_i, p_i)
        f(z).real = f(x)
        f(z).dual = D_p_{f} 
        """
        scalars = [float, int, np.float64]
        if type(x) in scalars:
            z = Dual(x)
        elif isinstance(x, list) or isinstance(x, np.ndarray):
            if len(p)!=len(x):
                raise Exception('length of p should be the same as length of x')
            if len(x)==1:
                z=Dual(x[0])
            else:
                z = [0] * len(x) 
                for i in range(len(x)):
                    z[i] = Dual(x[i], p[i])
        else:
            raise TypeError(f'Unsupported type for derivative function. X is of type {type(x)}')

        if type(self.f(z)) is Dual:
            return self.f(z).dual 
        else:
            output=[]
            for i in self.f(z):
                output.append(i.dual)
            return output
        

    def Jacobian(self, x):
        # construct a dual number 
        deri_array = []
        for i in range(len(x)):
            p = np.zeros(len(x))
            p[i] = 1
            deri_array.append(self.derivative(x, p))
        return np.array(deri_array).T


 
class ReverseDiff:

    def __init__(self, f):
        self.f = f
        


    def Jacobian(self, vector):
        
        iv_nodes = [Node(1-k, value = vector[k]) for k in range(len(vector))] #nodes of independent variables, key value numbering according to vs
        #for i, iv_node in enumerate(iv_nodes):
        #    iv_node.value = vector[i]

        
        tree = self.f([*iv_nodes]) 
        #print(type(tree))
        if type(tree) is Node:
            #tree._eval()
            #print(tree)
            tree.sensitivity = 1
            tree._sens()
            return [iv_node.sensitivity for iv_node in iv_nodes]

        else:
            deri_array = []
            for line in tree:
                #line._eval()
                line._reset()
                line.sensitivity=1
                line._sens()
                
                line_partials = [iv_node.sensitivity for iv_node in iv_nodes]
                deri_array.append(line_partials )
            return deri_array


