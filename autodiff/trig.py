#!/usr/bin/env python3

import numpy as np 
from autodiff.dual import Dual 
from autodiff.reverse import Node


def sin(x):
    """
    overwrite sine function
    """
    supported_types = (int, float, np.float64, Dual, Node)
    if type(x) not in supported_types:
        raise TypeError('type of input argument not supported')
    elif type(x) is Dual:
        return Dual(np.sin(x.real), np.cos(x.real)*x.dual)
    elif type(x) is Node:
        return Node('sin', left = x, operation = lambda x:sin(x))
    else:
        return np.sin(x)  

def cos(x):
    """
    overwrite cosine function
    """
    supported_types = (int, float, np.float64, Dual, Node)
    if type(x) not in supported_types:
        raise TypeError('type of input argument not supported')
    elif type(x) is Dual:
        return Dual(np.cos(x.real), -np.sin(x.real)*x.dual)
    elif type(x) is Node:
        return Node('cos', left = x, operation = lambda x:cos(x))
    else:
        return np.cos(x)
 

def tan(x):
    """
    overwrite tangent
    """
    supported_types = (int, float, np.float64, Dual, Node)
    if type(x) not in supported_types:
        raise TypeError('type of input argument not supported')
    elif type(x) is Dual:
        return Dual(np.tan(x.real), 1/(np.cos(x.real))**2*x.dual)
    elif type(x) is Node:
        return Node('tan', left = x, operation = lambda x:tan(x))
    else:
        return np.tan(x)
 

def log(x):
    """
    overwrite log
    """
    supported_types = (int, float, np.float64, Dual, Node)
    if type(x) not in supported_types:
        raise TypeError('type of input argument not supported')
    elif type(x) is Dual:
        return Dual(np.log(x.real), 1/x.real*x.dual)
    elif type(x) is Node:
        return Node('log', left = x, operation = lambda x:log(x))
    else:
        return np.log(x)

def log2(x):
    """ 
    overwrite hyberbolic sine
    """
    supported_types = (int, float, np.float64, Dual, Node)
    if type(x) not in supported_types:
        raise TypeError('type of input argument not supported')
    elif type(x) is Dual:
        return Dual(np.log2(x.real), (1/(x.real*np.log(2)))*x.dual)
    elif type(x) is Node:
        return Node('log2', left = x, operation = lambda x:log2(x))
    else:
        return np.log2(x)   

def log10(x):
    """ 
    overwrite log10
    """
    supported_types = (int, float, np.float64, Dual, Node)
    if type(x) not in supported_types:
        raise TypeError('type of input argument not supported')
    elif type(x) is Dual:
        return Dual(np.log10(x.real), (1/(x.real*np.log(10)))*x.dual)
    elif type(x) is Node:
        return Node('log10', left = x, operation = lambda x:log10(x))
    else:
        return np.log10(x)   

def sinh(x):
    """ 
    overwrite hyberbolic sine
    """
    supported_types = (int, float, np.float64, Dual, Node)
    if type(x) not in supported_types:
        raise TypeError('type of input argument not supported')
    elif type(x) is Dual:
        return Dual(np.sinh(x.real), np.cosh(x.real) * x.dual)
    elif type(x) is Node:
        return Node('sinh', left = x, operation = lambda x:sinh(x))
    else:
        return np.sinh(x)  

def cosh(x):
    """ 
    overwrite hyberbolic cosine
    """
    supported_types = (int, float, np.float64, Dual, Node)
    if type(x) not in supported_types:
        raise TypeError('type of input argument not supported')
    elif type(x) is Dual:
        return Dual(np.cosh(x.real), np.sinh(x.real) * x.dual)
    elif type(x) is Node:
        return Node('cosh', left = x, operation = lambda x:cosh(x))
    else:
        return np.cosh(x)  

def tanh(x):
    """ 
    overwrite hyberbolic tangent
    """
    supported_types = (int, float, np.float64, Dual, Node)
    if type(x) not in supported_types:
        raise TypeError('type of input argument not supported')
    elif type(x) is Dual:
        return Dual(np.tanh(x.real), x.dual / np.cosh(x.real)**2)
    elif type(x) is Node:
        return Node('tanh', left = x, operation = lambda x:tanh(x))
    else:
        return np.tanh(x) 

def exp(x):
    """
    overwrite exponential
    """
    supported_types = (int, float, np.float64, Dual, Node)
    if type(x) not in supported_types:
        raise TypeError('type of input argument not supported')
    elif type(x) is Dual:
        return Dual(np.exp(x.real), np.exp(x.real) * x.dual)
    elif type(x) is Node:
        return Node('exp', left = x, operation = lambda x:exp(x))
    else:
        return np.exp(x)

def sqrt(x):
    supported_types = (int, float, np.float64, Dual, Node)
    if type(x) not in supported_types:
        raise TypeError('type of input argument not supported')
    elif type(x) is Dual:
        return Dual(np.sqrt(x.real), 1/2/np.sqrt(x.real) * x.dual)
    elif type(x) is Node:
        return Node('sqrt', left = x, operation = lambda x:sqrt(x))
    else:
        return np.sqrt(x)

def power(x, other):
    if type(x) is Node:
        return Node('sqrt', left = x, operation = lambda x:power(x))
    else:
        return x.__pow__(other) 

def arcsin(x):
    """ 
    overwrite arc sine
    """
    supported_types = (int, float, np.float64, Dual, Node)
    if type(x) not in supported_types:
        raise TypeError('type of input argument not supported')
    elif type(x) is Dual:
        return Dual(np.arcsin(x.real), 1 / np.sqrt(1 - x.real ** 2) * x.dual)
    elif type(x) is Node:
        return Node('arcsin', left = x, operation = lambda x:arcsin(x))
    else:
        return np.arcsin(x)   

def arccos(x):
    """ 
    overwrite arc cosine
    """
    supported_types = (int, float, np.float64, Dual, Node)
    if type(x) not in supported_types:
        raise TypeError('type of input argument not supported')
    elif type(x) is Dual:
        return Dual(np.arccos(x.real), -1 / np.sqrt(1 - x.real**2) * x.dual)
    elif type(x) is Node:
        return Node('arccos', left = x, operation = lambda x:arccos(x))
    else:
        return np.arccos(x)    

def arctan(x):
    """ 
    overwrite arc tangent
    """
    supported_types = (int, float, np.float64, Dual, Node)
    if type(x) not in supported_types:
        raise TypeError('type of input argument not supported')
    elif type(x) is Dual:
        return Dual(np.arctan(x.real), 1 / (1 + x.real**2) * x.dual)
    elif type(x) is Node:
        return Node('arctan', left = x, operation = lambda x:arctan(x))
    else:
        return np.arctan(x)    

def logist(x, loc=0, scale=1):
    """
    overwrite logistic
    default set loc and scale to be 0 and 1
    """
    supported_types = (int, float, np.float64, Dual, Node)
    if type(x) not in supported_types:
        raise TypeError('type of input argument not supported')
    elif type(x) is Dual:
        return Dual(np.exp((loc-x.real)/scale)/(scale*(1+np.exp((loc-x.real)/scale))**2), 
                   np.exp((loc-x.real)/scale)/(scale*(1+np.exp((loc-x.real)/scale))**2)/ \
                   (scale*(1+np.exp((loc-x.real)/scale))**2)**2* \
                   ((-1/scale)*(scale*(1+np.exp((loc-x.real)/scale))**2)- \
                   ((loc-x.real)/scale)*(scale*2*(1+np.exp((loc-x.real)/scale)))*np.exp((loc-x.real)/scale)*(-1)/scale)*x.dual)
    elif type(x) is Node:
        return Node('logist', left = x, operation = lambda x:logist(x))
    else:
        return np.exp((loc-x)/scale)/(scale*(1+np.exp((loc-x)/scale))**2)