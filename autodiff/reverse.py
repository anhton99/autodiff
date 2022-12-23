#!/usr/bin/env python3
import numpy as np
from autodiff.dual import Dual

class Node:
    """
    Node class to implement the reverse mode auto differentiation. Elementary operations are overloaded to create the tree structure
    to represent the function. A forward pass process is implemented in the _
    """
    _supported_scalars = (int, float, np.float64)

    def __init__(self, key, *, value = None, left_partial = None , right_partial = None, operation = None, left = None, right = None, sensitivity = 0):
        self.key = key
        self.left = left
        self.right = right
        self.value = value
        self.left_partial = left_partial  ## save partial at the self level is not the best choice. => does not account for recycled nodes unless leaf nodes are redefined 
        self.right_partial = right_partial
        self.operation = operation # the elementary operation performed at each node
        self.sensitivity = sensitivity
        self._eval()


    def __add__(self, other):
        """
        overload addition operation
        """
        #self.partial = 1 #### Calculate partial at the creation step will not work for sin, cos, etc!!!
        #other.partial = 1
        if not isinstance(other, (*self._supported_scalars, Node)):
            raise TypeError(f'Type not supported for reverse mode auto differentiation')
        if isinstance(other, self._supported_scalars):
            operation = lambda x: x + other
            return Node('add', left = self, right = None, operation = operation)
        else:
            operation = lambda x,y: x+y 
            return Node('add', left = self, right = other, operation = operation)

    def __radd__(self, other): 
        """
        overload reverse addition operation
        """
        return self.__add__(other) 

    def __sub__(self, other):
        
        #self.partial = 1
        #other.partial = -1
        if not isinstance(other, (*self._supported_scalars, Node)):
            raise TypeError(f'Type not supported for reverse mode auto differentiation')
        if isinstance(other, self._supported_scalars):
            operation = lambda x: x - other
            return Node('sub', left = self, right = None, operation = operation)
        else:
            operation = lambda x,y: x-y 
            return Node('sub', left = self, right = other, operation = operation)

    def __rsub__(self, other): 
        """
        overload reverse subtraction operation
        """
        return -self.__sub__(other) 

    def __mul__(self, other):
        
        #self.partial = other.value
        #other.partial = self.value
        if not isinstance(other, (*self._supported_scalars, Node)):
            raise TypeError(f'Type not supported for reverse mode auto differentiation')
        if isinstance(other, self._supported_scalars):
            operation = lambda x: x*other
            return Node('mul', left = self, right = None, operation = operation)
        else:
            operation = lambda x,y: x*y 
            return Node('mul', left = self, right = other, operation = operation)

    def __rmul__(self, other): 
        """
        overload reverse multiplication operation
        """
        return self.__mul__(other) 

    def __truediv__(self, other):
        """
        overload division operation
        """
        if not isinstance(other, (*self._supported_scalars, Node)):
            raise TypeError(f'Type not supported for reverse mode auto differentiation')
        if isinstance(other, self._supported_scalars):
            operation = lambda x: x/other
            return Node('div', left = self, right = None, operation = operation)
        else:
            operation = lambda x,y: x/y
            return Node('div', left = self, right = other, operation = operation)

    def __rtruediv__(self, other): 
        """
        overload reverse division operation
        """
        if not isinstance(other, (*self._supported_scalars, Node)):
            raise TypeError(f'Type not supported for reverse mode auto differentiation')
        else: 
            operation = lambda x: other/x
            return Node('div', left = self, right = None, operation = operation)
 
    def __pow__(self, other):
        """
        overload the power operation
        """
        if not isinstance(other, (*self._supported_scalars, Node)):
            raise TypeError(f'Type not supported for reverse mode auto differentiation')
        if isinstance(other, self._supported_scalars):
            operation = lambda x: x**other
            return Node('pow', left = self, right = None, operation = operation)
        else:
            operation = lambda x,y: x**y
            return Node('pow', left = self, right = other, operation = operation)

    def __rpow__(self, other):
        if not isinstance(other, (*self._supported_scalars, Node)):
            raise TypeError(f'Type not supported for reverse mode auto differentiation')
        else: 
            operation = lambda x: other**x
            return Node('exp', left = self, right = None, operation = operation)

    def __neg__(self):
        """
        overload the unary negation operation
        """
        operation = lambda x: -x
        return Node('neg', left = self, right = None, operation = operation)


    def __lt__(self, other):
        """
        overload the < operation
        """
        if not isinstance(other, (*self._supported_scalars, Node)):
            raise TypeError(f'Type not supported for reverse mode auto differentiation')
        elif isinstance(other, Node):
            return self.value < other.value
        else:
            return self.value < other 

    def __gt__(self, other):
        """
        overload the > operation
        """
        if not isinstance(other, (*self._supported_scalars, Node)):
            raise TypeError(f'Type not supported for reverse mode auto differentiation')
        elif isinstance(other, Node):
            return self.value > other.value
        else:
            return self.value > other 

    def __eq__(self, other):
        """
        overload the = operation
        """
        if not isinstance(other, (*self._supported_scalars, Node)):
            raise TypeError(f'Type not supported for reverse mode auto differentiation')
        elif isinstance(other, Node):
            return self.value == other.value and self.sensitivity == other.sensitivity
        else:
            return self.value == other 

    def __ne__(self, other):
        """
        overload the != operation
        """
        if not isinstance(other, (*self._supported_scalars, Node)):
            raise TypeError(f'Type not supported for reverse mode auto differentiation')
        elif isinstance(other, Node):
            return self.value != other.value or self.sensitivity != other.sensitivity
        else:
            return self.value != other 


    def __le__(self, other):
        """
        overload the <= operation
        """
        if not isinstance(other, (*self._supported_scalars, Node)):
            raise TypeError(f'Type not supported for reverse mode auto differentiation')
        elif isinstance(other, Node):
            return self.value <= other.value
        else:
            return self.value <= other 

    def __ge__(self, other):
        """
        overload the >= operation
        """
        if not isinstance(other, (*self._supported_scalars, Node)):
            raise TypeError(f'Type not supported for reverse mode auto differentiation')
        elif isinstance(other, Node):
            return self.value >= other.value
        else:
            return self.value >= other 

    def __str__(self):
        return self._pretty(self)

    def _eval(self):
        """
        Forward pass of the reverse mode auto differentiation.
        Calculate the value of all nodes of the tree, as well as the partial derivative of the current node wrt all child nodes.
        """
        
        #if (self.left is None) and (self.right is None):
        #    return self.value
        if self.value is not None:
            return self.value
        elif self.right is None:
            dual = self.operation(Dual(self.left._eval()))   # real part evaluates the current node, dual part evaluates the partial derivative
            self.value = dual.real
            self.left_partial = dual.dual
            return self.value
        else: 
            self.left._eval()
            self.right._eval()
            dual1 = Dual(self.left.value, 1)
            dual2 = Dual(self.right.value, 0)
            dual = self.operation(dual1, dual2)
            self.value = dual.real
            self.left_partial = dual.dual
            self.right_partial = self.operation(Dual(self.left.value, 0), Dual(self.right.value, 1)).dual
            return self.value


    
    def _sens(self):
        """
        Reverse pass of the reverse mode auto differentiation.
        Calculate the sensitivity (adjoint) of all child nodes with respect to the current node 
        """
        
        if (self.left is None) and (self.right is None):
            pass
        elif self.right is None:
            self.left.sensitivity += self.sensitivity*self.left_partial
            self.left._sens()
        else: 
            self.left.sensitivity += self.sensitivity*self.left_partial
            self.right.sensitivity += self.sensitivity*self.right_partial
            self.left._sens()
            self.right._sens()

    def _reset(self):
        """
        Reset the sensitivty of leaf nodes too zero to allow the reverse mode auto differentiation of the next component of a vector function.
        Calculate the sensitivity (adjoint) of all child nodes with respect to the current node 
        """
        
        if (self.left is None) and (self.right is None):
            pass

        elif self.right is None:
            self.left.sensitivity = 0
            self.left._reset()
        else: 
            self.left.sensitivity = 0
            self.right.sensitivity = 0 
            self.left._reset()
            self.right._reset()



    @staticmethod
    def _pretty(node):
        """Pretty print the expression tree (called recursively)"""
        if node.left is None and node.right is None:
            return f'{node.key}' + f': value = {node.value}'
        if node.left is not None and node.right is None:
            return f'{node.key}({node._pretty(node.left)})' + f': value = {node.value}'
        return f'{node.key}({node._pretty(node.left)}, {node._pretty(node.right)})' + f': value = {node.value}'


