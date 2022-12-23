#!/usr/bin/env python3

"""Dual number implementation for AD forward mode.

This module contains dunder methods to overload built-in Python operators. 
"""

import numpy as np

class Dual:
    
    _supported_scalars = (int, float, np.float64)

    def __init__(self, real, dual = 1):
        self.real = real 
        self.dual = dual 

    def __add__(self, other):
        """
        overload add operation
        """

        if not isinstance(other, (*self._supported_scalars, Dual)):
            raise TypeError(f'Type not supported for Dual number operations')
        if isinstance(other, self._supported_scalars):
            return Dual(self.real + other, self.dual)
        else:
            return Dual(self.real + other.real, self.dual + other.dual)



    def __radd__(self, other): 
        """
        overload reverse subtration operation
        """
        return self.__add__(other) 

    def __sub__(self, other):
        """
        overload subtraction operation
        """
        if not isinstance(other, (*self._supported_scalars, Dual)):
            raise TypeError(f'Type not supported for Dual number operations')
        if isinstance(other, self._supported_scalars):
            return Dual(self.real - other, self.dual)
        else:
            return Dual(self.real - other.real, self.dual - other.dual)



    def __rsub__(self, other):
        """
        overload reverse subtraction operation
        """
        return Dual(other - self.real, -self.dual)


    def __mul__(self, other): 
        """
        overwrite multiplication operation
        """
        if not isinstance(other, (*self._supported_scalars, Dual)):
            raise TypeError(f'Type not supported for Dual number operations')
        if isinstance(other, self._supported_scalars):
            return Dual(other*self.real, other*self.dual)
        else:
            return Dual(self.real*other.real, self.dual*other.real + self.real*other.dual)



    def __rmul__(self, other):
        """
        overwrite reverse multiplication operation
        """
        return self.__mul__(other)

    def __pow__(self, other):
        """
        overwrite power law operation
        """
        if not isinstance(other, self._supported_scalars):
            raise TypeError(f'Type not supported for Dual number operations')
        if isinstance(other, self._supported_scalars):
            return Dual(self.real**other, other*self.real**(other - 1)*self.dual)
        
    def __rpow__(self, other):
        """
        overwrite reverse power law operation
        """
        if not isinstance(other, self._supported_scalars):
            raise TypeError(f'Type not supported for Dual number operations')
        if isinstance(other, self._supported_scalars):
            return Dual(other**self.real, np.log(other)*other**self.real*self.dual)

    def __truediv__(self, other): 
        """
        Overload the division operator (/) to handle Dual class
        """
        if not isinstance(other, (*self._supported_scalars, Dual)):
            raise TypeError(f'Type not supported for Dual number operations')
        if isinstance(other, self._supported_scalars):
            return Dual(self.real/other,self.dual/other)
        else:
            return Dual(self.real/other.real, self.dual/other.real - self.real*other.dual/other.real/other.real) 

    def __rtruediv__(self, other):
        """
        Overload the reverse division operator (/) to handle Dual class
        """
        return Dual(other/self.real, -other*self.dual/self.real/self.real )

    def __neg__(self):
        """
        Overload the negative operator to handle Dual class
        """         
        return Dual(-self.real, -self.dual) 


    def __neq__(self, other):
        """ 
        Overload the inequality operator (!=) to handle Dual class
        """
        if isinstance(other, Dual):
            return self.real != other.real
        return self.real != other
        

    def __lt__(self, other):
        """
        Overload the less than operator to handle Dual class
        """
        if isinstance(other, Dual):
            return self.real < other.real
        return self.real < other 

    def __gt__(self, other):
        """ 
        Overload the greater than operator to handle Dual class
        """
        if isinstance(other, Dual):
            return self.real > other.real
        return self.real > other 


    def __le__(self, other):
        """ 
        Overload the  <= operator to handle Dual class
        """
        if isinstance(other, Dual):
            return self.real <= other.real
        return self.real <= other 

    def __ge__(self, other):
        """ 
        Overload the  >= operator to handle Dual class
        """
        if isinstance(other, Dual):
            return self.real >= other.real
        return self.real >= other 

    def __repr__(self):
        """
        Print class definition
        """
        return f'Dual({self.real},{self.dual})' 

    def __str__(self):
        """
        prettier string representation
        """
        return f'Forward mode dual number object(real: {self.real}, dual: {self.dual})'

    def __len__(self):
        """
        Return length of input vector
        """
        return (type(self.real) in (int, float)) and (type(self.dual) in (int, float))

    def __eq__(self,other):
        if isinstance(other, Dual):
            return (self.real == other.real and self.dual == other.dual)
        return self.real == other


