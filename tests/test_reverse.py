#!/usr/bin/env python3
import sys
sys.path.append('.')
import pytest
from autodiff.reverse import Node 
import numpy as np 

def test_add():
    """Test of addition special method (__add__) of Node class."""
    t = Node('x2', value = 4) + Node('x1', value = 2)
    assert t.value == 6

    f = Node('x1', value = 2) + 1.5
    assert f.value == 3.5

    with pytest.raises(TypeError):
        Node('x1', value = 2) + 'a'  

def test_radd():
    """Test of reverse addition special method (__radd__) of Node class."""
    t = Node('x2', value = 4) + Node('x1', value = 2)
    assert t.value == 6

    f =  1.5 + Node('x1', value = 2)
    assert f.value == 3.5

    with pytest.raises(TypeError):
        Node('x1', value = 2) + 'a'   

def test_sub():
    """Test of subtraction special method (__sub__) of Node class."""
    t = Node('x2', value = 4) - Node('x1', value = 2)
    assert t.value == 2

    f = Node('x1', value = 2) - 1.5
    assert f.value == 0.5

    with pytest.raises(TypeError):
        Node('x1', value = 2) - 'a' 

def test_rsub():
    """Test of reverse subtraction special method (__rsub__) of Node class."""
    t = Node('x2', value = 4) - Node('x1', value = 2)
    assert t.value == 2

    f =  1.5 - Node('x1', value = 2)
    assert f.value == -0.5

    with pytest.raises(TypeError):
        Node('x1', value = 2) - 'a'  

def test_mul():
    """Test of multiplication special method (__mul__) of Node class."""
    t = Node('x1', value = 2) * Node('x2', value = 4) 
    assert t.value == 8 

    num = Node('x1', value = 2) * 3
    assert num.value == 6

    f = Node('x1', value = 2) * 1.5
    assert f.value == 3.0

    with pytest.raises(TypeError):
        Node('x1', value = 2) * 'a'

def test_rmul():
    """Test of reverse multiplication special method (__rmul__) of Node class."""
    t = Node('x2', value = 4) * Node('x1', value = 2)
    assert t.value == 8 

    num = 3 * Node('x1', value = 2) 
    assert num.value == 6

    f = Node('x1', value = 2) * 1.5
    assert f.value == 3.0

    with pytest.raises(TypeError):
        Node('x1', value = 2) * 'a' 


def test_truediv():
    """Test of division special method (__truediv__) of Node class."""
    t = Node('x2', value = 2) / Node('x1', value = 4)
    assert t.value == 0.5

    num = Node('x1', value = 2) / 2
    assert num.value == 1.0

    with pytest.raises(TypeError):
        Node('x1', value = 2) / 'a'  

def test_rtruediv():
    """Test of reverse division special method (__rtruediv__) of Node class."""
    t = Node('x2', value = 2) / Node('x1', value = 4)
    assert t.value == 0.5

    num = 2/Node('x1', value = 2)
    assert num.value == 1.0

    with pytest.raises(TypeError):
        'a' / Node('x1', value=2)

def test_pow():
    """Test of power special method (__pow__) of Node class."""
    t = Node('x2', value = 2)
    t2 = Node('x3', value=3)

    with pytest.raises(TypeError):
        Node('x1', value = 2) ** 'a'   

def test_rpow():
    """Test of reverse power special method (__rpow__) of Node class."""
    t = 2**Node('x2', value = 3)  
    assert t.value == 8

    with pytest.raises(TypeError):
        'a' ** Node('x1', value=2)

def test_neg():
    """Test of unary negation special method (__neg__) of Node class."""
    t = -Node('x1', value = 2)
    assert t.value == -2

    with pytest.raises(TypeError):
        -Node('x1', value = 'a') 

def test_str():
    """Test of the string special method (__str__) of Node class."""
    assert(str(-Node('x1', value = 2))) == "neg(x1: value = 2): value = -2"
    assert(str(Node('x1', value = 2) / 2)) == "div(x1: value = 2): value = 1.0"
    assert(str(Node('x1', value = 4) / Node('x2', value = 2))) == "div(x1: value = 4, x2: value = 2): value = 2.0"

def test_lt():
    '''Check if less than operator is functional'''
    t1 = Node('x1', value=1)
    t2 = Node('x2', value=2)
    assert t1 < 2
    assert t1 < t2

    with pytest.raises(TypeError):
        t1 < 'bad type'
    
def test_gt():
    '''Check if the greater than operator works as intended'''
    t1 = Node('x1', value=1)
    t2 = Node('x2', value=2)
    assert t1 > 0
    assert t2 > t1

    with pytest.raises(TypeError):
        t1 > 'bad type'

def test_eq():
    '''Check if the equal  operator works as intended'''
    t1 = Node('x1', value=1)
    t2 = Node('x2', value=1)
    assert t1 == 1
    assert t2 == t1

    with pytest.raises(TypeError):
        t1 == 'bad type'


def test_ne():
    '''test to make sure the not equal operator is working as intended'''
    t1 = Node('x1', value=1)
    t2 = Node('x2', value=2)
    assert t1 != 2
    assert t2 != t1

    with pytest.raises(TypeError):
        t1 != 'bad type'


def test_le():
    '''test to make sure the <= operator is working as intended'''
    t1 = Node('x1', value=1)
    t2 = Node('x2', value=2)
    assert t1 <= 2
    assert t1 <= 1
    assert t1 <= t2
    assert t1 <= t1

    with pytest.raises(TypeError):
        t1 <= 'bad type'

def test_ge():
    ''' greater than operator'''
    t1 = Node('x1', value=1)
    t2 = Node('x2', value=2)
    assert t1 >= 0
    assert t1 >= 1
    assert t1 >= t1
    assert t2 >= t1

    with pytest.raises(TypeError):
        t1 >= 'bad type'

