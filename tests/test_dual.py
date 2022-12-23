#!/usr/bin/env python3
import pytest
import sys
sys.path.append('.')
from autodiff.dual import Dual 
import numpy as np 

def test_add():
    """Test of addition special method (__add__) of Dual class."""

    add_dual = Dual(1, 2) + Dual(3.5, 4)
    assert add_dual.real == 4.5 
    assert add_dual.dual == 6

    add_dual = Dual(4.0) + Dual(3.5)
    assert add_dual.real == 7.5
    assert add_dual.dual == 2

    add_dual = Dual(4.0) + 3
    assert add_dual.real == 7
    assert add_dual.dual == 1

    with pytest.raises(TypeError):
        Dual(1) + 'a'

def test_radd():
    """Test of reverse addition special method (__radd__) of Dual class."""
    radd_dual = 4 + Dual(3.5)
    assert radd_dual.real == 7.5 
    assert radd_dual.dual == 1
    
    with pytest.raises(TypeError):
        'a' + Dual(1)

def test_sub():
    """Test of subtraction special method (__sub__) of Dual class."""

    sub_dual = Dual(1, 2) - Dual(3.5, 4)
    assert sub_dual.real == -2.5
    assert sub_dual.dual == -2

    sub_dual = Dual(4.0) - Dual(3.5)
    assert sub_dual.real == 0.5
    assert sub_dual.dual == 0

    sub_dual = Dual(4.0) - 3
    assert sub_dual.real == 1
    assert sub_dual.dual == 1

    with pytest.raises(TypeError):
        Dual(1) - 'a'

def test_rsub():
    """Test of reverse subtraction special method (__rsub__) of Dual class."""
    rsub_dual = 4 - Dual(3.5)
    assert rsub_dual.real == 0.5
    assert rsub_dual.dual == -1

    with pytest.raises(TypeError):
        'a' - Dual(1)

def test_mul():
    """Test of multiplication special method (__mul__) of Dual class."""

    mul_dual = Dual(1, 2)*Dual(3, 4)
    assert mul_dual.real == 3
    assert mul_dual.dual == 10

    mul_dual = Dual(4.0)*Dual(3)
    assert mul_dual.real == 12
    assert mul_dual.dual == 7

    mul_dual = Dual(4.0)*3
    assert mul_dual.real == 12
    assert mul_dual.dual == 3

    with pytest.raises(TypeError):
        Dual(1)*'a'

def test_rmul():
    """Test of reverse multiplication special method (__radd__) of Dual class."""
    rmul_dual = 4*Dual(3)
    assert rmul_dual.real == 12
    assert rmul_dual.dual == 4

    with pytest.raises(TypeError):
        'a'*Dual(1)

def test_truediv():
    """Test of division special method (__truediv__) of Dual class."""
    
    div_dual = Dual(2, 4)/Dual(1, 2)
    assert div_dual.real == 2
    assert div_dual.dual == 0

    div_dual = Dual(4.0)/Dual(2)
    assert div_dual.real == 2
    assert div_dual.dual == -1/2

    div_dual = Dual(4.0)/3
    assert div_dual.real == 4/3
    assert div_dual.dual == 1/3

    with pytest.raises(TypeError):
        Dual(1)/'a'

def test_rtruediv():
    """Test of reverse division special method (__rtruediv__) of Dual class."""
    rdiv_dual = 4/Dual(3)
    assert rdiv_dual.real == 4/3
    assert rdiv_dual.dual == -4/9

    with pytest.raises(TypeError):
        'a'/Dual(1) 

def test_pow():
    """Test of power special method (__pow__) of Dual class."""

    pow_dual = Dual(2, 4)**3
    assert pow_dual.real == 8
    assert pow_dual.dual == 3*2**2*4


    pow_dual = Dual(4.0)**3
    assert pow_dual.real == 4**3
    assert pow_dual.dual == 3*4**2

    with pytest.raises(TypeError):
        Dual(1)**'a'

def test_rpow():
    """Test of reverse power law special method (__rpow__) of Dual class."""
    rpow_dual = 3**Dual(2,4)
    assert rpow_dual.real == 3**2
    assert rpow_dual.dual == np.log(3)*3**2*4

    with pytest.raises(TypeError):
        'a'**Dual(1)

def test_eq():
    """Test of the dunder equal method (__eq__) of the Dual class"""
    eq_dual = Dual(1,3)
    assert eq_dual == Dual(1.0,3.0)
    
    eq_dual = Dual(3.0)
    assert eq_dual == Dual(3,1)

def test_neg():
    """Test of the dunder negation method (__neg__) of the Dual class """
    neg_dual = -Dual(1)
    assert neg_dual.real == -1
    assert neg_dual.dual == -1


def test_repr():
    """Test of the representation special method (__repr__) of Dual class."""
    assert repr(Dual(6,7)) == 'Dual(6,7)'
    assert repr(Dual(4, 2.5)) == 'Dual(4,2.5)'


def test_str():
    """Test of the string special method (__str__) of Dual class."""
    assert str(Dual(2)) == 'Forward mode dual number object(real: 2, dual: 1)'
    assert str(Dual(2, 3.5)) == 'Forward mode dual number object(real: 2, dual: 3.5)'

def test_len():
    """Test of the length special method (__len__) of Dual class.""" 
    assert len(Dual(2)) == 1
    assert len(Dual(4.5, 5.5)) == 1

def test_neq():
    """Test of the the inequality operator (!=) to handle Dual class"""
    assert (Dual(5) != 5) == False
    assert (Dual(5) != 1) == True 

def test_lt():
    """Test of the less than operator to handle Dual class"""
    assert (Dual(2) < 5) == True
    assert (Dual(5) < 1) == False 

def test_gt():
    """Test of the greater than operator to handle Dual class"""
    assert (Dual(1) > 3) == False
    assert (Dual(10) > 1) == True 

def test_le():
    """Test of the <= operator to handle Dual class"""
    assert (Dual(5) <= 3) == False
    assert (Dual(3) <= 5) == True 
 

def test_ge():
    """Test of the  >= operator to handle Dual class"""
    assert (Dual(1) >= 3) == False
    assert (Dual(1) >= 1) == True 
