#!/usr/bin/env python3
import sys
sys.path.append('.')
import pytest
from autodiff.trig import *
import numpy as np 
from autodiff.dual import Dual 

def test_sin():
	"""Test of sin method of the trig class."""
	test = 3
	dual = Dual(3,2)
	assert sin(test) == np.sin(test)
	dual_result=np.cos(3)*2
	assert sin(dual) == Dual(np.sin(3),dual_result)
	test_string = 'test'
	with pytest.raises(TypeError):
		sin(test_string) 

	test_f=3.0
	assert sin(test_f) == np.sin(3.0)

	r=sin(Node('x', value=1.5))
	assert r.value == 0.9974949866040544

def test_cos():
	"""Test of cos method of the trig class."""
	test = 3
	dual = Dual(3,2)
	assert cos(test) == np.cos(test)
	assert cos(dual) == Dual(np.cos(3),-np.sin(3)*2)
	test_string = 'test'
	with pytest.raises(TypeError):
		cos(test_string) 

	test_f=3.0
	assert cos(test_f) == np.cos(3.0)

	r=cos(Node('x', value=1.5))
	assert r.value == 0.0707372016677029


def test_tan():
	"""Test of tan method of the trig class."""
	test = 3
	dual = Dual(3,2)
	assert tan(test) == np.tan(test)
	assert tan(dual) == Dual(np.tan(3),1/(np.cos(3))**2*2)
	test_string = 'test'
	with pytest.raises(TypeError):
		tan(test_string) 

	test_f=3.0
	assert tan(test_f) == np.tan(3.0)

	r=tan(Node('x', value=0.5))
	assert r.value == np.tan(0.5)


def test_log():
	"""Test of log method of the trig class."""
	test = 3
	dual = Dual(3,2)
	assert log(test) == np.log(test)
	assert log(dual) == Dual(np.log(3),1/3*2)
	test_string = 'test'
	with pytest.raises(TypeError):
		log(test_string) 

	test_f=3.0
	assert log(test_f) == np.log(3.0)

	r=log(Node('x', value=0.5))
	assert r.value == np.log(0.5)


def test_log2():
	"""Test of log method of the trig class."""
	test = 3
	d = log2(Dual(3,2)) 
	assert log2(test) == np.log2(test)
	assert d == Dual(np.log2(3), (1/(3 * np.log(2))) * 2)
	test_string = 'test'
	with pytest.raises(TypeError):
		log2(test_string) 

	test_f=3.0
	assert log2(test_f) == np.log2(3.0)

	r=log2(Node('x', value=2))
	assert r.value == 1.0

def test_log10():
	"""Test of log method of the trig class."""
	test = 3
	d = log10(Dual(3,2)) 
	assert log10(test) == np.log10(test)
	assert d == Dual(np.log10(3), (1/(3*np.log(10)))*2)
	test_string = 'test'
	with pytest.raises(TypeError):
		log10(test_string) 

	test_f=3.0
	assert log10(test_f) == np.log10(3.0)

	r=log10(Node('x', value=10))
	assert r.value == 1.0

def test_exp():
	"""Test of exp method of the trig class."""
	test = 3
	dual = Dual(3,2)
	assert exp(test) == np.exp(test)
	assert exp(dual) == Dual(np.exp(3),np.exp(3)*2)
	test_string = 'test'
	with pytest.raises(TypeError):
		exp(test_string) 

	test_f=3.0
	assert exp(test_f) == np.exp(3.0)

	r=exp(Node('x', value=0))
	assert r.value == 1.0

def test_sqrt():
	"""Test of sqrt method of the trig class."""
	test = 3
	dual = Dual(3,2)
	assert sqrt(test) == np.sqrt(test)
	assert sqrt(dual) == Dual(np.sqrt(3),0.5/np.sqrt(3)*2)
	test_string = 'test'
	with pytest.raises(TypeError):
		sqrt(test_string) 

	test_f=3.0
	assert sqrt(test_f) == np.sqrt(3.0)

	r=sqrt(Node('x', value=4))
	assert r.value == 2.0


def test_arcsin():
	"""Test of arcsin function method of the trig class."""
	test = 1/2
	dual = Dual(1/2,2)
	assert arcsin(test) == np.arcsin(test)
	assert arcsin(dual) == Dual(np.arcsin(1/2), 1 / np.sqrt(1 - (1/2) ** 2) * 2)
	test_string = 'test'
	with pytest.raises(TypeError):
		arcsin(test_string) 

	test_f=0.5
	assert arcsin(test_f) == np.arcsin(0.5)

	r=arcsin(Node('x', value=0.5))
	assert r.value == np.arcsin(0.5)
 

def test_arccos():
	"""Test of arccos function method of the trig class."""
	test = 0.5
	dual = Dual(0.5,2)
	assert arccos(test) == np.arccos(test)
	assert arccos(dual) == Dual(np.arccos(0.5), -1 / np.sqrt(1 - 0.5**2) * 2)
	test_string = 'test'
	with pytest.raises(TypeError):
		arccos(test_string) 

	test_f= 0.5
	assert arccos(test_f) == np.arccos(1/2)

	r=arccos(Node('x', value=0.5))
	assert r.value == np.arccos(0.5)

def test_arctan():
	"""Test of arctan function method of the trig class."""
	test = 3
	dual = Dual(3,2)
	assert arctan(test) == np.arctan(test)
	assert arctan(dual) == Dual(np.arctan(3), 1 / (1 + 3**2) * 2)
	test_string = 'test'
	with pytest.raises(TypeError):
		arctan(test_string) 

	test_f=3.0
	assert arctan(test_f) == np.arctan(3.0)

	r=arctan(Node('x', value=0))
	assert r.value == 0.0

def test_sinh():
	"""Test of sinh function method of the trig class."""
	test = 3
	dual = Dual(3,2)
	assert sinh(test) == np.sinh(test)
	assert sinh(dual) == Dual(np.sinh(3), np.cosh(3) * 2)
	test_string = 'test'
	with pytest.raises(TypeError):
		sinh(test_string) 

	test_f=3.0
	assert sinh(test_f) == np.sinh(3.0)

	r=sinh(Node('x', value=0.5))
	assert r.value == 0.5210953054937474

def test_cosh():
	"""Test of cosh function method of the trig class."""
	test = 3
	dual = Dual(3,2)
	assert cosh(test) == np.cosh(test)
	assert cosh(dual) == Dual(np.cosh(3), np.sinh(3) * 2)
	test_string = 'test'
	with pytest.raises(TypeError):
		cosh(test_string) 

	test_f=3.0
	assert cosh(test_f) == np.cosh(3.0)

	r=cosh(Node('x', value=0.5))
	assert r.value == 1.1276259652063807

def test_tanh():
	"""Test of tanh function method of the trig class."""
	test = 3
	dual = Dual(3,2)
	assert tanh(test) == np.tanh(test)
	assert tanh(dual) == Dual(np.tanh(3), 2 / np.cosh(3)**2)
	test_string = 'test'
	with pytest.raises(TypeError):
		tanh(test_string) 

	test_f=3.0
	assert tanh(test_f) == np.tanh(3.0)

	r=tanh(Node('x', value=0.5))
	assert r.value == 0.46211715726000974

def test_logist():
	"""Test of logistic function method of the trig class."""
	test = 3
	dual = Dual(3,2)

	def logist_real(x, loc=0, scale=1):
         
		return np.exp((loc-x)/scale)/(scale*(1+np.exp((loc-x)/scale))**2)

	def logist_dual(real, dual, loc=0, scale=1):
         
		return np.exp((loc-real)/scale)/(scale*(1+np.exp((loc-real)/scale))**2)/ \
                   (scale*(1+np.exp((loc-real)/scale))**2)**2* \
                   ((-1/scale)*(scale*(1+np.exp((loc-real)/scale))**2)- \
                   ((loc-real)/scale)*(scale*2*(1+np.exp((loc-real)/scale)))*np.exp((loc-real)/scale)*(-1)/scale)*dual

	assert logist(test) == logist_real(test)
	assert logist(dual) == Dual(logist_real(dual.real), logist_dual(dual.real, dual.dual))
	test_string = 'test'
	with pytest.raises(TypeError):
		logist(test_string) 

	test_f=3.0
	assert logist(test_f) == logist_real(test_f)

	r=logist(Node('x', value=0.5))
	assert r.value == 0.2350037122015945