import sys
sys.path.append('.')
import numpy as np
from autodiff.trig import *
import pytest     
from autodiff.autoDiff import ForwardDiff, ReverseDiff

class TestAutoDiff:
    
    def test_forwardDiff_init(self):
        func = lambda x: 3*x
        obj=ForwardDiff(func)
        assert obj.f == func

    def test_forwardDiff_deriv(self):
        ''' tests the derivative of elementary operations and trig operations'''
        func1 = lambda x: 3 * x
        func2 = lambda x: 3 + x
        func3 = lambda x: sin(x)
        func4 = lambda x: cos(x)
        func5 = lambda x: tan(x)
        func6 = lambda x: x**2
        func7 = lambda x: 3 - x
        func8 = lambda x: 3 / x

        obj1 = ForwardDiff(func1)
        obj2 = ForwardDiff(func2)
        obj3 = ForwardDiff(func3)
        obj4 = ForwardDiff(func4)
        obj5 = ForwardDiff(func5)
        obj6 = ForwardDiff(func6)
        obj7 = ForwardDiff(func7)
        obj8 = ForwardDiff(func8)
       
        assert obj1.derivative(x=2) == 3
        assert obj2.derivative(x=2) == 1
        assert obj3.derivative(x=2) == cos(2)
        assert obj4.derivative(x=2) == -sin(2)
        assert obj5.derivative(x=2) == 1 / (cos(2) * cos(2))
        assert obj6.derivative(x=2) == 2 * 2
        assert obj7.derivative(x=2) == -1
        assert obj8.derivative(x=2) == -3/4 

        # need to add vector of functions
        func_vector = lambda x: (3*x, cos(x))
        obj_vector = ForwardDiff(func_vector)
        assert obj_vector.derivative(x=2) == [3,-np.sin(2)]

    def test_forwardDiff_Jacobian(self):
        f=lambda x: x[0] + 2*x[1]
        obj=ForwardDiff(f)
        assert (obj.Jacobian([1,1]) == np.array([1,2])).all()

        # need to add vector of functions
        func_vector = lambda x: (x[0] + 2*x[1], x[0]+3*x[1])
        obj_vector = ForwardDiff(func_vector)
        
        assert (obj_vector.Jacobian([1,1]) == [[1,2],[1,3]]).all()


    def test_x_and_p_length_restrictions(self):
        f = lambda x: 3 * x
        obj = ForwardDiff(f)
        x = np.array([1,2,3])
        p = [1,1]
        # x and p are not the same length so this should throw an exception
        with pytest.raises(Exception):
            obj.derivative(x,p)

    def test_input_types(self):
       ''' makes sure that the derivative fsunction correctly raises an error when given an unsupported type as input'''
       f = lambda x: 4 + x
       obj = ForwardDiff(f)
       inputs = ['Bad', ('Bad'), {'bad': "Don't allow dictionaries"}]
       with pytest.raises(TypeError):
            for x in inputs:
                obj.derivative(x)


    def test_reverseDiff_init(self):
        func = lambda x: 3*x
        obj=ReverseDiff(func)
        assert obj.f == func

    def test_reverseDiff_Jacobian(self):
        f=lambda x: x[0] + 2*x[1]
        obj=ReverseDiff(f)
        assert (obj.Jacobian([1,1]) == np.array([1,2])).all()

        func_vector = lambda x: (x[0] + 2*x[1], x[0]+3*x[1])
        obj_vector = ReverseDiff(func_vector)
        print(obj_vector.Jacobian([1,1]))
        assert obj_vector.Jacobian([1,1]) == [[1,2],[1,3]]




        

