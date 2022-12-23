## Introduction
AD27 is a library for Automatic Differentiation that provides users a simple and intuitive API for the automatic calculation and storage of derivatives.
For more details on background and project implementation, see the `docs` folder. 

Contributors: Anqi Chen, Emma Ton, Walt Williams, Wenyun Wang, Zhang Wu

## Software Organization
```
AD27/
├── __init__.py
|
└── autodiff/
    ── __init__.py
        ├── autoDiff.py
        ├── dual.py
        ├── reverse.py
        ├── trig.py        
|
├── docs
│   ├── graph_example.md
│   ├── milestone1.md
│   ├── milestone2.md
│   └── milestone2_progress.md
│   └── milestone3.md
|
├── tests/
│   ├── __init__.py
│   └── test_autoDiff.py
│   └── test_dual.py
│   └── test_reverse.py
│   └── test_trig.py
|
├── .DS_Store
├── .gitignore
├── LICENSE
├── README.md  
├── check_coverage.py    
├── requirements.txt     
├── run_tests.sh
├── pyproject.toml

```
- `autoDiff` module that defines both ForwardDiff and ReverseDiff class to compute the derivative of a function at a given point x and direction p or the Jacobian at a given point x with forward mode and reverse mode automatic differentiation, respectively. It will return a numpy array that represents the directional derivative or the Jacobian of the function that was passed to it. 

- `dual` module that defines the Dual class which overloads basic and comparison operators of +, -, *, ^, /, negation, =, <, >, <=, >=, !=, etc for dual numbers.

- `trig` module that overloads the basic trigonometric operators of sin, cos, tan, log, log10, log2, sinh, cosh, tanh, exp, sqrt, power, arcsin, arccos, arctan and etc for dual numbers as well as Node objects.

- `reverse` module that defines the Node class which overloads basic and comparison operators of +, -, *, ^, /, negation, =, <, >, <=, >= for Node objects, calculates the corresponding value, forward pass and reverse pass (sensivity) of a node in a expression tree as well as prints the expression tree. 

## Code Testing
- We use CI to perform tests and the tests live in the tests folder. We also generate a code coverage report for the test suites.

## How to Install Our Package

Our package is released on PyPI. Therefore, you will be able to easily pip install our package with the following command:
```bash
$ python3 -m pip install AD27
```
## How to Use Our Package

You will first need to create a virtual environment based on the version of Python interpreter you are using.

- Step1. Change into a working directory where you want to install the package:
```
$ cd ./YOUR_WORKING_DIRECTORY
```

- Step2. Create a .gitignore file in your working directory with the following:
```bash
$ echo '/test_env' >.gitignore
```

- Step3. Create your virtual Python environment by running the following command:
```python
$ python -m venv test_env
```

- Step4. Activtate the virtual enviornment with the following command:
```bash
$ source test_env/bin/activate
```

- Step5. Install our package

```bash
$ python3 -m pip install AD27
```

- Step6. Install the Numpy Library
```bash
$ pip install numpy==1.22.3
```

- Step7. Create a demo.py file
```bash
$ vim demo.py
```

- Step8. Import the modules from our package in your demo.py file:
```python
from autodiff.trig import *
from autodiff.autoDiff import ForwardDiff
from autodiff.autoDiff import ReverseDiff
from autodiff.reverse import Node
```

- Step9. Now you can start implement your own simple or more complex algorithms using the automatic differentiation either with ForwardDiff or ReverseDiff. A basic demo of a simple example is provided below:

```python
#!/usr/bin/env python3

from autodiff.trig import *
from autodiff.autoDiff import ForwardDiff
from autodiff.autoDiff import ReverseDiff
from autodiff.reverse import Node

def main():
    """
    Demo of a simple example with forward mode automatic differentiation

    Description: calculate the directional derivative and Jacobian of func = ∑Xi, 
    where Xi is the component in 2D dimensions, at the point x=[1,1] and direction p=[1,1]
    """
    
    # Note: if you have multiple function as inputs, you can just instantiate multiple objects of ForwardDiff or ReverseDiff class, with each corresponding to one of the function inputs

    # For scalar functions
    # Step 1: Define function
    function = lambda x: x * x

    
    # Step 2: Create forward mode autodifferentiation object
    forward_autodiffer =  ForwardDiff(function)
    
    # Step 3: Evaluate derivative of function at certain point
    dfdx = forward_autodiffer.derivative(x=3) # This will evaluate the derivative at x=3 so it will return 6 in this case
    
    
    # For vector valued functions
    # Step 1: Define function
    func = lambda x: x[0] + x[1]

    # Step 2: Create autodifferentiation object
    obj = ForwardDiff(func)
    print(obj.f)
    
    # Step 3: Find directional derivative and Jacobian of function (need to specify p for directional derivatives)
    print(obj.derivative(x =[1,1], p = [1,1]))
    print(obj.Jacobian([1,1]))


    """
    Demo of a simple example with reverse mode automatic differentiation

    Description: calculate the Jacobian of func = a-b+sin(c/d+e/f) + b*c, 
    where a,b,c,d,e,f are the multi-inputs for the function, at the test_vector = [0,1,2,3,4,5]
    """
    # Step 1: Intialize the vi (i<=0) nodes
    a,b,c,d,e,f = [Node(k) for k in list('abcdef')]
    for node,i in zip([a,b,c,d,e,f],range(len(list('abcdef')))):
        node.value = i
        print(node)   

    # Step 2: Define an express function and parse an expression tree
    func = lambda x: x[0]-x[1]+sin(x[2]/x[3]+x[4]/x[5]) + x[1]*x[2]
    tree = a-b*(c/d+e/f) 
    print(tree) # print the expression tree
    
    # Step 3: Create reverse mode autodifferentiation object
    reverse_autodiffer =  ReverseDiff(func)
    
    # Step 4: Evaluate jacobian of function at certain point represented by test_vector
    test_vector = [0,1,2,3,4,5]
    print(reverse_autodiffer.Jacobian(test_vector))


if __name__ == "__main__":
    main()
```

- Step10. Run the demo.py with the following command:
```bash
$ chmod +x demo.py
$ ./demo.py
```

- Step11. Deactivate the virtual environment with the following command:
```python
$ deactivate
```
