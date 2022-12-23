## Introduction

There are many problems in modern computation that require computing gradients and derivatives of complex functions. From calculating the movement or flow of electricity, motion of an object to and from like a pendulum, to explaining thermodynamics concepts, solving differential equations is essential for science and engineering. 

In the case of extremely complicated functions such as neural networks, the calculation and storage of these derivatives can be quite complex and cumbersome. The problem grows even worse when taking into account that the input to these functions can be extremely high-dimensional. To this end we introduce AD27, a library for Automatic Differentiation that provides users a simple and intuitive API for the automatic calculation and storage of derivatives.

## Background

**Automatic Differentiation** is a method of computing the derivative of a function using a computer program. 

Unlike symbolic differentiation - which requires manual calculations and exponential runtime - and numerical differentiation - which fails to give an accurate approximation when the error is too small or too large, automatic differentiation can **achieve machine precision** in constant time. 

To evaluate the derivative of a function using automatic differentiation, we decompose complex functional forms into a series of elementary function operations based on the order math is carried out by definition. After the decomposition, any complicated analytical formula of the original function can be represented by a computational graph. The independent variables and the outermost function will be the input and output of this graph. All intermediate variables will form a computational network in between, as is shown in the following example: 

$$f(x) = \pi x^2 + sin(3e^x + 7x) - e^x$$

![Image](./graph_example.png)

In this graphical representation, all dependent variables show up as nodes connected through elementary operations. By zooming in to a single parent-child node structure, we observe that the value of each dependent variable node can be evaluated once the value of all its parent nodes are given. 

To evaluate the derivatives of each dependent variable, we should recall the chain rule: 

$$\frac{\partial f}{\partial x} = \frac{\partial f}{\partial u}\frac{\partial u}{\partial x}$$

Therefore, the derivative of each child node variable with respect to a given direction in the independent variable space can be evaluated given the derivatives of all its parent nodes in that direction, as well as the value of the parent node variables. The series of values of dependent variables form the primal trace of the computational graph. The series of directional derivatives of the dependent variables form the tangent trace of the computational graph.

The actual evaluation of the primal and tangent trace depend on a tool structure - the dual number. A dual number z is written as $z=a+\epsilon b$, where a and b are real numbers while $\epsilon$ is a nilpotent number satisfying $\epsilon^2=0$ and $\epsilon \neq 0$. 

Once we store the value of a dependent variable as a, the real part of the dual number, and the directional derivative of a dependent variable as the b, the dual part of the dual number, the special properties of a dual number structure enables the combined evaluation of the primal trace and the tangent trance of a given parent-child node with dual number operation. 

The forward mode autodiff part of our package is implemented by defining the dual number structure, overloading dual number operations and evaluating node-by-node in the computational graph the dependent variables and their directional derivatives. 

## How to Use Our Package

You will first need to create a virtual environment based on the version of Python interpreter you are using.

Step1. Change into a working directory where you want to install the package:
```
$ cd ./YOUR_WORKING_DIRECTORY
```

Step2. Create a .gitignore file in your working directory with the following:
```bash
$ echo '/test_env' >.gitignore
```

Step3. Create your virtual Python environment by running the following command:
```python
$ python -m venv test_env
```

Step4. Activtate the virtual enviornment with the following command:
```bash
$ source test_env/bin/activate
```

- Step5. Install our package

At this point, the package is not on PyPI yet. We will install the package manually at milestone2.
```bash
$ git clone git@code.harvard.edu:CS107/team27.git
```

- Step6. Change into the team27 directory, and install the Numpy Library if you don't have it already
```bash
$ cd ./team27
$ pip install -r requirements.txt
```

- Step7. Launch a jupyter Notebook and create a new jupyter notebook
```bash
$ jupyter notebook
```
then create a new jupyter notebook by clicking the "New -> Python 3 (ipykernel)" on the top-right corner.

- Step8. Import the modules from our package in your jupyter notebook:
```python
from forward.trig import *
from forward.autoDiff import autoDiff
```

- Step9. Now you can start implement your own simple or more complex algorithms using the automatic differentiation. A basic demo of a simple example is provided below:

```python
# Description: calculate the directional derivative and Jacobian of func = ∑Xi, 
# where Xi is the component in 2D dimensions, at the point x=[1,1] and direction p=[1,1]
from forward.trig import *
from forward.autoDiff import autoDiff

def main():
    # For scalar functions
    # Step 1: Define function
    function = lambda x: x * x 
    
    # Step 2: Create autodifferentiation object
    autodiffer =  autoDiff(function)
    
    # Step 3: Evaluate derivative of function at certain point
    dfdx = autodiffer.derivative(x=3) # This will evaluate the derivative at x=3 so it will return 6 in this case
    
    
    # For vector valued functions (Will be further implemented for next milestone)
    # Step 1: Define function
    func = lambda x: x[0] + x[1]

    # Step 2: Create autodifferentiation object
    obj = autoDiff(func)
    print(obj.f)
    
    # Step 3: Find Jacobian of function (need to specify p for directional derivatives)
    print(obj.derivative(x =[1,1], p = [1,1]))
    print(obj.jacob([1,1]))


if __name__ == "__main__":
    main()
```

- Step10. Deactivate the virtual environment with the following command:
```python
$ deactivate
```

## Software Organization
### Directory Structure
```
team27/
├── __init__.py
├── check_coverage.py
|
├── docs
│   ├── milestone1.md
│   ├── milestone2.md
│   └── milestone2_progress.md
├── .gitignore
├── setup.py /* CI/CD set up  */
|
├── tests/
│   ├── __init__.py
│   └── test_autoDiff.py
│   └── test_dual.py
│   └── test_trig.py
|
└── forward/
    ── __init__.py
        ├── autoDiff.py
        ├── dual.py
        ├── trig.py        
    └── reverse/
        └── __init__.py
        ├── dual.py
        ├── node.py
        ├── computationgraph.py            
        ├── ... 
|           
├── run_tests.sh
├── pyproject.toml
├── requirements.txt
├── LICENSE
├── README.md  
```

### Basic Modules and Their Functionalities
- autoDiff module that computes the derivative of a function at a given point x and direction p or the Jacobian at a given point x. It will return a numpy array that represents the directional derivative or the Jacobian of the function that was passed to it.
- dual module that defines the Dual class and overloads basic operators of +, -, *, ^, /, negation, =, <, >, <=, >=.
- trig module that overloads the basic trigonometric operators of sin, cos, tan, log, log10, log2, sinh, cosh, tanh, exp, sqrt, power, arcsin, arccos, arctan and etc.

### Code Testing
- We will use CI to perform tests and the tests will live in the test folder.

### How to Install Our Package

At this point, the package is not on PyPI yet. We will install the package manually at milestone2. Users will need to install our package by manually clone our github repo into their own working directory on their computer. 

At final milestone, we will use PyPI to distribute our package. Therefore, you will be able to easily pip install our package after the final milestone is finished.

## Implementation Details
### Core Data Structures
Within the forward subpackage, dual numbers will be created in Dual class, which stores its real and dual parts as attributes and also has methods for operations. The multiple inputs of a function is stored as an array.

### Core Classes and Important Attributes
- DualNumber class in the forward subpackage

The DualNumber class has name attributes of real and dual, where the real stores the real part (i.e. function value) of a dual number, and the dual part stores the dual part (i.e., first-order derivative of a function. The methods includes basic operations for dual numbers such as add, sub, mul, etc. It will also include elementary functions, such as sin, cos, etc.

- autoDiff class in the forward subpackage

The autoDiff class has name attributes of function, which is taken in from the user input. It has methods of derivative (calculates the directional derivative of the target function at a given point in a specific direction), and jacob (calculats the Jacobian of the target function at a given point).

### External Dependencies
- NumPy Library

### Basic Operators and Elementary Functions
The basic operations for dual numbers are implemented as follow:

Addition (commutative) (+): real=real1+real2, dual=dual1+dual2

Subtraction (-): real=real1-real2, dual=dual1-dual2

Multiplication (*) (commutative): real=real1*real2, dual=real1*dual2+dual1*real2

Division (/): real=real1/real2, dual=(dual1*real2-real1*dual2)/real2**2

Power (**): real and dual number part can be calculated by performing power(n) times multiplication operation with itself.

Negation (-): real=-real, dual=-dual

#### Method - Elementary Functions: 
Elementary functions are calculated with the formula: real=f(real1), dual=f'(real1)*dual1. For example: sin(dualnumber1)=sin(real1)+cos(real1)*dual1*ε

### Other Aspects of Implementation On the Plan
- Function overloading to handle cases for f: R^m - > R^n
We will define a multifunction decorator. Inside the multifunction decorator, there will first be an __init__ constructor which creates a multifunction object with a given name and an empty typemap. Secondly, there will be a dunder __call__ method which makes the multifunction object itself callable. Thirdy, there will be register() method which adds a new function to the typemap, which is similar to how we define a wrapper function in PP7.

- extensions

## Future Features
- Reverse mode

