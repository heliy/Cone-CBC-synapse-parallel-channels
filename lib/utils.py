#coding:UTF-8

import numpy as np

def get_func_from_parameters(func_type, parameters):
    if func_type == "None":
        func = lambda x: x
    elif func_type == "constant":
        value = parameters[0]
        func = lambda x: value
    elif func_type == "sigmoid":
        half = parameters[0]
        slope = float(parameters[1])
        m = parameters[2]
        func = lambda x: m/(1.+np.exp(-(x-half)/slope))
    elif func_type == "sigmoid_fix":
        half = parameters[0]
        slope = float(parameters[1])
        func = lambda x: 1/(1.+np.exp(-(x-half)/slope))
    elif func_type == "1minhill":
        k = parameters[0]
        n = parameters[1]
        if k <= 0 or n <=0:
            raise Exception()
        func = lambda x: k**n/(k**n+x**n)
    elif func_type == "hill":
        k = parameters[0]
        n = parameters[1]
        m = parameters[2]
        if n <= 0 or m <= 0:
            raise Exception()
        func = lambda x: m*x**n/(k**n+x**n)
    elif func_type == "ahill":
        k = parameters[0]
        m = parameters[1]
        func = lambda x: m/(1.+k/x)
    elif func_type == "rlinear":
        k = parameters[0]
        m = float(parameters[1])
        func = lambda x: m/(x+k)
    elif func_type == "rlinear_fix":
        k = float(parameters[0])
        func = lambda x: k/(x+k)
    elif func_type == "exp":
        k = float(parameters[0])
        m = parameters[1]
        c = parameters[2]
        func = lambda x: m*np.exp(x/k)+c
    elif func_type == "exp_fix":
        k = float(parameters[0])
        func = lambda x: np.exp(x/k)
    elif func_type == "pow":
        A = parameters[0]
        B = parameters[1]
        C = parameters[2]
        if B <= 0:
            raise Exception()
        func = lambda x: A*x**B+C
    elif func_type == "relu":
        th = parameters[0]
        slope = parameters[1]
        base = parameters[2]
        def func(x):
            if x < th:
                return base
            else:
                return base+slope*(x-th)
    elif func_type == "linear":
        func = lambda x: parameters[0]*x
    elif func_type == "linear_kb":
        func = lambda x: parameters[0]*x+parameters[1]
    elif func_type == "linear_b_fix":
        func = lambda x: parameters[0]*x+1
    else:
        raise Exception("UNKNOWN func types for %s"%func_type)
    return np.vectorize(func, otypes=[np.float])
