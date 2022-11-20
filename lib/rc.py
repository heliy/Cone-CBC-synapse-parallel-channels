#coding:UTF-8

"""
"""

DT = 1

U_REST = 0.05
U_MAX = 5

# labels for setting functions in the RAB module
STABLE_FUNC_NAME = "stable_response_func"
ALPHA_FUNC_NAME = "alpha_p_func"
EXT_NAMES = {
    STABLE_FUNC_NAME: "Stable",
    ALPHA_FUNC_NAME: "Release",
}

# defination of functions for candidates
FUNC_PARAMS_N = {
    "linear": 1,
    "linear_kb": 2,
    "linear_b_fix": 1,
    "relu": 3,
    "sigmoid": 3,
    "sigmoid_fix": 2,
    "1minhill": 2,
    "hill": 3,
    "rlinear": 2,
    "rlinear_fix": 1,
    "exp": 3,
    "exp_fix": 1,
    "pow": 3,
}

FUNC_NAMES = {
    "1minhill": "Hill",
    "linear": "Linear",
    "exp": "Exp",
    "linear_b_fix": "Linear",
    "sigmoid_fix": "Sigmoid",
    "exp_fix": "Exp",
    "hill": "Hill",
    "pow": "Pow",
    "relu": "Relu",
    "linear_kb": "Linear",
}
