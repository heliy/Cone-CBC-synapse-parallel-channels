import numpy as np

from .rc import *
from .utils import get_func_from_parameters
from .filters import ConeImpluseFilter

class LN:
    def __init__(self, parameters, dt=DT):
        self.linear = ConeImpluseFilter(gamma=parameters[0], A=1, dt=dt)
        self.nonlinear = get_func_from_parameters("sigmoid", parameters[-3:])
        self.name = "LN"
        self.dt = dt

    def init(self, stim):
        self.linear.init(stim)
        return self.nonlinear(stim)

    def run(self, trace, with_init=True):
        if with_init:
            self.init(trace[0])
        r = [self.linear.filter(trace)]
        r = [self.nonlinear(r[0])] + r
        return r

def generate_ITER_LN_grabner(idx):
    x0s = {
        2: [0.88254779, 0.5286513,  0.53503432,  1.05224104],
        3: [0.888300182821393, 0.5364457762200756, 0.5385562851075412, 1.0608594869286327],
    }
    ps = {}
    label = "cif"
    def iter_func():
        yield LN, x0s[idx], ps, label
    return [iter_func]
