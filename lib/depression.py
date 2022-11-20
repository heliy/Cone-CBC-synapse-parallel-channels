#coding:UTF-8

"""
protocol for synaptic depression/recovery
"""

import numpy as np
from .rc import *

INTERNS = [5, 10, 50, 100, 150, 250, 300, 400, 500, 750, 1000, 1250, 1500, 2000]

def get_traces_ppd_v(base, peak, dur, pre_time=1000, pre_v=None, has1st=True, dt=DT):
    """ Traces of voltage stimulus"""
    traces = []
    if pre_v is None:
        pre_v = base
    pre = [pre_v]*int(pre_time/dt)
    plus = [peak]*int(dur/dt)
    for t in INTERNS:
        if has1st:
            trace = np.copy(pre+plus).tolist()
        else:
            trace = np.copy(pre).tolist()
        trace += [base]*int(t/dt)+plus
        trace += [base]*int((INTERNS[-1]-t+200)/dt)
        traces.append(trace)
    return np.array(traces)
