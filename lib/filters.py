#coding:UTF-8

"""
Linear Temporal Filters in Simulations
"""

import numpy as np

from .rc import DT

class Container:
    def __init__(self, variables, step_func):
        self.variables = variables
        self.step_func = step_func

    def step(self, **parameters):
        self.variables = self.step_func(self.variables, parameters)

    def get_variable(self, name):
        return self.variables[name]

    def set_variable(self, name, value):
        self.variables[name] = value
        
class ContainerFilter:
    def __init__(self, tau_func, output_func=lambda x:x, dt=DT):
        self.dt = dt
        self.variables = {
            "v": 0,
        }
        self.tau_func = tau_func
        def step(variables, parameters):
            d = -1*self.variables['v']+output_func(parameters["s"])
            variables["v"] += self.dt*d/max(float(tau_func(parameters["s"])), self.dt)
            return variables
        self.container = Container(self.variables, step)

    def f(self, s):
        self.container.step(s=s)
        return self.variables['v']
    
    def init(self, value):
        self.variables["v"] = value

    def filter(self, trace):
        return np.array([self.f(float(i)) for i in trace])

class TemporalFilter:
    def __init__(self, weights):
        """ 
        last weight in weights is the latest input 
        """
        weights = np.array(weights)
        self.weights = weights
        self.buffer = np.zeros(self.weights.shape)

    def filter(self, trace):
        trace = np.array(trace)
        assert len(trace.shape) == 1, "Need 1-D Temporal Stimulus"
        trace = np.array(trace)
        def f(x):
            self.buffer[:-1] = self.buffer[1:]
            self.buffer[-1] = x
            return (self.buffer*self.weights).sum()
        return np.vectorize(f)(trace)

    def init(self, stim):
        self.buffer[:] = stim

class FunctionTemporalFilter(TemporalFilter):
    def __init__(self, func, step, dt=DT, normalize_weight=False):
        step = int(step/dt)
        weights = np.zeros(step)
        for i in range(step):
            weights[i] = func(i*dt)
        if normalize_weight:
            weights /= abs(weights.sum())
        TemporalFilter.__init__(self, weights[::-1])

class ConeImpluseFilter(FunctionTemporalFilter):
    def __init__(self, gamma=1, tau_rise=70, tau_decay=70, tau_phase=100, phi=-np.pi/5, A=1e3, resting_stim=0.5, dt=DT):
        func = lambda x: -((x/(gamma*tau_rise))**3)*np.exp(-(x/(gamma*tau_decay))**2)*np.cos(0.002*np.pi*x/(gamma*phi)+tau_phase)/(1+x/(gamma*tau_rise))
        step = 5000
        self.A = A
        self.resting_stim = resting_stim
        FunctionTemporalFilter.__init__(self, func, step, dt=dt, normalize_weight=True)

    def filter(self, trace):
        resp = FunctionTemporalFilter.filter(self, trace)
        # resp += self.resting_stim
        resp *= self.A
        return resp