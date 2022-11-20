#coding:UTF-8

"""
the Ribbon Adaptive Block module
"""

import numpy as np
import matplotlib.pyplot as plt

from .rc import *
from .utils import get_func_from_parameters

class RAB:
    def __init__(self, active_infi_func, ext_func_name, ext_func, N, u_resting=U_REST, u_max=U_MAX, dt=DT):
        assert dt == 1
        self.active_infi_func = np.vectorize(active_infi_func)
        self.ext_func_name = ext_func_name
        self.ext_func = np.vectorize(ext_func)
        self.N = N
        self.u_resting = u_resting
        self.u_max = u_max
        self.dt = dt
        self.a = 1

    def input_filter(self, inputs):
        """ ensuring the input is within the [u_resting, u_max] """
        inputs = np.array([inputs]).flatten()
        inputs[inputs < self.u_resting] = self.u_resting
        inputs[inputs > self.u_max] = self.u_max
        return inputs

    def get_active_infi(self, inputs):
        """ get steady states of the active state """
        return self.active_infi_func(self.input_filter(inputs))

    def get_active_tau(self, inputs):
        """ get time constants """
        inputs = self.input_filter(inputs)
        active_infis = self.active_infi_func(inputs)
        if self.ext_func_name == STABLE_FUNC_NAME:
            stables = self.ext_func(inputs)
            taus = self.N*active_infis*(1-active_infis)/stables
        elif self.ext_func_name == ALPHA_FUNC_NAME:
            alpha_ps = self.ext_func(inputs)
            taus = (1-active_infis)/alpha_ps
        else:
            raise Exception()
        if len(inputs) > 1:
            return taus
        else:
            return taus[0]

    def get_p_alpha(self, inputs):
        """ get alpha transient rate constant """
        inputs = self.input_filter(inputs)
        if self.ext_func_name == STABLE_FUNC_NAME:
            active_infis = self.active_infi_func(inputs)
            stables = self.ext_func(inputs)
            active_taus = self.N*active_infis*(1-active_infis)/stables
            ps_alpha = (1-active_infis)/active_taus
        elif self.ext_func_name == ALPHA_FUNC_NAME:
            ps_alpha = self.ext_func(inputs)
        else:
            raise Exception()
        if len(inputs) > 1:
            return ps_alpha
        else:
            return ps_alpha[0]

    def get_p_beta(self, inputs):
        """ get beta transient rate constant """
        inputs = self.input_filter(inputs)
        if self.ext_func_name == STABLE_FUNC_NAME:
            active_infis = self.active_infi_func(inputs)
            stables = self.ext_func(inputs)
            active_taus = self.N*active_infis*(1-active_infis)/stables
            ps_beta = active_infis/active_taus
        elif self.ext_func_name == ALPHA_FUNC_NAME:
            active_infis = self.active_infi_func(inputs)
            ps_alpha = self.ext_func(inputs)
            ps_beta = ps_alpha*active_infis/(1-active_infis)
        else:
            raise Exception()
        if len(inputs) > 1:
            return ps_beta
        else:
            return ps_beta[0]

    def get_stable_response(self, inputs):
        """ get stable responses """
        inputs = self.input_filter(inputs)
        if self.ext_func_name == STABLE_FUNC_NAME:
            stables = self.ext_func(inputs)
        elif self.ext_func_name == ALPHA_FUNC_NAME:
            active_infis = self.active_infi_func(inputs)
            ps_alpha = self.ext_func(inputs)
            stables = ps_alpha*active_infis*self.N
        else:
            raise Exception()
        if len(inputs) > 1:
            return stables
        else:
            return stables[0]

    def init(self, init, is_stim=True):
        """ initialize the inner active state """
        if is_stim:
            init = max(init, self.u_resting)
            init = min(init, self.u_max)
            self.a = self.active_infi_func(init)
            return self.a*self.N*self.get_p_alpha(init)
        else: # set active state
            self.a = init
            return self.a

    def run(self, inputs, with_init=True, return_resupply=False):
        """ simulation, return responses and innner variables """
        inputs = self.input_filter(inputs)
        if with_init:
            self.init(inputs[0], is_stim=True)
        active_infis = self.active_infi_func(inputs)
        if self.ext_func_name == STABLE_FUNC_NAME:
            stables = self.ext_func(inputs)
            active_taus = self.N*active_infis*(1-active_infis)/stables
            ps_alpha = (1-active_infis)/active_taus
        elif self.ext_func_name == ALPHA_FUNC_NAME:
            ps_alpha = self.ext_func(inputs)
            active_taus = (1-active_infis)/ps_alpha
        else:
            raise Exception()
        outputs = np.zeros(active_infis.shape)
        actives = np.zeros(active_infis.shape)
        for i, infi in enumerate(active_infis):
            response = ps_alpha[i]*self.a*self.N
            tau = active_taus[i]
            self.a = infi+(self.a-infi)*np.exp(-self.dt/tau)
            outputs[i] = response
            actives[i] = self.a
        if return_resupply:
            ps_beta = ps_alpha*active_infis/(1-active_infis)
            resupplies = ps_beta*(1-actives)*self.N
            return [outputs, resupplies, actives, active_infis, active_taus, ps_alpha, ps_beta]
        else:
            return [outputs, actives, active_infis, active_taus]

    def draw_rates(self, u_min=None, u_max=None, LOG=True):
        if u_min is None:
            u_min = self.u_resting
        if u_max is None:
            u_max = self.u_max
        inputs = np.linspace(u_min, u_max, 1000)
        ps_alpha = self.get_p_alpha(inputs)
        ps_beta = self.get_p_beta(inputs)

        ax = plt.subplot()
        ax.plot(inputs, ps_alpha, color='r', label="alpha")
        ax.plot(inputs, ps_beta, color='b', label='beta')
        plt.legend()
        if LOG:
            ax.set_yscale('log')
        ax.set_ylabel("Rate (/ms)")
        ax.set_xlabel(u"[Ca++] (μM)")
        plt.show()

    def draw_parameters(self, u_min=None, u_max=None):
        if u_min is None:
            u_min = self.u_resting
        if u_max is None:
            u_max = self.u_max
        inputs = np.linspace(u_min, u_max, 1000)
        active_infis = self.get_active_infi(inputs)
        active_taus = self.get_active_tau(inputs)

        ax = plt.subplot()
        ax2 = ax.twinx()
        ax.plot(inputs, active_infis*100, color='r', label="Stable active (%)")
        ax2.plot(inputs, active_taus, color='b', label='Adaptation Tau (ms)')
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles1+handles2, labels1+labels2)
        ax.set_ylabel("Stable active (%)", color='r')
        ax2.set_ylabel("Adaptation tau (ms)", color='b')
        ax.set_xlabel(u"[Ca++] (μM)")
        plt.show()

    def draw_parameters_voltages(self, cafunc, vm_min=-70, vm_max=-10):
        vms = np.linspace(vm_min, vm_max, 100)
        inputs = cafunc(vms)
        active_infis = self.get_active_infi(inputs)
        active_taus = self.get_active_tau(inputs)

        ax = plt.subplot()
        ax2 = ax.twinx()
        ax.plot(vms, active_infis*100, color='r', label="Stable active (%)")
        ax2.plot(vms, active_taus, color='b', label='Adaptation Tau (ms)')
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles1+handles2, labels1+labels2)
        ax.set_ylabel("Stable active (%)", color='r')
        ax2.set_ylabel("Adaptation tau (ms)", color='b')
        ax.set_xlabel(u"Voltage (mV)")
        plt.show()

    def get_alpha_slopes(self, inputs):
        current_values = self.get_p_alpha(inputs)
        delta_values = self.get_p_alpha(inputs+0.001)-current_values
        slopes = delta_values/0.001
        return slopes
        
    def estimate_gain_trace(self, trace, with_init=True):
        trace = np.array(trace)
        slopes = self.get_alpha_slopes(trace)
        current_states = self.run(trace, with_init=with_init)[1]
        gains = slopes*current_states*self.N
        return gains

    def estimate_gain_trace_pulse(self, trace, test_timings, pulse=0.01, pulse_time=10, peak=True):
        r_origin = self.run(trace, with_init=True)[0]
        r_resting = r_origin[0]
        r_scale = max(r_origin-r_origin[0])
        r_origin_norm = (r_origin-r_resting)/r_scale

        resps = []
        for test_timing in test_timings:
            new_trace = np.copy(trace)
            new_trace[test_timing:test_timing+pulse_time] += pulse
            r_new = (self.run(new_trace)[0]-r_resting)/r_scale
            r_diff = r_new-r_origin_norm
            if peak:
                resp = r_diff.max()
            else:
                # sum the abs differences between two curves
                resp = np.abs(r_diff).sum()
            resps.append(resp)
        return np.array(resps)


def generate_RAB_class(active_infi_func_type, ext_func_name, ext_func_type):
    class SpecificRAB(RAB):
        def __init__(self, parameters, **ext_parameters):
            self.origin_ps = parameters
            self.origin_ext = ext_parameters

            if "N" in ext_parameters and ext_parameters["N"]:
                self.ps = np.copy(parameters)
            else:
                ext_parameters["N"] = parameters[-1]
                self.ps = np.copy(parameters[:-1])
            active_infi_func = get_func_from_parameters(active_infi_func_type, 
                                                self.ps[:FUNC_PARAMS_N[active_infi_func_type]])
            ext_func = get_func_from_parameters(ext_func_type, 
                                                self.ps[-FUNC_PARAMS_N[ext_func_type]:])
            self.ext_parameters = ext_parameters
            self.name = "%s%s%sRibbon"%(FUNC_NAMES[ext_func_type], EXT_NAMES[ext_func_name], FUNC_NAMES[active_infi_func_type])
            
            RAB.__init__(self, active_infi_func, ext_func_name, ext_func,  **ext_parameters)

    return SpecificRAB
