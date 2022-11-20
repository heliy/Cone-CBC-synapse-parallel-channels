#coding:UTF-8

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from lib.utils import get_func_from_parameters
from lib.filters import ContainerFilter
from lib.depression import get_traces_ppd_v, INTERNS
from lib.RAB import generate_RAB_class
from lib.LN import generate_ITER_LN_grabner, LN
from lib.pipeline import OPTResult as OriginOPTResult

"""
Experiments on
Mechanism of High-Frequency Signaling at a Depressing Ribbon Synapse
(Grabner et al., 2016, Neuron)
"""


##################################
#
# generate light protocols
#
##################################

DT = 1
ALL_IDXS = [2, 3]
HZs = [1, 2, 4, 8, 16, 32, 333]

def get_traces(hzs=HZs, pre_time=500, stim_time=5000, tail_time=500, dark=0, light=1, light_ratio=0.5, dt=DT):
    """ the light trace with varied frequencies """
    traces = []
    pre = [dark]*int(pre_time/dt)
    tail = [dark]*int(tail_time/dt)
    n_stim = int(stim_time/dt)
    for hz in hzs:
        n_single = int(1000/(hz*dt))
        n_single_light = int(n_single*light_ratio)
        n_single_dark = n_single-n_single_light
        single = [light]*n_single_light+[dark]*n_single_dark
        n_repeat = int(stim_time*hz)+1
        stim = (single*n_repeat)[:n_stim]
        trace = pre+stim+tail
        traces.append(np.array(trace))
    return traces

LIGHTTRACEs = get_traces()

def get_tuning_range(peaks, threshold=0.05):
    """ the tuning range of given peak responses """
    amps = peaks/peaks.max()
    if amps[-1] == 0:
        return [HZs[np.where(amps>=1-threshold)[0][-1]], HZs[np.where(amps<=threshold)[0][0]]]
    else:
        return [HZs[np.where(amps>=1-threshold)[0][-1]], HZs[-1]]

##################################
#
# get the features of recordings
#
##################################

PEAKS2 = 345.9-np.array([271.6, 275.2, 283.3, 294.4, 302.7, 339.2, 342.4])
PEAKS2 /= PEAKS2.max()
PEAKS3 = 345.9-np.array([303.9, 315.5, 328.5, 337.1, 338.7, 344.3, 345.9])
PEAKS3 /= PEAKS3.max()
PEAKS = {
    2: PEAKS2,
    3: PEAKS3,
}

# PEAK_RATIOS = {
#     2: 0.75,
#     3: 0.36,
# }

PEAK_RATIOS = {
    2: 1.0,
    3: 0.3,
}
BOTTOM_RATIOS = {
    2: 1.42,
    3: 1.14,
}

OFFTAUS_MEAN = {
    2: 72,
    3: 744,
}
OFFTAUS_STD = {
    2: 10,
    3: 22,
}

##################################
#
# Models and fixed parameters
#
##################################

# Data from "Reconstruction of the electrical responses of turtle cones to flashes and steps of light"
# 40ms for maximum light intensity (Fig.8)
LinearTau = 40
# (Peak) Response (Fig.7), ps come from `get_nonlinear_parameters`
NonlinearFunc = get_func_from_parameters("hill", [ 0.54199339,  4.92147578, 26.38135944])

def get_nonlinear_parameters():
    """ the cone menbrane potential data """
    NON_X = np.array([62.8, 99.5, 108.3, 116.4, 125.8, 134.1, 142.7, 151.3, 158.2, 165.6, 175.1, 183.8, 193.0, 201.1, 210.3, 219.2])
    NON_X -= NON_X.min()
    NON_X /= NON_X.max()
    NON_Y = 391.2-np.array([391.2, 386.8, 383.3, 378.0, 364.1, 350.8, 332.5, 311.3, 300.7, 290.3, 277.6, 272.7, 265.1, 261.8, 260.7, 256.3])
    NON_Y -= NON_Y.min()
    NON_Y /= (391.2-257.9)
    NON_Y *= 25
    func = lambda x, K, N, M: M*(x**N)/(x**N+K**N)
    ps, _ = curve_fit(func, NON_X, NON_Y, p0=[0.5, 2, 25])
    return NON_X, NON_Y, ps

RIBBONCLASS = generate_RAB_class("1minhill", "stable_response_func", "pow")

class LNS:
    def __init__(self, parameters, temporal_tau=LinearTau, conenonlinear=NonlinearFunc, hasBCnonlinear=True, u_resting=0.05, dt=DT):
        self.dt = dt
        self.name = "LNS"
        index = 0

        self.linear = ContainerFilter(lambda x: temporal_tau, dt=dt)
        self.nonlinear = conenonlinear

        self.ca_func = lambda x: 0.078*(41-x)/(1+np.exp((-39.3-x)/4.88))+u_resting
        self.index_ribbon = index
        self.ribbon = RIBBONCLASS(parameters[self.index_ribbon:self.index_ribbon+6], u_resting=u_resting, u_max=5)
        index += 6

        if hasBCnonlinear:
            self.index_bc = index
            if parameters[self.index_bc+1] < 0:
                raise Exception()
            self.bcnonlinear = get_func_from_parameters("sigmoid", list(parameters[self.index_bc:self.index_bc+2])+[1.])

    def init(self, stim, is_voltage=False):
        if not is_voltage:
            self.linear.init(stim)
        if not is_voltage:
            stim = -self.nonlinear(stim)-45
        stim = self.ca_func(stim)
        self.ribbon.init(stim, is_stim=True)

    def run(self, trace, with_init=True, is_voltage=False):
        if with_init:
            self.init(trace[0], is_voltage=is_voltage)
        rs = [trace]
        if not is_voltage:
            rs = [self.linear.filter(rs[0])]+rs
        if not is_voltage:
            rs = [self.nonlinear(rs[0])]+rs
            rs = [-rs[0]-45]+rs
        rs = [self.ca_func(rs[0])]+rs
        rs = self.ribbon.run(rs[0], with_init=False, return_resupply=False)+rs
        if hasattr(self, "bcnonlinear"):
            rs = [self.bcnonlinear(rs[0])]+rs
        return rs[:-1]


################################################
#
# analysis response features and evaluate model
#
################################################

def get_ratio(model, TRACE=LIGHTTRACEs[0], resp_index=0):
    """ get peak/base ratio of a model """
    r = model.run(TRACE, with_init=True)[resp_index]
    A = r[int(490/model.dt)]
    B = r[-int(1010/model.dt)]
    C = r[-int(1010/model.dt):-int(500/model.dt)].max()
    return (C-A)/(A-B)

def ana_peaks(rs, hzs=HZs, func=max, dt=DT):
    """ get peaks if response traces """
    peaks = []
    for resp, hz in zip(rs, hzs):
        resting_index = -int(2000/dt)
        end_index = -int(1000/dt)
        peak = max(func(resp[resting_index:end_index])-resp[int(490/dt)], 0)
        peaks.append(peak)
    peaks = np.array(peaks)
    return peaks

def evaluate_model(model, cell_idx=2, **ps):
    assert model.dt == DT
    traces = LIGHTTRACEs
    rs = [model.run(i, with_init=True)[0] for i in traces]
    peaks = ana_peaks(rs, dt=model.dt)
    if peaks.max() < 1e-4:
        raise Exception()
    peaks /= peaks.max()
    error = np.abs(peaks-PEAKS[cell_idx]).sum() # /len(peaks)
    # do not norm, as PEAKS[-1] == 0
    # and all features are ratios (same scale)
    
    peak_ratio = get_ratio(model, TRACE=traces[0])
    error += np.abs(peak_ratio-PEAK_RATIOS[cell_idx])
    return error

##################################
#
# simulation results
#
##################################

def generate_ITER_CellModel(idx, **ps):
    if idx == 2:
        x0 = [0.12335551196749212, 0.21146292963728885, 0.9100002315668032, 0.09494682753240102, 3.077081229031924e-05, 181.4468141504375, 2.961595963791545, 4.312653167949996]
    elif idx == 3:
        x0 = [0.0001895623361493186, 0.47445354559020714, 1.3456074899361468, 0.7501713091835653, 0.0003395368660982531, 689.6572728654386, -7.390058896953825, 2.9901614575950783]
    else:
        raise Exception()
    def iter_func():
        yield LNS, x0, {}, "LNFix"
    return [iter_func]

class OPTResult(OriginOPTResult):
    def __init__(self, best_n=10, root_dirs=["parameters/grabner/"], cell_idxs = ALL_IDXS, dt=DT):
        generate_ITER_FUNCs = [
            generate_ITER_CellModel,
            generate_ITER_LN_grabner,
        ]
        iter_pss = [
            {},
            {},
        ]
        OriginOPTResult.__init__(self, best_n=best_n, root_dirs=root_dirs, cell_idxs=cell_idxs, 
            generate_ITER_FUNCs=generate_ITER_FUNCs, iter_pss=iter_pss, dt=dt)

    def plot_model_synps(self, models, colors, labels):
        voltages_taus = np.linspace(-70, -30, 100)
        voltages_infis = np.linspace(-70, -40, 100)
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        for model, color, label in zip(models, colors, labels):
            assert isinstance(model, LNS)
            cas_taus = model.ca_func(voltages_taus)
            cas_infis = model.ca_func(voltages_infis)
            taus = model.ribbon.get_active_tau(cas_taus)
            infis = model.ribbon.get_active_infi(cas_infis)
            ax1.plot(voltages_taus, taus, color=color, label=label)
            ax2.plot(voltages_infis, infis, color=color, label=label)
        plt.legend()
        plt.show()

    def get_scaler(self, cell_idx, model):
        trace = LIGHTTRACEs[0]
        r = model.run(trace, with_init=True)[0]
        amp = r.max()-r[0]
        if cell_idx == 2:
            return 15/amp # maximum delta mV
        elif cell_idx == 3:
            return 8.63/amp 

    def get_model_range(self, model):
        if type(model) == int:
            return get_tuning_range(PEAKS[model])
        rs = [model.run(i, with_init=True)[0] for i in LIGHTTRACEs]
        peaks = ana_peaks(rs, dt=model.dt)
        return get_tuning_range(peaks)

    def plot_ln_parameters(self, m2, m3):
        assert isinstance(m2, LN)
        assert isinstance(m3, LN)
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(m2.linear.weights[::-1][:500], color="lightgreen", label="LN - cb2")
        ax1.plot(m3.linear.weights[::-1][:500], color="green", label="LN - cb3a")
        ax1.legend()
        ax1.set_ylabel("weights")
        ax1.set_xlabel("Time (ms)")
        ax2 = plt.subplot(1, 2, 2)
        linears2 = np.array([m2.run(i, with_init=True)[1] for i in LIGHTTRACEs]).flatten()
        xs2 = np.linspace(linears2.min(), linears2.max(), 100)
        ax2.plot(xs2, (m2.nonlinear(xs2)-m2.nonlinear(0))*self.get_scaler(2, m2), color="lightgreen", label="LN - cb2")
        linears3 = np.array([m3.run(i, with_init=True)[1] for i in LIGHTTRACEs]).flatten()
        xs3 = np.linspace(linears3.min(), linears3.max(), 100)
        ax2.plot(xs3, (m3.nonlinear(xs2)-m3.nonlinear(0))*self.get_scaler(3, m3), color="green", label="LN - cb3a")
        ax2.legend()
        ax2.set_ylabel("response (mV)")
        ax2.set_xlabel("weights")
        plt.show()

    def plot_lns_parameters(self, m2, m3):
        assert isinstance(m2, LNS)
        assert isinstance(m3, LNS)
        voltages = np.linspace(-70, -30, 100)
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(voltages, m2.ribbon.get_active_tau(m2.ca_func(voltages)), color="pink", label="LNS - cb2")
        ax1.plot(voltages, m3.ribbon.get_active_tau(m3.ca_func(voltages)), color="red", label="LNS - cb3a")
        ax1.legend()
        ax1.set_ylabel("Voltage (mV)")
        ax1.set_xlabel("Time Contant (ms)")
        ax2 = plt.subplot(1, 2, 2)
        ax2.plot(voltages, m2.ribbon.get_active_infi(m2.ca_func(voltages)), color="pink", label="LNS - cb2")
        ax2.plot(voltages, m3.ribbon.get_active_infi(m3.ca_func(voltages)), color="red", label="LNS - cb3a")
        ax2.legend()
        ax2.set_ylabel("Stable State")
        ax2.set_xlabel("Time Contant (ms)")
        plt.show()

    def plot_model_response_peaks(self, cell_idxs, models, names, alphas, colors, resp_index=0, 
            NORM=True, traces=LIGHTTRACEs, LOG=True):
        """ draw response traces """
        ax = plt.subplot()
        data = []
        for cell_idx, model in zip(cell_idxs, models):
            if cell_idx is None:
                rs = [model.run(i, with_init=True)[resp_index] for i in traces]
            else:
                scale = self.get_scaler(cell_idx, model)
                rs = [model.run(i, with_init=True)[resp_index]*scale for i in traces]
            rs = [i-i[0] for i in rs] # remove resting
            peaks = ana_peaks(rs)
            if NORM:
                peaks /= peaks.max()
            data.append(peaks)
        if NORM:
            ax.set_ylabel("Normalized Response")
        else:
            ax.set_ylabel("Voltage Response (mV)")
        for alpha, name, cell_idx, peaks, color in zip(alphas, names, cell_idxs, data, colors):
            ax.plot(HZs[:len(peaks)], peaks, color=color, label="%s - Model"%name, alpha=alpha)
            if cell_idx is not None:
                ax.scatter(HZs[:len(PEAKS[cell_idx])], PEAKS[cell_idx], color="k", label="%s - Data"%name, alpha=alpha)
        if LOG:
            ax.set_xscale('log')
        ax.set_ylim([-0.1, max(peaks)*1.1])
        ax.set_xlabel("Frequency (Hz)")
        plt.legend()
        plt.show()

    def plot_model_response_traces(self, cell_idxs, models, names, colors, axisoff=True, resp_index=0,
            traces=LIGHTTRACEs):
        """ draw response traces """
        for index, cell_idx, model, name, color in zip(range(len(models)), cell_idxs, models, names, colors):
            if cell_idx is None:
                rs = [model.run(i, with_init=True)[resp_index] for i in traces]
            else:
                scale = self.get_scaler(cell_idx, model)
                rs = [model.run(i, with_init=True)[resp_index]*scale for i in traces]
            rs = [i-i[0] for i in rs] # remove resting
            N = len(rs)
            ax = plt.subplot(1, len(models), index+1)
            y_min = min(min(i) for i in rs)
            y_max = max(max(i) for i in rs)
            if y_min < 0:
                y_min *= 1.1
            else:
                y_min /= 1.1
            if y_max < 0:
                y_max /= 1.1
            else:
                y_max *= 1.1
            for i, resp in enumerate(rs):
                ax = plt.subplot(N, len(models), (N-i-1)*len(models)+index+1)
                ax.plot(resp, color=color)
                ax.set_ylim([y_min, y_max])
                if axisoff:
                    ax.axis("off")
                else:
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    if i > 0:
                        ax.get_xaxis().set_visible(False)
                if i == N-1:
                    ax.set_title(name)
                if i == 0:
                    ax.set_xlabel("Time (ms)")
        plt.show()

    def plot_model_response_ranges(self, models, names, yvalues, colors, alphas, ywidth):
        """ draw response traces """
        ax = plt.subplot()
        for model, alpha, yvalue, color in zip(models, alphas, yvalues, colors):
            tunings = self.get_model_range(model)
            box = Rectangle((tunings[0], yvalue-ywidth/2), tunings[1]-tunings[0], ywidth)
            ax.add_collection(PatchCollection([box], facecolor=color, alpha=alpha))
        ax.set_yticks(yvalues)
        ax.set_yticklabels(names)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xscale('log')
        ax.set_ylim([min(yvalues)-ywidth, max(yvalues)+ywidth])
        ax.set_xlabel("Frequency (Hz)")
        plt.show()

    def do_PPD(self, models, colors, labels, base_v=-70, dur=15, peak_v=-30, draw=True):
        traces_vm = get_traces_ppd_v(base=base_v, peak=peak_v, dur=dur, dt=self.dt)
        peaks = []
        for model, color, label in zip(models, colors, labels):
            assert isinstance(model, LNS)
            traces_ca = [model.ca_func(i) for i in traces_vm]
            rs = [model.ribbon.run(i, with_init=True) for i in traces_ca]
            if draw:
                ax = plt.subplot()
                for i, response in enumerate(rs):
                    ax.plot(response[0], color=color)
                ax.set_ylabel("Synaptic Response")
                ax.set_xlabel("Time (ms)")
                ax.set_title("Response of %s"%label)
                plt.show()
            peak = [r[0][int((1000+dur+INTERNS[0])/self.dt):].max() for r in rs]
            peak = np.array(peak)
            peak -= rs[0][0][int(990/self.dt)]
            peak_1sts = [r[0][int(1000/self.dt):int((1000+dur+INTERNS[0])/self.dt)].max() for r in rs]
            peak_1sts = np.array(peak_1sts)
            peak_1sts -= rs[0][0][int(990/self.dt)]
            peak /= peak_1sts
            peaks.append(peak)
        func = lambda x, FLU, TAU, BASE: FLU*(1-np.exp(-(x)/TAU))+BASE
        strengths = []
        taus = []
        if draw:
            ax = plt.subplot()
        for peak, color, label in zip(peaks, colors, labels):
            i_begin = np.argsort(peak)[0]
            X = INTERNS[i_begin:]
            Y = peak[i_begin:]
            if len(Y) <= 1:
                ratio = 1-Y[0]
                tau = 0.
            else:
                popt, _ = curve_fit(func, X, Y, p0=[1-min(Y), 100, min(Y)], maxfev=30000)
                ratio, tau = tuple(popt[:2])
            strengths.append((1-min(peak))*100)
            taus.append(tau)
            if draw:
                ax.scatter(INTERNS, peak, color=color)
                ax.plot(X, np.vectorize(func)(X, *popt), color=color, label="%s Tau=%d(ms)"%(label, int(tau)))
        if draw:
            ax.set_ylim([0, 1.1])
            ax.legend()
            plt.show()
        else:
            return strengths, taus

    def plot_intern_values(self, model, indices, labels, colors, from_time, to_time, overlayed={}, T=False):
        res = [model.run(i) for i in LIGHTTRACEs]
        for i in range(len(HZs)):
            for j, index, label, color in zip(range(len(indices)), indices, labels, colors):
                if T:
                    ax = plt.subplot(len(indices), len(HZs), j*len(HZs)+i+1)
                else:
                    ax = plt.subplot(len(HZs), len(indices), (len(HZs)-i-1)*len(indices)+j+1)
                miny = min(r[index].min() for r in res)
                maxy = max(r[index].max() for r in res)
                if index in overlayed:
                    ax.plot(res[i][overlayed[index]][from_time: to_time], color="gray")
                    miny = min(miny, min(r[overlayed[index]].min() for r in res))
                    maxy = max(maxy, max(r[overlayed[index]].max() for r in res))
                ax.plot(res[i][index][from_time:to_time], color=color)
                if miny < 0:
                    miny *= 1.1
                else:
                    miny /= 1.1
                if maxy < 0:
                    maxy /= 1.1
                else:
                    maxy *= 1.1
                ax.set_ylim(miny, maxy)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                if T:
                    if j == 0:
                        ax.set_title("%d Hz"%HZs[i])
                    if j < len(indices)-1:
                        ax.get_xaxis().set_visible(False)
                    if i == 0:
                        ax.set_ylabel(label)
                else:
                    if i == len(HZs)-1:
                        ax.set_title(label)
                    if i > 0:
                        ax.get_xaxis().set_visible(False)
                    if j == 0:
                        ax.set_ylabel("%d Hz"%HZs[i])
        plt.show()

    def plot_intern_values_peak(self, model, indices, labels, colors, styles, depressed=[]):
        res = [model.run(i) for i in LIGHTTRACEs]
        ax = plt.subplot()
        for index, label, color, style in zip(indices, labels, colors, styles):
            if index in depressed:
                peaks = np.array([r[index][0]-r[index].min() for r in res])
            else:
                peaks = ana_peaks([r[index] for r in res])
            peaks -= peaks.min()
            peaks /= peaks.max()
            ax.plot(HZs, peaks, color=color, label=label, linestyle=style)
        ax.set_xlabel("Frequency (hz)")
        ax.set_ylabel("Normalized Peak Value")
        ax.set_xscale('log')
        plt.legend()
        plt.show()

    def plot_intern_values_multi(self, model, indices, labels, stim, scales=None):
        return super().plot_intern_values(model, indices, labels, stim, scales)


if __name__ == "__main__":
    print("Experiments on (Grabner et al., 2016)")
    print("Load Optimized Models")
    optres = OPTResult()

    print()
    print("##################################################")
    print("### The performance of LN models # Figure 1")
    print("##################################################")
    ln_cb2 = optres.models[2]["LN_cif"]["MODEL"][0]
    ln_cb3 = optres.models[3]["LN_cif"]["MODEL"][0]
    print("1. Response Traces # Figure 1A")
    optres.plot_model_response_traces(cell_idxs=[2, 3], models=[ln_cb2, ln_cb3], names=["LN - CB2", "LN - CB3a"], colors=["lightgreen", "green"], axisoff=False)
    print("2. Peak Responses # Figure 1B")
    optres.plot_model_response_peaks(cell_idxs=[2, 3], models=[ln_cb2, ln_cb3], names=["LN - CB2", "LN - CB3a"], colors=["lightgreen", "green"], alphas=[1, 0.5])
    print("3. Tuning ranges # Figure 1D")
    optres.plot_model_response_ranges(models=[2, ln_cb2, 3, ln_cb3], names=["cb2", "LN", "cb3a", "LN"], yvalues=[10, 9, 7, 6], 
        colors=["k", "lightgreen", "k", "green"], alphas=[1, 1, 1, 1], ywidth=0.6)
    print("4. Parameters in LN models # Figure 1C")
    optres.plot_ln_parameters(ln_cb2, ln_cb3)

    print()
    print("##################################################")
    print("### The performance of LNS models # Figure 2")
    print("##################################################")
    lns_cb2 = optres.models[2]["LNS_LNFix"]["MODEL"][0]
    lns_cb3 = optres.models[3]["LNS_LNFix"]["MODEL"][0]
    print("1. Response Traces # Figure 2B")
    optres.plot_model_response_traces(cell_idxs=[2, 3], models=[lns_cb2, lns_cb3], names=["LNS - CB2", "LNS - CB3a"], colors=["pink", "red"], axisoff=False)
    print("2. Peak Responses # Figure 2C")
    optres.plot_model_response_peaks(cell_idxs=[2, 3], models=[lns_cb2, lns_cb3], names=["LNS - CB2", "LNS - CB3a"], colors=["pink", "red"], alphas=[1, 0.5])
    print("3. Tuning ranges # Figure 2D")
    optres.plot_model_response_ranges(models=[2, lns_cb2, ln_cb2, 3, lns_cb3, ln_cb3], names=["cb2", "LNS", "LN", "cb3a", "LNS", "LN"], yvalues=[10, 9, 8, 6, 5, 4], 
        colors=["k", "pink", "lightgreen", "k", "red", "green"], alphas=[1]*6, ywidth=0.6)

    print()
    print("##################################################")
    print("### The synaptic depression experiment on LNS models # Figure 3")
    print("##################################################")
    print("1. Parameters of LNS models # Figure 3B")
    optres.plot_lns_parameters(lns_cb2, lns_cb3)
    print("2. The PPD experment on two LNS models # Figure 3C&D")
    optres.do_PPD(models=[lns_cb2, lns_cb3], colors=["pink", "red"], labels=["LNS - CB2", "LNS - CB3a"])

    print()
    print("##################################################")
    print("### Kinetics inside LNS models # Figure 4")
    print("##################################################")
    print("1. The inner variables of the cb2 LNS model # Figure 4A")
    optres.plot_intern_values(lns_cb2, indices=[-3, 2, 1], labels=["Cone Vm", "A", "EPSC"], colors=["k", "gold", "purple"], from_time=0, to_time=3000, T=False)
    print("2. The depression of inner variables of the cb2 LNS model # Figure 4B")
    optres.plot_intern_values_peak(lns_cb2, indices=[-3, 3, 2, 1, 0], labels=["Cone Depressed Voltage", "A if fully restored", "A", "cb2 EPSC", "cb2 Voltage"], colors=["k", "gold", "gold", "purple", "red"], styles=["solid", "dashed", "solid", "solid", "solid"], depressed=[-3])
    print("3. The inner variables of the cb2 LNS model with time constants # Figure 4C")
    optres.plot_intern_values(lns_cb2, indices=[-3, 4, 2], labels=["Cone Vm", "tau", "A"], colors=["k", "blue", "gold"], from_time=0, to_time=3000, overlayed={2: 3}, T=True)
    print("4. The inner variables of the cb3a LNS model with time constants # Figure 4D")
    optres.plot_intern_values(lns_cb3, indices=[-3, 4, 2], labels=["Cone Vm", "tau", "A"], colors=["k", "blue", "gold"], from_time=0, to_time=3000, overlayed={2: 3}, T=True)

