#coding:UTF-8

import os

import numpy as np
import matplotlib.pyplot as plt

from lib.filters import ContainerFilter
from Grabner import LNS as GLNS

"""
Experiments on
Distinct synaptic transfer functions in same-type photoreceptors
(Schroder et al., 2021, eLife)
"""

DT = 1
DATA_DIR = "datas/Schroder/"
PLOT_SCALE = 1/9.8
GLU_TEMPORAL = ContainerFilter(lambda x: 60, dt=DT)


def load_data(type_label="AZ"):
    # load data for optimize and testing
    cas = np.loadtxt(os.path.join(DATA_DIR, "cas_%s.txt"%type_label))
    resp = np.loadtxt(os.path.join(DATA_DIR, "resp_%s.txt"%type_label))
    return cas, resp

class LNS(GLNS):
    def __init__(self, parameters, dt=DT):
        GLNS.__init__(self, parameters=parameters[1:], temporal_tau=parameters[0], hasBCnonlinear=False, u_resting=0.05, dt=dt)

def get_stim_light(repeat_n=5):
    single = np.array([0]*1000+[1]*1000).tolist()
    trace = [0.5]*1000+single*repeat_n
    trace = np.array(trace)
    return trace

def evaluate_model_light(model, type_label="N", repeat_n=5, glu_temporal=GLU_TEMPORAL):
    _, resp = load_data(type_label=type_label)
    trace = get_stim_light(repeat_n=repeat_n)
    r = model.run(trace, with_init=True)[0]
    glu_temporal.init(r[0])
    r = glu_temporal.filter(r)[-2000:]
    error = np.abs(r-resp).sum()
    return error

""" optimized models for zones """
LNS_AZ = LNS([21.080219591082535, 0.40074838240003763, 0.5326845935216658, 4.017144435410888, 1.2043788732089387, 0.7150148942275466, 3589.2783009246987])
LNS_D = LNS([31.78434814958831, 0.07065499004994902, 0.46552018179356536, 0.5669353742184673, 4.3516908465551865, 1.1110286853752962, 4445.790302493069])
LNS_N = LNS([33.49170490202222, 0.07071235370903231, 0.2511129197991299, 0.23256649717137395, 5.690319759738195, 0.8981904663962155, 769.9373636610474])

COLORS = {
    "AZ": "red",
    "D": "blue",
    "N": "green",
}

def plot_optimized_traces(repeat_n=5):
    trace = get_stim_light(repeat_n=repeat_n)
    def plot_sub(model, type_label, ax):
        _, resp = load_data(type_label=type_label)
        r = model.run(trace, with_init=True)[0]
        GLU_TEMPORAL.init(r[0])
        r = GLU_TEMPORAL.filter(r)[-2000:]
        ax.plot(resp*PLOT_SCALE, label="%s - Data"%type_label, color="k")
        ax.plot(r[-2000:]*PLOT_SCALE, label="%s - Model"%type_label, color=COLORS[type_label])
        ax.set_ylim([0, 1.1])
        ax.set_xlabel("Time (ms)")
        ax.legend()
    
    ax1 = plt.subplot(1, 3, 1)
    plot_sub(LNS_AZ, "AZ", ax1)
    ax2 = plt.subplot(1, 3, 2)
    plot_sub(LNS_D, "D", ax2)
    ax3 = plt.subplot(1, 3, 3)
    plot_sub(LNS_N, "N", ax3)
    plt.show()

def plot_optimized_parameters(vm_min=-70, vm_max=-45):
    vms = np.linspace(vm_min, vm_max, 100)
    cas = LNS_AZ.ca_func(vms)

    ax1 = plt.subplot(121)
    ax1.plot(vms, LNS_AZ.ribbon.get_active_infi(cas), label="AZ", color=COLORS["AZ"])
    ax1.plot(vms, LNS_D.ribbon.get_active_infi(cas), label="D", color=COLORS["D"])
    ax1.plot(vms, LNS_N.ribbon.get_active_infi(cas), label="N", color=COLORS["N"])
    ax1.legend()
    ax1.set_xlabel("Cone Membreane Voltage (mV)")
    ax1.set_ylabel("tau (ms)")

    ax2 = plt.subplot(122)
    ax2.plot(vms, LNS_AZ.ribbon.get_active_infi(cas), label="AZ", color=COLORS["AZ"])
    ax2.plot(vms, LNS_D.ribbon.get_active_infi(cas), label="D", color=COLORS["D"])
    ax2.plot(vms, LNS_N.ribbon.get_active_infi(cas), label="N", color=COLORS["N"])
    ax2.legend()
    ax2.set_xlabel("Cone Membreane Voltage (mV)")
    ax2.set_ylabel("stable state")

    plt.show()


if __name__ == "__main__":
    print("Experiments on (Schroder et al., 2021)")
    print()
    print("##################################################")
    print("### Data fitting and parameters of LNS models # Figure 5")
    print("##################################################")
    print("1. Response Traces # Figure 5D")
    plot_optimized_traces()
    print("2. Parameters # Figure 5E")
    plot_optimized_parameters()

    print()
    print("##################################################")
    print("### Prediction of temporal tuning # Figure 6")
    print("##################################################")
    print("1. Filtering responses # Figure 6A")
    from Grabner import OPTResult
    optres = OPTResult()
    optres.plot_model_response_traces([None, None, None], models=[LNS_AZ, LNS_D, LNS_N], names=["AZ", "D", "N"], colors=[COLORS["AZ"], COLORS["D"], COLORS["N"]], axisoff=False)
    print("2. Peak responses # Figure 6B")
    optres.plot_model_response_peaks([None, None, None], models=[LNS_AZ, LNS_D, LNS_N], names=["AZ", "D", "N"], colors=[COLORS["AZ"], COLORS["D"], COLORS["N"]], alphas=[1]*3)
