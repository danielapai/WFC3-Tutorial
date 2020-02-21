import matplotlib.pyplot as plt
import numpy as np
from codes.rampLightCurve import rampModel, rampCorrection


def rampModelPlot(
    nTrap_s,
    eta_s,
    tau_s,
    nTrap_f,
    eta_f,
    tau_f,
    crate=200,  # electron per second
    exptime=100):
    """plot the ramp model profiles with various model parameters

    nTrap: number of traps for slow (_s) and fast (_fast) traps
    eta: trapping efficiencies
    tau: trap lifetimes
    """
    # rampModel function is made to conveniently change the charge trapping
    # related parameters
    # This function is not for correction purposes
    lc, t = rampModel(
        nTrap_s,
        eta_s,
        tau_s,
        nTrap_f,
        eta_f,
        tau_f,
        crate,
        exptime
    )
    fig = plt.gcf()
    plt.cla()
    plt.plot(t, lc, 'o')
    plt.xlabel('Time [s]')
    plt.ylabel('Flux count [e]')
    plt.ylim(crate*exptime * 0.98, crate*exptime*1.003)
    plt.draw()
