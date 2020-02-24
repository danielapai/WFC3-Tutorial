import numpy as np
import matplotlib.pyplot as plt
from codes.rampLightCurve import rampModel, rampCorrection

def rampCorrectionPlot(
    trap_pop_s,
    trap_pop_f,
    dTrap_s,
    dTrap_f,
    crate=200,
    exptime=100):
    """plot the ramp model profiles with parameters that are used
        in ramp effect corrections

    trap_pop: initial states of the charge traps
    dTrap: added charges during earth occulation
    """
    lc, t = rampCorrection(
        crate,
        exptime,
        trap_pop_s,
        trap_pop_f,
        dTrap_s,
        dTrap_f,
    )
    trap_pop_s0 = 310
    trap_pop_f0 = 33
    dTrap_s0 = 76
    dTrap_f0 = 22

    lc0, t0 = rampCorrection(crate,
                             exptime,
                             trap_pop_s0,
                             trap_pop_f0,
                             dTrap_s0,
                             dTrap_f0)
    orbits = t0.copy()
    dt0 = t0[1] - t0[0]
    o0 = 0
    for i in range(len(orbits)-1):
        orbits[i] = o0
        if t0[i+1] - t0[i] > 5 * dt0:
            o0+=1
    orbits[-1] = o0
    fig = plt.gcf()
    plt.cla()
    for o_i in range(o0+1):
        plt.plot(t0[orbits == o_i], lc0[orbits == o_i],
                 color='C1', lw=2, alpha=0.8)
    plt.plot(t, lc, 'o', color='C0', mfc='none', label='Model')
    plt.xlabel('Time [s]')
    plt.ylabel('Flux count [e]')
    plt.ylim(crate*exptime * 0.98, crate*exptime*1.003)
    plt.draw()
