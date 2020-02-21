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
    fig = plt.gcf()
    plt.cla()
    plt.plot(t, lc, 'o')
    plt.xlabel('Time [s]')
    plt.ylabel('Flux count [e]')
    plt.ylim(crate*exptime * 0.98, crate*exptime*1.003)
    plt.draw()
