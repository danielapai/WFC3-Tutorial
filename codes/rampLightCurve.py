#! /usr/bin/env python
from __future__ import division, absolute_import
from __future__ import print_function
import itertools
import numpy as np
from .HST_utility import HSTsinLC, HSTtransitLC


def RECTE(
        nTrap_s,
        eta_trap_s,
        tau_trap_s,
        nTrap_f,
        eta_trap_f,
        tau_trap_f,
        cRates,
        tExp,
        exptime=180,
        trap_pop_s=0,
        trap_pop_f=0,
        dTrap_s=0,
        dTrap_f=0,
        dt0=0,
        lost=0,
        mode='scanning'
):
    """Hubble Space Telescope ramp effet model

    Parameters:
    cRates -- intrinsic count rate of each exposures, unit e/s
    tExp -- start time of every exposures
    expTime -- (default 180 seconds) exposure time of the time series
    trap_pop -- (default 0) number of occupied traps at the beginning of the observations
    dTrap -- (default [0])number of extra trap added in the gap
    between two orbits
    dt0 -- (default 0) possible exposures before very beginning, e.g.,
    possible guiding adjustment
    lost -- (default 0, no lost) proportion of trapped electrons that are not eventually detected
    (mode) -- (default scanning, scanning or staring, or others), for scanning mode
      observation , the pixel no longer receive photons during the overhead
      time, in staring mode, the pixel keps receiving elctrons
    """

    try:
        dTrap_f = itertools.cycle(dTrap_f)
        dTrap_s = itertools.cycle(dTrap_s)
        dt0 = itertools.cycle(dt0)
    except TypeError:
        dTrap_f = itertools.cycle([dTrap_f])
        dTrap_s = itertools.cycle([dTrap_s])
        dt0 = itertools.cycle([dt0])
    obsCounts = np.zeros(len(tExp))
    trap_pop_s = min(trap_pop_s, nTrap_s)
    trap_pop_f = min(trap_pop_f, nTrap_f)
    for i in range(len(tExp)):
        try:
            dt = tExp[i+1] - tExp[i]
        except IndexError:
            dt = exptime
        f_i = cRates[i]
        c1_s = eta_trap_s * f_i / nTrap_s + 1 / tau_trap_s  # a key factor
        c1_f = eta_trap_f * f_i / nTrap_f + 1 / tau_trap_f
        # number of trapped electron during one exposure
        dE1_s = (eta_trap_s * f_i / c1_s - trap_pop_s) * (1 - np.exp(-c1_s * exptime))
        dE1_f = (eta_trap_f * f_i / c1_f - trap_pop_f) * (1 - np.exp(-c1_f * exptime))
        dE1_s = min(trap_pop_s + dE1_s, nTrap_s) - trap_pop_s
        dE1_f = min(trap_pop_f + dE1_f, nTrap_f) - trap_pop_f
        trap_pop_s = min(trap_pop_s + dE1_s, nTrap_s)
        trap_pop_f = min(trap_pop_f + dE1_f, nTrap_f)
        obsCounts[i] = f_i * exptime - dE1_s - dE1_f
        if dt < 5 * exptime:  # whether next exposure is in next batch of exposures
            # same orbits
            if mode == 'scanning':
                # scanning mode, no incoming flux between exposures
                dE2_s = - trap_pop_s * (1 - np.exp(-(dt - exptime)/tau_trap_s))
                dE2_f = - trap_pop_f * (1 - np.exp(-(dt - exptime)/tau_trap_f))
            elif mode == 'staring':
                # for staring mode, there is flux between exposures
                dE2_s = (eta_trap_s * f_i / c1_s - trap_pop_s) * (1 - np.exp(-c1_s * (dt - exptime)))
                dE2_f = (eta_trap_f * f_i / c1_f - trap_pop_f) * (1 - np.exp(-c1_f * (dt - exptime)))
            else:
                # others, same as scanning
                dE2_s = - trap_pop_s * (1 - np.exp(-(dt - exptime)/tau_trap_s))
                dE2_f = - trap_pop_f * (1 - np.exp(-(dt - exptime)/tau_trap_f))
            trap_pop_s = min(trap_pop_s + dE2_s, nTrap_s)
            trap_pop_f = min(trap_pop_f + dE2_f, nTrap_f)
        elif dt < 1200:
            # considering in orbit download scenario
            trap_pop_s = min(trap_pop_s * np.exp(-(dt-exptime)/tau_trap_s), nTrap_s)
            trap_pop_f = min(trap_pop_f * np.exp(-(dt-exptime)/tau_trap_f), nTrap_f)
        else:
            # switch orbit
            dt0_i = next(dt0)
            trap_pop_s = min(trap_pop_s * np.exp(-(dt-exptime-dt0_i)/tau_trap_s) + next(dTrap_s), nTrap_s)
            trap_pop_f = min(trap_pop_f * np.exp(-(dt-exptime-dt0_i)/tau_trap_f) + next(dTrap_f), nTrap_f)
            f_i = cRates[i + 1]
            c1_s = eta_trap_s * f_i / nTrap_s + 1 / tau_trap_s  # a key factor
            c1_f = eta_trap_f * f_i / nTrap_f + 1 / tau_trap_f
            dE3_s = (eta_trap_s * f_i / c1_s - trap_pop_s) * (1 - np.exp(-c1_s * dt0_i))
            dE3_f = (eta_trap_f * f_i / c1_f - trap_pop_f) * (1 - np.exp(-c1_f * dt0_i))
            dE3_s = min(trap_pop_s + dE3_s, nTrap_s) - trap_pop_s
            dE3_f = min(trap_pop_f + dE3_f, nTrap_f) - trap_pop_f
            trap_pop_s = min(trap_pop_s + dE3_s, nTrap_s)
            trap_pop_f = min(trap_pop_f + dE3_f, nTrap_f)
        trap_pop_s = max(trap_pop_s, 0)
        trap_pop_f = max(trap_pop_f, 0)

    return obsCounts


def rampModel(
        nTrap_s,
        eta_s,
        tau_s,
        nTrap_f,
        eta_f,
        tau_f,
        cRate,
        exptime,
        trap_pop_s=0,
        trap_pop_f=0,
        dTrap_s=0,
        dTrap_f=0,
        dt0=0):
    """ calculate the light curve suffer from ramp effect

    :param nTrap_s: number of traps (slow population)
    :param eta_s: trapping efficiency (slow population)
    :param tau_s: trapping timescale (slow population)
    :param nTrap_f: number of traps (fast population)
    :param eta_f: trapping efficiency (fast population)
    :param tau_f: trapping timescale (fast population)
    :param tExp: exposure time stamp
    :param cRate: average count rate
    :param trap_pop_s: initial trapped charges (slow populaiton)
    :param trap_pop_f: initial trapped charges (fast population)
    :param dTrap_s: extra trapped charges (slow population)
    :param dTrap_f: extra trapped charges (fast population)
    :param dt0: extra time before the first exposure
    :returns: time stamps, observed count
    :rtype: numpy array

    """
    param_sin = {}
    param_sin['period'] = 4.5
    param_sin['phase'] = np.random.uniform(0, 2 * np.pi, 1)
    param_sin['amplitude'] = 0.05
    param_transit = {}
    param_transit['period'] = 1.58040464894  # days
    param_transit['rp'] = 0.0134**0.5  # stellar radii
    param_transit['t0'] = 220
    param_transit['a'] = 15
    params = param_transit
    counts, tExp, counts_mod, t_mod = HSTtransitLC(
        exptime, cRate, params, orbits=4, plot=False)
    obsCount = RECTE(nTrap_s,
                     eta_s,
                     tau_s,
                     nTrap_f,
                     eta_f,
                     tau_f,
                     counts / exptime,
                     tExp,
                     exptime,
                     trap_pop_s=0,
                     trap_pop_f=0,
                     dTrap_s=0,
                     dTrap_f=0,
                     dt0=0,
                     lost=0)
    return obsCount, tExp, counts_mod, t_mod


def rampCorrection(
        cRate,
        exptime,
        trap_pop_s=0,
        trap_pop_f=0,
        dTrap_s=0,
        dTrap_f=0,
        dt0=0):
    """ calculate the light curve suffer from ramp effect

    :param nTrap_s: number of traps (slow population)
    :param eta_s: trapping efficiency (slow population)
    :param tau_s: trapping timescale (slow population)
    :param nTrap_f: number of traps (fast population)
    :param eta_f: trapping efficiency (fast population)
    :param tau_f: trapping timescale (fast population)
    :param tExp: exposure time stamp
    :param cRate: average count rate
    :param trap_pop_s: initial trapped charges (slow populaiton)
    :param trap_pop_f: initial trapped charges (fast population)
    :param dTrap_s: extra trapped charges (slow population)
    :param dTrap_f: extra trapped charges (fast population)
    :param dt0: extra time before the first exposure
    :returns: time stamps, observed count
    :rtype: numpy array

    """
    param_sin = {}
    param_sin['period'] = 4.5
    param_sin['phase'] = np.random.uniform(0, 2 * np.pi, 1)
    param_sin['amplitude'] = 0.05
    param_transit = {}
    param_transit['period'] = 1.58040464894  # days
    param_transit['rp'] = 0.0134**0.5  # stellar radii
    param_transit['t0'] = 220
    param_transit['a'] = 15
    params = param_transit
    counts, tExp, counts_mod, t_mod = HSTtransitLC(exptime, cRate, params, orbits=4, plot=False)
    nTrap_s = 1525.38  # 1320.0
    eta_s = 0.013318  # 0.01311
    tau_s = 1.63e4
    nTrap_f = 162.38
    eta_f = 0.008407
    tau_f = 281.463
    obsCount = RECTE(nTrap_s,
                     eta_s,
                     tau_s,
                     nTrap_f,
                     eta_f,
                     tau_f,
                     counts / exptime,
                     tExp,
                     exptime,
                     trap_pop_s=trap_pop_s,
                     trap_pop_f=trap_pop_f,
                     dTrap_s=dTrap_s,
                     dTrap_f=dTrap_f,
                     dt0=0,
                     lost=0)
    return obsCount, tExp


if __name__ == '__main__':
    nTrap_s = 1000
    eta_s = 0.015
    tau_s = 10000
    nTrap_f = 500
    eta_f = 0.005
    tau_f = 100
    cRate = 200
    exptime = 100
    lc, t, lc0, t0 = rampModel(nTrap_s,
                               eta_s,
                               tau_s,
                               nTrap_f,
                               eta_f,
                               tau_f,
                               cRate,
                               exptime)
