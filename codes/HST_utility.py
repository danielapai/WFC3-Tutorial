"""HST related convenience functions
"""

import matplotlib.pyplot as plt
import batman
import numpy as np


def HSTtiming(exptime,
         orbits=4,
         orbitLength=96,  # min
         visibility=50,  # min
         overhead=20):
    """generate a series of time for the starting of exposure

    :param exptime: exposure time
    :param orbits: number of orbits
    :param orbitLength: (default 96 minutes) the length of one orbit
    :param visibility: (default 20 minutes) the length of visible time
    period per orbit
    :param overhead: (default 20  seconds) the length of overhead per exposure

    """
    t = np.arange(0, visibility * 60, (exptime + overhead))
    t0 = np.arange(orbits) * orbitLength * 60  # starting time of each orbits
    return np.concatenate([t + t0i for t0i in t0])


def HSTsinLC(expTime, cRate, param,
             orbits=4,
             orbitLength=96,  # min
             visibility=50,  # min
             overhead=20,  # s
             plot=False):
    """Simulate a sinusoidal signal observed by HST

    :param expTime: exposure time
    :param cRate: average count rate
    :param param: parameters describing the sinusoid, a dictionary
    :param orbits: number of orbits
    :param orbit: (default 96 min) length of a orbit
    :param visibility: (default 50 min) length of visible period per orbit
    :param overhead: overhead [s] per exposure
    :param plot: wheter to make a plot
    :returns: count, t
    :rtype: tuple

    """

    t = HSTtiming(expTime, orbits, orbitLength, visibility, overhead)
    # calculate count, be careful that period in h
    count = cRate * expTime * \
        (1 + np.sin((2 * np.pi * t / (param['period'] * 3600)) +
                    param['phase']) * param['amplitude'])

    if plot:
        fig, ax = plt.subplots()
        ax.plot(t/60, count, 'o')
        ax.set_xlabel('Time [min]')
        ax.set_ylabel('count [$\mathsf{e^-}$]')
        plt.show()
    return count, t


def HSTtransitLC(expTime, cRate,
                 param,
                 orbits=4,
                 orbitLength=96,  # min
                 visibility=50,  # min
                 overhead=20,  # s
                 plot=False):
    """Simulate a transit signal observed by HST

    :param expTime: exposure time
    :param cRate: average count rate
    :param param: parameters describing the sinusoid, a batman dictionary
    :param orbits: number of orbits
    :param orbit: (default 96 min) length of a orbit
    :param visibility: (default 50 min) length of visible period per orbit
    :param overhead: overhead [s] per exposure
    :param plot: wheter to make a plot
    :returns: count, t
    :rtype: tuple

    """
    params = batman.TransitParams()  # object to store transit parameters
    params.t0 = param['t0'] / (24 * 60)  # time of inferior conjunction
    params.per = param['period']  # orbital period
    params.rp = param['rp']  # planet radius (in units of stellar radii)
    params.a = param['a']  # semi-major axis (in units of stellar radii)
    params.ecc = 0  # eccentricity
    params.inc = 89.1  # inclination
    params.w = 90.  # longitude of periastron (in degrees)
    params.limb_dark = "linear"  # limb darkening model
    params.u = [0.28]  # limb darkening coefficients

    t = HSTtiming(expTime, orbits, orbitLength, visibility, overhead)
    m = batman.TransitModel(params, t / (24 * 3600))
    # calculate count, be careful that period in h
    count = cRate * expTime * m.light_curve(params)
    t_mod = np.linspace(t.min(), t.max(), 10*len(t))
    m = batman.TransitModel(params, t_mod / (24*3600))
    count_mod = cRate * expTime * m.light_curve(params)
    if plot:
        fig, ax = plt.subplots()
        ax.plot(t / 60, count, 'o')
        ax.set_xlabel('Time [min]')
        ax.set_ylabel(r'count [$\mathsf{e^-}$]')
        plt.show()
    return count, t, count_mod, t_mod
