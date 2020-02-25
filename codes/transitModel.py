import numpy as np
import pandas as pd
import shelve
from os import path
import batman
import emcee
from lmfit.models import GaussianModel
import matplotlib.pyplot as plt
# from scipy.stats import shapiro, kstest, probplot
import sys
from scipy.interpolate import interp1d


# define the transit model as a global parameter
transit_params = batman.TransitParams()  # object to store transit parameters
transit_params.t0 = 0  # time of inferior conjunction
transit_params.per = 1.0  # orbital period
transit_params.rp = 0.1  # planet radius (in units of stellar radii)
transit_params.a = 15.23  # semi-major axis (in units of stellar radii)
transit_params.ecc = 0  # eccentricity
transit_params.inc = 89.1  # inclination
transit_params.w = 90.  # longitude of periastron (in degrees)
transit_params.limb_dark = "quadratic"  # limb darkening model
transit_params.u = [0.15, 0.45]  # limb darkening coefficients
# initialize the model
m = batman.TransitModel(transit_params, np.linspace(0, 10))


def transitModel(params, t, expTime):
    """construct a double transit

    :param params: transit planet parameters
    :param t: time stamps in seconds
    :param expTime: exposure time for each frame
    :returns: light curve model
    :rtype: np.array

    """

    transit_params.t0 = 0.153  # fix the transit timing from broadband light curve fit
    transit_params.per = 1.51087081  # orbital period
    transit_params.a = 20.4209
    transit_params.inc = 89.65
    transit_params.rp = params[0]  # planet radius (in units of stellar radii)
    transit_params.u = [params[2],
                        params[3]]  # linear limb darkening coefficients
    m_b = batman.TransitModel(
        transit_params, (t + 0.5 * expTime) / 86400, exp_time=expTime / 86400)
    lc1 = m_b.light_curve(transit_params)

    transit_params.t0 = 0.146  # fix the transit timing from broadband light curve fit
    transit_params.per = 2.4218233  # orbital period
    transit_params.a = 27.9569
    transit_params.inc = 89.67
    transit_params.rp = params[1]  # planet radius (in units of stellar radii)
    transit_params.u = [params[2], params[3]]  # linear limb darkening
    # coefficients
    m_c = batman.TransitModel(
        transit_params, (t - 0.5 * expTime) / 86400, exp_time=expTime / 86400)
    lc2 = m_c.light_curve(transit_params)
    lc = (lc1 + lc2) - 1  # two transit, remove one baseline
    return lc
