from misc import rebin
import numpy as np
import pandas as pd
import shelve
from os import path
import batman
import emcee
from lmfit.models import GaussianModel
import matplotlib.pyplot as plt
# from scipy.stats import shapiro, kstest, probplot
from misc import makedirs, mcmcRunner
from plot import plotMCMC
import sys
from scipy.interpolate import interp1d
plt.style.use('paper')

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


def transitModel2_white(params, t, expTime):
    """construct a double transit

    :param params: transit planet parameters
    :param t: time stamps in seconds
    :param expTime: exposure time for each frame
    :returns: light curve model
    :rtype: np.array

    """

    transit_params.t0 = params[1]  # time of inferior conjunction
    transit_params.per = 1.51087081  # orbital period
    transit_params.a = 20.4209
    transit_params.inc = 89.65
    transit_params.rp = params[0]  # planet radius (in units of stellar radii)
    transit_params.u = [params[4],
                        params[5]]  # linear limb darkening coefficients
    m_b = batman.TransitModel(
        transit_params, (t - 0.5 * expTime) / 86400, exp_time=expTime / 86400)
    lc1 = m_b.light_curve(transit_params)

    transit_params.t0 = params[3]
    transit_params.per = 2.4218233  # orbital period
    transit_params.a = 27.9569
    transit_params.inc = 89.67
    transit_params.rp = params[2]  # planet radius (in units of stellar radii)
    transit_params.u = [params[4], params[5]]  # linear limb darkening
    # coefficients
    m_c = batman.TransitModel(
        transit_params, (t - 0.5 * expTime) / 86400, exp_time=expTime / 86400)
    lc2 = m_c.light_curve(transit_params)
    lc = (lc1 + lc2) - 1  # two transit, remove one baseline
    return lc


def lnlike_white(params, t, f, ferr, expTime):
    """likelihood function

    :param params: transit parameters
    :param t: time-stamp of each transit
    :param f: flux sieries
    :param ferr: error sieries
    :param expTime: exposure time
    :returns: likelihood
    :rtype: float

    """
    L = -np.sum((f - transitModel2_white(params, t, expTime))**2 / (2*(ferr)**2)) -\
        0.5 * np.sum(np.log(2 * np.pi * (ferr)**2))
    return L


def lnpriori_white(params, a, asig, b, bsig):
    if params[3] < params[1]:
        prior_a = -0.5 * np.log(2 * np.pi * asig**2) - (params[4] - a)**2 / (
            2 * asig**2)
        prior_b = -0.5 * np.log(2 * np.pi * bsig**2) - (params[5] - b)**2 / (
            2 * bsig**2)
        return prior_a + prior_b
    else:
        return -np.inf


def lnprob_white(params, t, f, ferr, expTime, a, asig, b, bsig):
    """log of the posterior likelihood
    """
    return lnpriori_white(params, a, asig, b, bsig) + lnlike_white(
        params, t, f, ferr, expTime)


def transitModel2(params, t, expTime, t01, t02):
    """construct a double transit

    :param params: transit planet parameters
    :param t: time stamps in seconds
    :param expTime: exposure time for each frame
    :returns: light curve model
    :rtype: np.array

    """

    transit_params.t0 = t01  # time of inferior conjunction
    transit_params.per = 1.51087081  # orbital period
    transit_params.a = 20.4209
    transit_params.inc = 89.65
    transit_params.rp = params[0] # planet radius (in units of stellar radii)
    transit_params.u = [params[2],
                        params[3]]  # linear limb darkening coefficients
    m_b = batman.TransitModel(
        transit_params, (t - 0.5 * expTime) / 86400, exp_time=expTime / 86400)
    lc1 = m_b.light_curve(transit_params)

    transit_params.t0 = t02
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


def lnlike(params, t, f, ferr, expTime, t01, t02):
    """likelihood function

    :param params: transit parameters
    :param t: time-stamp of each transit
    :param f: flux sieries
    :param ferr: error sieries
    :param expTime: exposure time
    :returns: likelihood
    :rtype: float

    """
    L = -np.sum((f - transitModel2(params, t, expTime, t01, t02))**2 / (2*(ferr)**2)) -\
        0.5 * np.sum(np.log(2 * np.pi * (ferr)**2))
    return L


def lnpriori(params, a, asig, b, bsig):
    prior_a = -0.5 * np.log(2 * np.pi * asig**2) - (params[2] - a)**2 / (
        2 * asig**2)
    prior_b = -0.5 * np.log(2 * np.pi * bsig**2) - (params[3] - b)**2 / (
        2 * bsig**2)
    return prior_a + prior_b


def lnprob(params, t, f, ferr, expTime, t01, t02, a, asig, b, bsig):
    """log of the posterior likelihood
    """
    return lnlike(params, t, f, ferr, expTime, t01, t02) +\
        lnpriori(params, a, asig, b, bsig)


def fitTransit(dbFN, saveDIR, plotDIR):
    # first fit global limb_darkening coeff

    deRampDBFN = path.join(dbFN)
    deRampDB = shelve.open(deRampDBFN)
    t = deRampDB['time']
    orbit = deRampDB['orbit']
    LCmatrix = deRampDB['LCmatrix']
    Errmatrix = deRampDB['Errmatrix']
    expTime = deRampDB['expTime']
    xAim = deRampDB['x']
    wavelength = deRampDB['wavelength']
    deRampDB.close()

    # fit the broadband light curve first
    wlc = np.sum(
        LCmatrix / Errmatrix**2, axis=0) / np.sum(
            1 / Errmatrix**2, axis=0)
    werr = np.sqrt(1 / np.sum(1 / Errmatrix**2, axis=0))
    aw = 0.1526
    awsig = 0.01645
    bw = 0.4516
    bwsig = 0.02605

    params0 = [0.0852, 0.153, 0.08288, 0.146, aw, bw]
    nDim = len(params0)
    nWalkers = 64
    pos0 = [
        params0 * (1 + 0.001 * np.random.randn(nDim)) for k in range(nWalkers)
    ]
    sampler = emcee.EnsembleSampler(
        nWalkers,
        nDim,
        lnprob_white,
        args=(t, wlc, werr, expTime, aw, awsig, bw, bwsig),
        threads=4)
    nStep = 500
    burnin = 200
    width = 50
    print('Be Calm! MCMC starting!!!!')
    chain, params, param_stds = mcmcRunner(sampler, pos0, nStep, burnin, width)
    plt.close('all')
    fig1, fig2 = plotMCMC(
        t,
        wlc,
        werr,
        chain,
        params,
        transitModel2_white,
        expTime,
        argName=['d1', 't1', 'd2', 't2', 'u1', 'u2'])
    # save results
    fig1.savefig(path.join(plotDIR, 'corner_white.png'))
    fig2.savefig(path.join(plotDIR, 'lc_fit_white.png'))
    t01 = params[1]
    t02 = params[3]
    res_white = wlc - transitModel2_white(params, t, expTime)
    with shelve.open(path.join(saveDIR, 'mcmc_white.shelve')) as db:
        db['chain'] = chain
        db['params'] = params
        db['param_stds'] = param_stds

    # precalculated limb-darkening coefficients
    w0 = np.linspace(1.15, 1.65, 11)
    amid = np.array([
        0.1101, 0.1235, 0.0983, 0.123, 0.2686, 0.3099, 0.2734, 0.2053, 0.1438,
        0.0998, 0.08
    ])
    bmid = np.array([
        0.4713, 0.4646, 0.4405, 0.4319, 0.4669, 0.4387, 0.449, 0.4681, 0.4572,
        0.4265, 0.4095
    ])
    amin = np.array([
        0.1015, 0.1115, 0.0905, 0.1105, 0.219, 0.2524, 0.2333, 0.1863, 0.1305,
        0.0892, 0.0717
    ])
    amax = np.array([
        0.1194, 0.1369, 0.1063, 0.1391, 0.3352, 0.3856, 0.3232, 0.2264, 0.1559,
        0.1082, 0.0863
    ])
    bmin = np.array([
        0.4399, 0.4288, 0.409, 0.4072, 0.452, 0.4341, 0.4336, 0.441, 0.429,
        0.4022, 0.3793
    ])
    bmax = np.array([
        0.5064, 0.5034, 0.4725, 0.4569, 0.4864, 0.4543, 0.4408, 0.4993, 0.4897,
        0.4548, 0.434
    ])
    asigs = (amax - amin) / 2
    bsigs = (bmax - bmin) / 2

    aInterp = interp1d(w0, amid, kind='cubic', fill_value='extrapolate')
    amid = aInterp(wavelength)
    bInterp = interp1d(w0, bmid, kind='cubic', fill_value='extrapolate')
    bmid = bInterp(wavelength)
    asigInterp = interp1d(w0, asigs, kind='cubic', fill_value='extrapolate')
    asigs = asigInterp(wavelength)
    bsigInterp = interp1d(w0, bsigs, kind='cubic', fill_value='extrapolate')
    bsigs = bsigInterp(wavelength)
    nChannel = len(wavelength)

    # start of mcmc fitting
    Rp1List = np.zeros(nChannel)
    Rp2List = np.zeros(nChannel)
    RpErr1List = np.zeros(nChannel)
    RpErr2List = np.zeros(nChannel)
    resList = np.zeros(nChannel)
    for i in range(nChannel):
        a = amid[i]
        b = bmid[i]
        asig = asigs[i]
        bsig = bsigs[i]
        lc = LCmatrix[i, :]
        err = Errmatrix[i, :]
        params0 = [0.086, 0.084, a, b]
        nDim = len(params0)
        nWalkers = 64
        pos0 = [
            params0 * (1 + 0.001 * np.random.randn(nDim))
            for k in range(nWalkers)
        ]
        sampler = emcee.EnsembleSampler(
            nWalkers,
            nDim,
            lnprob,
            args=(t, lc, err, expTime, t01, t02, a, asig, b, bsig),
            threads=4)
        nStep = 500
        burnin = 200
        width = 50
        print('Be Calm! MCMC starting!!!!')
        chain, params, param_stds = mcmcRunner(sampler, pos0, nStep, burnin,
                                               width)
        lc_best_fit = transitModel2(params, t, expTime, t01, t02)
        res = lc - lc_best_fit
        plt.close('all')
        fig1, fig2 = plotMCMC(
            t,
            lc,
            err,
            chain,
            params,
            transitModel2,
            expTime,
            t01,
            t02,
            argName=['d1', 'd2', 'u1', 'u2'])
        # save results
        fig1.savefig(
            path.join(plotDIR, 'corner_white_channel_{0:02d}.png'.format(i)))
        fig2.savefig(
            path.join(plotDIR, 'lc_fit_white_channel_{0:02d}.png'.format(i)))
        with shelve.open(path.join(saveDIR, 'mcmc_white.shelve')) as db:
            db['chain'] = chain
            db['params'] = params
            db['param_stds'] = param_stds
            db['lc'] = lc
            db['err'] = err
            db['lc_fit'] = lc_best_fit
        resList[i] = res.std()
        Rp1List[i] = params[0]
        RpErr1List[i] = param_stds[0]
        Rp2List[i] = params[1]
        RpErr2List[i] = param_stds[1]
    return Rp1List, RpErr1List, Rp2List, RpErr2List, resList, res_white,wavelength


if __name__ == '__main__':
    dbFN = path.expanduser(
        '~/Documents/TRAPPIST-1/pickle/visit_1/deRamp_visit_01.shelve')
    saveDIR = path.expanduser(
        '~/Documents/TRAPPIST-1/results/Trappist_1_Visit_1/transit_fit')
    plotDIR = path.expanduser(
        '~/Documents/TRAPPIST-1/results/Trappist_1_Visit_1/transit_fit')
    if not path.exists(saveDIR):
        makedirs(saveDIR)
    if not path.exists(plotDIR):
        makedirs(plotDIR)
    Rp1, RpErr1, Rp2, RpErr2, res, resw, wavelength = fitTransit(
        dbFN, saveDIR, plotDIR)
    Rp1_0 = Rp1.mean()
    Rp2_0 = Rp2.mean()
    Rp1_ppm = (Rp1**2 - Rp1_0**2) * 1e6
    Rp2_ppm = (Rp2**2 - Rp2_0**2) * 1e6
    RpErr1_ppm = 2 * Rp1 * RpErr1 * 1e6
    RpErr2_ppm = 2 * Rp2 * RpErr2 * 1e6
    plt.close('all')
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.errorbar(wavelength, Rp1_ppm, yerr=RpErr1_ppm, ls='none', marker='o')
    ax2 = fig.add_subplot(212)
    ax2.errorbar(wavelength, Rp2_ppm, yerr=RpErr2_ppm, ls='none', marker='o')
    ax1.set_ylim([-1500, 1500])
    ax2.set_ylim([-1500, 1500])
    plt.show()
