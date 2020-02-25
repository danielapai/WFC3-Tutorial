import matplotlib.pyplot as plt
from lmfit import Parameters, Model

from codes.RECTE import RECTE
from codes.RECTECorrector import RECTECorrector2

def removeRamp(p0,
               time,
               LCArray,
               ErrArray,
               orbits,
               orbits_transit,
               expTime,
               scanDirect):
    """
    remove Ramp systemetics with RECTE

    :param p0: initial parameters
    :param time: time stamp of each exposure
    :param LCArray: numpy array that stores all light curves
    :param ErrArray: light curve uncertainties
    :param orbits: orbit number for each exposure
    :param orbits_transit: orbit number that transits occur. These orbits
    are excluded in the fit
    :param expTime: exposure time
    :param scanDirect: scanning direction for each exposure. 0 for forward,
    1 for backward
    """
    nLC = LCArray.shape[0]  # number of light curves
    correctedArray = LCArray.copy()
    correctedErrArray = ErrArray.copy()
    modelArray = LCArray.copy()
    crateArray = LCArray.copy()
    slopeArray = LCArray.copy()
    p = p0.copy()
    for i in range(nLC):
        correctTerm, crate, bestfit, slope = RECTECorrector2(
            time,
            orbits,
            orbits_transit,
            LCArray[i, :],
            p,
            expTime,
            scanDirect)
        # corrected light curve/error are normalized to the baseline
        correctedArray[i, :] = LCArray[i, :] / correctTerm / (crate)
        correctedErrArray[i, :] = ErrArray[i, :] / correctTerm / (crate)
        modelArray[i, :] = bestfit
        crateArray[i, :] = crate
        slopeArray[i, :] = slope
    return correctedArray, correctedErrArray, modelArray, crateArray, slopeArray
