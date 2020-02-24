from RECTE import RECTE
from lmfit import Parameters, Model
from RECTECorrector import RECTECorrector2
import matplotlib.pyplot as plt

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

infoFN = './demonstration_data/TRAPPIST_Info.csv'
info = pd.read_csv(infoFN)
grismInfo = info[info['Filter'] == 'G141']
scanDirect = grismInfo['ScanDirection'].values
p = Parameters()
p.add('trap_pop_s', value=0, min=0, max=200, vary=True)
p.add('trap_pop_f', value=0, min=0, max=100, vary=True)
p.add('dTrap_f', value=0, min=0, max=200, vary=True)
p.add('dTrap_s', value=50, min=0, max=100, vary=True)
LCarray_noRamp, ERRarray_noRamp, Modelarray, cratearray, slopearray = removeRamp(
    p,
    time,
    LCarray,
    ERRarray,
    orbit,
    orbit_transit,
    expTime,
    scanDirect)

fig2 = plt.figure(figsize=(10, 6))
ax1 = fig2.add_subplot(211)

ax1.errorbar(
    time / 3600,
    LCarray[1, :],
    yerr=ERRarray[1, :],
    fmt='.',
    ls='')
for o in [0, 1, 3]:
    ax1.plot(
        time[orbit == o] / 3600,
        Modelarray[1, orbit == o],
        '.-',
        color='C1')
ax1.set_title('Light curve for Channel 2 ($\lambda_c$={0:.2} micron)'.format(wavelength[1]))
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Fluence [e]')

ax2 = fig2.add_subplot(212)
ax2.errorbar(
    time / 3600,
    LCarray[6, :],
    yerr=ERRarray[6, :],
    fmt='.',
    ls='')
for o in [0, 1, 3]:
    ax2.plot(
        time[orbit == o] / 3600,
        Modelarray[6, orbit == o],
        '.-',
        color='C1')
ax2.set_title('Light curve for Channel 6 ($\lambda_c$={0:.2} micron)'.format(wavelength[6]))
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Fluence [e]')
fig2.tight_layout()
