import shelve
import numpy as np
import matplotlib.pyplot as plt

DBFileName = 'codes/binned_lightcurves_visit_01.shelve'
saveDB = shelve.open(DBFileName)

LCarray = saveDB['LCmat']
ERRarray = saveDB['ERRmat']
time = saveDB['time']
wavelength = saveDB['wavelength']
orbit = saveDB['orbit']
orbit_transit = np.array([2])  # transit occurs in the third orbits
expTime = saveDB['expTime']
saveDB.close()

# plot light curve of the second channel and the sixth channel
fig1 = plt.figure(figsize=(10, 6))
ax1 = fig1.add_subplot(211)
ax1.errorbar(time, LCarray[1, :], yerr=ERRarray[1, :], ls='none')
ax1.set_title('Light curve for Channel 2 ($\lambda_c$={0:.2} micron)'.format(wavelength[1]))
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Fluence [e]')
ax2 = fig1.add_subplot(212)
ax2.errorbar(time, LCarray[5, :], yerr=ERRarray[5, :], ls='none')
ax2.set_title('Light curve for Channel 6 ($\lambda_c$={0:.2} micron)'.format(wavelength[6]))
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Fluence [e]')
fig1.tight_layout()
