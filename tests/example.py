#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 10:12:26 2022

@author: pablo
"""

import numpy as np
from matplotlib import pyplot as plt
from wetc import signal2noise
import os

# Fiber injection f/ratio
fFP = 3.2
# Collimator f/ratio
fcol = 3.1
# Camera f/ratio
fcam = 1.8

instrument_mode = 'blueLR'
seeing = 1.0

rv = False
LIFU = True
offset = 0.1
profile = 'Moffat'
betam = 2.5


eff = None  # Efficiency
dark = 0.
rn = 2.5  # Read-out-noise

instrument_params = dict(
    blueLR=dict(res=5000., cwave=4900., pfEff=0.839, fiberEff=0.784,
                specEff=0.567, QE=0.924),
    redLR=dict(res=5000., cwave=7950., pfEff=0.870, fiberEff=0.852,
               specEff=0.607, QE=0.805),
    blueHR=dict(res=21000., cwave=4250., pfEff=0.800, fiberEff=0.691,
                specEff=0.337, QE=0.923),
    greenHR=dict(res=21000., cwave=5130., pfEff=0.833, fiberEff=0.794,
                 specEff=0.498, QE=0.927),
    redHR=dict(res=21000., cwave=6450., pfEff=0.861, fiberEff=0.837,
               specEff=0.505, QE=0.947)
    )

seeing = signal2noise.totalSeeing(seeing)

# fiber diameter
fiberD = 1.3045

# output radial velocity error?
if rv:
    rvscale = 0.6

# ----------------------------------------------------------------------------------------------------------------------
# %% Example using a single photometric input value
# ----------------------------------------------------------------------------------------------------------------------
S = signal2noise.Signal(offset=offset,
                        LIFU=True,
                        **instrument_params[instrument_mode],
                        fiberD=fiberD,
                        fFP=fFP, fcol=fcol, fcam=fcam,
                        profile=profile,
                        betam=betam, throughput_table=True, sky_model=True)

time = 1200.
mag = 24.
band = 'B'
sb = True

skysb = 22.7
skyband = 'B'
airmass = 1.2

n_exp = 3

snr = S.S2N(time=time, mag=mag, band=band, airmass=airmass, seeing=seeing, rn=rn, dark=dark,
            eff=eff, sb=sb, skysb=skysb, skyband=skyband, n_exposures=n_exp)

s2n = snr['SNR']

central_wl = S.get_band_central_wl('g')
print('Dispersion', S.disp, '\nEfficiency', S.efficiency())

print('Spectral resolving power R = {:5}'.format(int(S.res)))
print('Average dispersion (Ang/pixel) = {:8.4}'.format(S.disp))
print('Number of pixels/fiber along slit = {:5.2}'.format(S.fiberCCD))
print('Resolution element (Ang) = {:6.3}'.format(S.fiberCCD*S.disp))
print('Efficiency = {:4.2} Effective area = {:5.2} m^2'.format(
    snr['efficiency'], snr['effectivearea']))
print('Object photons/pixel = {} sky photons (between lines)/pixel = {}'
      .format(int(snr['objectphotons']), int(snr['skyphotons'])))
print('(both integrated over spatial direction)')
print('S/N/Ang = {:8.2f} S/N/resolution element = {:8.2f}'
      .format(s2n, snr['SNRres']))
print('S/N/pix = {:8.2f}'.format(snr['SNRpix']))

# ----------------------------------------------------------------------------------------------------------------------
# % Example using a spectra as input
# ----------------------------------------------------------------------------------------------------------------------
S = signal2noise.Signal(offset=offset,
                        LIFU=True,
                        **instrument_params[instrument_mode],
                        fiberD=fiberD,
                        fFP=fFP, fcol=fcol, fcam=fcam,
                        profile=profile,
                        betam=betam, throughput_table=True,
                        sky_model=True,
                        inc_moon=True,
                        moon_phase='new')

path_to_file = os.path.join(os.path.dirname(__file__), 'test_spectra_24_mag_arcsec')
wave, f_lambda = np.loadtxt(path_to_file, unpack=True)

mags = - 2.5 * np.log10(f_lambda * wave**2 / 3e18) - 48.6

snr_new = S.S2N_spectra(time=time, wave=wave, spectra=f_lambda.copy(),
                        airmass=airmass, seeing=seeing, sb=sb, n_exposures=n_exp)

S.build_sky_model(inc_moon=True, moon_phase='crescent')
snr_crescent = S.S2N_spectra(time=time, wave=wave, spectra=f_lambda.copy(),
                             airmass=airmass, seeing=seeing, sb=sb, n_exposures=n_exp)

S.build_sky_model(inc_moon=True, moon_phase='quarter')
snr_quarter = S.S2N_spectra(time=time, wave=wave, spectra=f_lambda.copy(),
                            airmass=airmass, seeing=seeing, sb=sb, n_exposures=n_exp)
S.build_sky_model(inc_moon=True, moon_phase='gibbous')
snr_gibbous = S.S2N_spectra(time=time, wave=wave, spectra=f_lambda.copy(),
                            airmass=airmass, seeing=seeing, sb=sb, n_exposures=n_exp)
#S.build_sky_model(inc_moon=True, moon_phase='full')
#snr_full = S.S2N_spectra(time=time, wave=wave, spectra=f_lambda.copy(),
#                         airmass=airmass, seeing=seeing, sb=sb, n_exposures=n_exp)

fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(10, 5), sharex=True)
ax = axs[0]
ax.plot(wave, mags)
ax.plot(central_wl, mag, 'ro')
ax.axvline(4686, color='k', label=r'$g$ band')
ax.set_ylabel(r'$F_\lambda$ (erg/s/AA/cm2/arcsec^2)')
ax.legend()
ax = axs[1]
ax.plot(wave, snr_new['SNR'], c='k', label='New moon')
ax.plot(wave, snr_crescent['SNR'], c='green', label='Crescent moon')
ax.plot(wave, snr_quarter['SNR'], c='orange', label='Quarter moon')
ax.plot(wave, snr_gibbous['SNR'], c='red', label='Gibbous moon')
# ax.plot(wave, snr_full['SNR'], c='fuchsia', label='Full moon')
ax.plot(central_wl, s2n, 'ro')
ax.axvline(4686, color='k')
ax.set_ylabel(r'$\rm SNR/\AA$')
# ax.set_yscale('log')
ax.grid(visible=True, which='both')
ax.legend(loc='center', bbox_to_anchor=(1., 0.5), framealpha=1)
ax = axs[2]
ax.plot(wave, S.efficiency(wave), c='g')
ax.plot(S.waveEff,
        S.QE[0]*S.fiberEff[0]*S.specEff[0]*S.pfEff[0]*S.m1Eff[0]*S.obscEff[0],
        c='b', lw=0.8)
ax.plot(S.waveEff,
        S.QE[1]*S.fiberEff[1]*S.specEff[1]*S.pfEff[1]*S.m1Eff[1]*S.obscEff[1],
        c='r', lw=0.8)
ax.set_ylabel(r'WEAVE Throughput')
ax.set_xlabel(r'$\lambda$')
plt.show()

#  \(ﾟ▽ﾟ)/