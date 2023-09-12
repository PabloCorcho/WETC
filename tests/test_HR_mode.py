#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 13:51:51 2022

@author: pablo
"""

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

seeing = 1.0

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

rv = False
LIFU = True
offset = 0.1
profile = 'Moffat'
betam = 2.5

eff = None  # Efficiency
dark = 0.
rn = 2.5  # Read-out-noise

seeing = signal2noise.totalSeeing(seeing)

# fiber diameter
fiberD = 1.3045

# output radial velocity error?
if rv:
    rvscale = 0.6


mag = 24.
band = 'B'
sb = True

skysb = 22.7
skyband = 'B'
airmass = 1.2

# time = 1200.
# n_exp = 3

path_to_file = os.path.join(os.path.dirname(__file__), 'test_spectra_24_mag_arcsec')
wave, f_lambda = np.loadtxt(path_to_file, unpack=True)
mags = - 2.5 * np.log10(f_lambda * wave**2 / 3e18) - 48.6
# ----------------------------------------------------------------------------------------------------------------------
# % Example using a spectra as input
# ----------------------------------------------------------------------------------------------------------------------
time = 1200.
n_exp = 3
S = signal2noise.Signal(offset=offset,
                        LIFU=True,
                        fiberD=fiberD,
                        **instrument_params['blueLR'],
                        fFP=fFP, fcol=fcol, fcam=fcam,
                        profile=profile,
                        betam=betam, throughput_table=True,
                        sky_model=True,
                        inc_moon=True,
                        moon_phase='new')


lr_snr_new = S.S2N_spectra(time=time, wave=wave, spectra=f_lambda.copy(),
                           airmass=airmass, seeing=seeing, sb=sb,
                           n_exposures=n_exp)

S.build_sky_model(inc_moon=True, moon_phase='crescent')
lr_snr_crescent = S.S2N_spectra(time=time, wave=wave, spectra=f_lambda.copy(),
                                airmass=airmass, seeing=seeing,
                                sb=sb, n_exposures=n_exp)

S.build_sky_model(inc_moon=True, moon_phase='quarter')
lr_snr_quarter = S.S2N_spectra(time=time, wave=wave, spectra=f_lambda.copy(),
                               airmass=airmass, seeing=seeing,
                               sb=sb, n_exposures=n_exp)
S.build_sky_model(inc_moon=True, moon_phase='gibbous')
lr_snr_gibbous = S.S2N_spectra(time=time, wave=wave, spectra=f_lambda.copy(),
                               airmass=airmass, seeing=seeing,
                               sb=sb, n_exposures=n_exp)

del S

# =============================================================================
# HR mode
# =============================================================================
time = 1200.
n_exp = 9

S_hr = signal2noise.Signal(offset=offset,
                           LIFU=True,
                           fiberD=fiberD,
                           fFP=fFP, fcol=fcol, fcam=fcam,
                           profile=profile,
                           betam=betam, throughput_table=True,
                           sky_model=True,
                           inc_moon=True,
                           mode='HR',
                           moon_phase='new')

snr_new = S_hr.S2N_spectra(
    time=time, wave=wave, spectra=f_lambda.copy(),
    airmass=airmass, seeing=seeing, sb=sb, n_exposures=n_exp)

S_hr.build_sky_model(inc_moon=True, moon_phase='crescent')
snr_crescent = S_hr.S2N_spectra(time=time, wave=wave, spectra=f_lambda.copy(),
                                airmass=airmass, seeing=seeing,
                                sb=sb, n_exposures=n_exp)

S_hr.build_sky_model(inc_moon=True, moon_phase='quarter')
snr_quarter = S_hr.S2N_spectra(time=time, wave=wave, spectra=f_lambda.copy(),
                               airmass=airmass, seeing=seeing,
                               sb=sb, n_exposures=n_exp)

S_hr.build_sky_model(inc_moon=True, moon_phase='gibbous')
snr_gibbous = S_hr.S2N_spectra(time=time, wave=wave, spectra=f_lambda.copy(),
                               airmass=airmass, seeing=seeing,
                               sb=sb, n_exposures=n_exp)

S_hr.build_sky_model(inc_moon=True, moon_phase='full')
snr_full = S_hr.S2N_spectra(time=time, wave=wave, spectra=f_lambda.copy(),
                            airmass=airmass, seeing=seeing, sb=sb, n_exposures=n_exp)

fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(10, 5), sharex=True)
ax = axs[0]
ax.plot(wave, mags)
ax.axvline(4686, color='k', label=r'$g$ band')
ax.set_ylabel(r'$F_\lambda$ (erg/s/AA/cm2/arcsec^2)')
ax.legend()
ax = axs[1]

ax.plot(wave, lr_snr_new['SNR'], c='k', ls=':')
ax.plot(wave, lr_snr_crescent['SNR'], c='green', ls=':')
ax.plot(wave, lr_snr_quarter['SNR'], c='orange', ls=':')
ax.plot(wave, lr_snr_gibbous['SNR'], c='red', ls=':')

ax.plot(wave, snr_new['SNR'], c='k', label='New moon')
ax.plot(wave, snr_crescent['SNR'], c='green', label='Crescent moon')
ax.plot(wave, snr_quarter['SNR'], c='orange', label='Quarter moon')
ax.plot(wave, snr_gibbous['SNR'], c='red', label='Gibbous moon')
# ax.plot(wave, snr_full['SNR'], c='fuchsia', label='Full moon')
ax.set_ylim(0, 8)
ax.axvline(4686, color='k')
ax.set_ylabel(r'$\rm SNR/\AA$')
# ax.set_yscale('log')
ax.grid(visible=True, which='both')
ax.legend(loc='center', bbox_to_anchor=(1., 0.5), framealpha=1)
ax = axs[2]
ax.plot(wave, S_hr.efficiency(wave), c='k', lw=2)
for i in range(S_hr.QE.shape[0]):
    ax.plot(S_hr.waveEff,
            S_hr.QE[i] * S_hr.fiberEff[i] * S_hr.specEff[i] * S_hr.pfEff[i]
            * S_hr.m1Eff[i] * S_hr.obscEff[i],
            lw=0.8)
ax.set_ylabel(r'WEAVE Throughput')
ax.set_xlabel(r'$\lambda$')
plt.show()

#  \(ﾟ▽ﾟ)/