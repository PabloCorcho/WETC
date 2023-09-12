#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 13:38:54 2023

@author: pablo
"""

from wetc import signal2noise
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

hdul = fits.open(
    "/home/pablo/Research/WEAVE-Apertif/weave_fl/supercube_2963102  .fit")
inst = hdul[0].header['INSTRUME']
#hdul.info()

wcs = WCS(hdul[1].header)
wl = wcs.spectral.pixel_to_world_values(
    np.arange(0, hdul[1].header['NAXIS3'])) * 1e10

good = slice(200, wl.size - 200)

wl = wl[good]

cube = hdul[1].data[good] * hdul[5].data[good, np.newaxis, np.newaxis]
err = hdul[2].data[good]**-0.5 * hdul[5].data[good, np.newaxis, np.newaxis]

sky = np.nanmedian((hdul[3].data[good] - hdul[1].data[good]), axis=(1, 2))
sky *= hdul[5].data[good]
# sky /= 0.5**2  # pixel to arcsec**2
sky_sb = - 2.5 * np.log10(sky * wl**2 / 3e18) - 48.6

snr = cube / err
exposure_time = hdul[0].header['EXPTIME']

i,j = 100, 75
S = signal2noise.Signal(
                        LIFU=True,
                        throughput_table=True,
                        sky_model=True,
                        inc_moon=True,
                        moon_phase='new')

S.build_sky_model(sky_brightness_mag=sky_sb,
                  sky_brightness_mag_zero=np.ones_like(sky_sb) * 3631,
                  sky_brightness_wl=wl)

snr_pred = np.full_like(snr, fill_value=np.nan)
for i in range(30, 150):
    for j in range(30, 150):
        print(i, j)
        snr_new = S.S2N_spectra(time=exposure_time / 6,
                                n_exposures=6,
                                wave=wl,
                                spectra=cube[:, i, j],
                                airmass=hdul[0].header.get('AIRMASS'),
                                seeing=hdul[0].header.get('SEEINGE'),
                                sb=False)
        snr_pred[:, i,j] = snr_new['SNRpix']

mask = np.isfinite(snr_pred)

# %%
snr_bins = np.logspace(-1, 2, 101)
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.hist2d(snr[mask], snr_pred[mask], bins=[snr_bins, snr_bins],
           cmap='inferno', norm=LogNorm(vmin=1e3), density=False)
plt.colorbar(label='Nº pixs', orientation='horizontal')
plt.plot([0, 100], [0, 100], c='r', ls='--', lw=2)
plt.xlim(1, 100)
plt.ylim(1, 100)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('SNR/pix First Light')
plt.ylabel('SNR/pix Predicted')

plt.subplot(122)
plt.hist(snr[mask] / snr_pred[mask], bins=100, range=[0, 2])
plt.xlabel(r'$SNR_{FL}/\hat{SNR}$')
pct = np.nanpercentile(snr[mask] / snr_pred[mask], [16, 50, 84])
for p in pct:
    plt.axvline(p, c='k')
plt.annotate("p(16)={:.2f}\np(50)={:.2f}\np(84)={:.2f}".format(*pct),
             xy=(0.1, 0.9), xycoords='axes fraction', va='top')
plt.xlabel(r'$SNR_{FL}/\hat{SNR}$')
plt.ylabel(r'Nº pix')
plt.savefig("ETC_first_light_test_{}.png".format(inst), bbox_inches='tight')

hdul.close()
# %%
i,j = 71, 110

plt.figure(figsize=(10, 5))
plt.suptitle("Spaxel ({},{})".format(i, j))
plt.subplot(211)
plt.plot(wl, snr_pred[:, i, j], label='Predicted', c='r', lw=0.5)
plt.plot(wl, snr[:, i, j], label='First light', c='k', lw=0.5)
# plt.yscale('log')
plt.ylim(np.nanpercentile(snr[:, i, j], [1, 99]))
plt.legend()
plt.ylabel('SNR/pix')

plt.subplot(212)
plt.plot(wl, snr_pred[:, i, j] / snr[:, i, j], c='k', lw=0.3)
plt.ylim(0.5, 1.5)
plt.axhline(np.nanmedian(snr_pred[:, i, j] / snr[:, i, j]), c='c',
            label='Median')
plt.ylabel('Ratio')
plt.legend()
plt.savefig("ETC_first_light_test_{}_spx_{}_{}.png".format(inst, i, j),
            bbox_inches='tight')