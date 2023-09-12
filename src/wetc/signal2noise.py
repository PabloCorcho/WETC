# signal2noise.py
#
# Scott C. Trager, based on "signal.f" by Chris Benn
# driving code for "signalWEAVE"
#
# v1.5 15 Feb 2018: added "totalSeeing" to account for blurring by PRI
# v1.4 15 Jan 2018: fixed major bug in Moffat profile normalization
# v1.3 12 Jan 2018: stripped out "blue" and "red" arguements, as no longer needed
# v1.2 19 Jun 2013: now allows for Moffat or Gaussian seeing profile
# v1.1 20 Nov 2011: now calculates dispersion self-consistently based on
#                   resolution and spectrograph parameters
#
# Parameter updates
#
# UPDATE 17.04.2012: central obstruction 1.5m -> 1.7m
# UPDATE 15.10.2012: PFC efficiency dropped to 0.70 in blue arm and 0.75 in red arm
# UPDATE 22.11.2012: added full-well depth (fullwell) of 265000 e- per Iain S's message of 30.10.2012
# UPDATE 02.12.2012: new PFC and spectrograph average efficiencies -
#                    blue  PFC: 0.6944  red  PFC: 0.7310
#                    blue spec: 0.6484  red spec: 0.6698
#                    new CCD efficiencies: blue=0.8612 red=0.9293
#                    new fibre efficiencies: blue=0.7887 red=0.8680
# UPDATE 11.12.2013: central obstruction 1.7m -> 1.83m
# UPDATE 30.01.2014: new efficiencies (blue: 400-600nm)
#                    blue  PFC: 0.7400  red  PFC: 0.7564 low res
#                    blue spec: 0.5394  red spec: 0.6168 low res
#                    blue  fib: 0.7242  red  fib: 0.8014 low res
#                    blue  CCD: 0.9310  red  CCD: 0.9269 low res
#                    blue  PFC: 0.7300  red  PFC: 0.7500 high res
#                    blue spec: 0.4192  red spec: 0.4597 high res
#                    blue  fib: 0.6600  red  fib: 0.7900 high res
#                    blue  CCD: 0.9500  red  CCD: 0.9300 high res
# defaults below are RED SIDE, LOW-RES efficiencies
# Some new functionalities added by P. Corcho-Caballero.
# TODO
# class signal renamed to Signal in order to match python standards.
#

import numpy as np
from scipy.integrate import quad, dblquad
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import os
import pandas

PRI_FWHM = 0.45
planck_cgs = 6.6261e-27  # erg s
c_linght_nm = 3e17  # nm/s
c_linght_ang = c_linght_nm * 10
jansky_to_cgs = 1e-23


def mag_to_flux(mags, zero, cgs=True):
    """Convert magnitudes to fluxes."""
    f = (10 ** (-0.4 * mags) * zero)
    if cgs:
        f *= jansky_to_cgs
    return f

def totalSeeing(seeing):
    """..."""
    return np.sqrt(seeing**2 + PRI_FWHM**2)


class FullWellException(Exception):
    """..."""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        """..."""
        return repr(self.value)


class SaturationException(Exception):
    """..."""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        """..."""
        return repr(self.value)


class Signal:
    """WEAVE Exposure Time Calculator.

    Attributes
    ----------
    - bands: (list) available photometric bands to estimate SNR for models
    based on single photometric values.
    - extinction_mag: (array) default atmospheric extinction function
    see D. L. King 1985.
    - extinction_wl: (array) extinction wavelength array.
    - sky_model: (bool, default=True) Set to True for building a sky model interpolating several photometric bands
    - inc_moon: (bool, default=False) Wheter to include moon birghtness on sky model.
    - moon_phase: (str, default='new') Moon phase to include on sky model.
    - sky_roque_mag: (array) default sky brightness model (Benn & Ellison 1999, https://arxiv.org/abs/astro-ph/9909153).
    - sky_roque_zero: (array) default sky model photometric zero point.s
    - sky_roque_flux: (array) default sky model flux density in
    erg/s/Hz/cm^2/arcsec^2.
    - telarea: (float) Primary mirror effective collector area (m^2).
    - fcam:
    - fcol:
    - LIFU: (bool, default=False) Switch to True to include LIFU fiber size and
    spectral resolution.
    - m1Eff: (narray) Primary mirror reflectivity efficiency.
    - QE: (narray) Detector quantum efficiency.
    - fiberEff:
    - specEff:
    - pfEff:
    - obscEff:
    - waveEff: (array) Efficiency wavelength array.

    Methods
    -------
    - sky_photon_model
    - build_sky_model
    - dispersion
    - efficiency
    - effectiveArea
    - extinction
    - lightfrac_old (deprecated method)
    - ligth_frac
    - gaussian_xy
    - moffat_xy
    - objectphotons
    - skyphotons
    - S2N
    - S2N_spectra
    - RVaccuracy
    - time4S2N
    - eff4timeS2N
    - mag4timeS2N
    - get_band_transmission
    - get_band_central_wl
    - get_band_photon_flux
    - get_spectra_photon_flux
    - get_extinction
    - get_throughput_table
    - build_sky_model

    Example
    -------

    """

    # Available photometric bands
    bands = ['U', 'B', 'V', 'R', 'I', 'u', 'g', 'r', 'i', 'z']
    # SKY MODEL
    # Observatory atmospheric extinction
    extinction_wl, extinction_mag = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', 'extinction',
                                                            'extinction_roque'), unpack=True)
    # Sky brightness
    sky_roque_mag = np.array([22.0, 22.7, 21.9, 21.0, 20.0])
    sky_roque_moon = {'new': np.array([0, 0, 0, 0, 0]),
                      'crescent': np.array([-0.5, -0.5, -0.5, -0.3, -0.2]),
                      'quarter': np.array([-2.0, -2.0, -2.0, -1.3, -1.1]),
                      'gibbous': np.array([-3.1, -3.1, -3.1, -2.4, -2.2]),
                      'full': np.array([-4.3, -4.3, -4.3, -3.5, -3.3]),
                      }
    sky_roque_wl = np.array([3600.0, 4400.0, 5500.0, 6400.0, 7900.0])
    sky_roque_zero = np.array([1810.0, 4260.0, 3640.0, 3080.0, 2550.0])  # Jy

    def __init__(self,
                 LIFU=False,
                 # Manually input throuput values
                 QE=0.9269, fiberEff=0.8014, specEff=0.6168, pfEff=0.7564,
                 obscEff=1,
                 # Istrument specs
                 res=5000.,
                 offset=0., fiberD=1.3045, fFP=3.2, fcol=3.1,
                 fcam=1.8, R5000fiberD=1.3045, telD=4.2, centralObs=1.83,
                 cwave=4900.,
                 # CCD
                 gain=1., saturation=6.e4, fullwell=2.65e5,
                 # PSF
                 profile='Gaussian', betam=None,
                 # Tabular throughput data
                 mode='LR',
                 throughput_table=False,
                 sky_model=True,
                 inc_moon=False,
                 moon_phase='new'):

        # WHT parameters
        # telescope area after removing central obstruction
        self.telarea = np.pi * ((telD/2)**2 - (centralObs/2)**2)
        self.fcam = fcam
        self.fcol = fcol

        # resolution
        self.res = res
        # fiber diameter in arcsec
        self.fiberD = fiberD
        if LIFU:
            self.res /= 2.
            self.fiberD *= 2.
        if mode == 'HR':
            self.res = 10000.

        if throughput_table:
            self.get_throughput_table(mode)
        else:
            # M1 reflectivity; average blue (390-600): 0.910; average red (600-980): 0.891
            self.m1Eff = np.array([0.90])
            # nominal instrumental parameters
            # detector QE
            self.QE = np.array([QE])
            # fiber efficiency
            self.fiberEff = np.array([fiberEff])
            # spectrograph efficiency
            self.specEff = np.array([specEff])
            # PF corrector efficiency
            self.pfEff = np.array([pfEff])
            # Obscuration
            self.obscEff = np.array([obscEff])
            # Wavelength
            self.waveEff = None

        # Sky brightness model
        self.sky_model = sky_model
        self.inc_moon = inc_moon
        self.moon_phase = moon_phase
        if self.sky_model:
            self.build_sky_model(
                inc_moon=self.inc_moon, moon_phase=self.moon_phase)
        else:
            self.sky_photon_model = None
        # spectrograph/detector parameters

        # fiber diameter in arcsec when R=5000
        self.R5000fiberD = R5000fiberD
        # pixel size in microns
        self.pixSize = 15.
        # fiber diameter in microns
        self.fiberDmu = self.fiberD * fFP * 4.2 * 1e6/206265
        # fiber diameter on CCD in pixels
        self.fiberCCD = self.fiberDmu * (self.fcam/self.fcol)/self.pixSize
        # dispersion at central wavelength, in Angstrom/pixel
        # new 20.11.2011
        self.disp = self.dispersion(cwave=cwave, fcam=self.fcam,
                                    fcol=self.fcol, beamsize=180.)
        # offset of fiber from object center
        self.offset = offset
        # CCD parameters
        self.gain = gain
        self.saturation = saturation
        self.fullwell = fullwell
        # seeing profile
        self.profile = profile
        self.alpha = None
        if self.profile == 'Moffat':
            self.betam = betam
        # print summary
        print('-' * 70 + '\n · Initialising WEAVE Exposure Time Calculator \n' + '-' * 70
              + '\n INPUT PARAMETERS'
              + '\n · LIFU setup: {}'.format(LIFU)
              + '\n · Resolution mode: {}'.format(mode)
              + '\n · Telescope params \n  -> TelArea (m^2): {:.2f},  fcam: {:.1f},  fcol: {:.1f}'.format(
                    self.telarea, self.fcam, self.fcol)
              + '\n · res (R): {:.1f}'.format(self.res)
              + '\n · FiberD (arcsec): {:.1f}'.format(self.fiberD)
              + '\n · Read throughput from table: {}'.format(throughput_table)
              + '\n · CCD \n  -> gain: {:.1f}, saturation (counts): {}, fullwell (counts): {}'.format(
                    self.gain, self.saturation, self.fullwell)
              + '\n · PSF profile: {}'.format(self.profile)
              + '\n · Use sky brightness model: {}'.format(self.sky_model)
              + '\n' + '-' * 70 + '\n\n')

    def dispersion(self, fcam, fcol, beamsize, cwave=None, wave=None):
        """Compute spectral dispersion in Ang/pixel.

        params:
        ------
         - fcam: (float) ...
         - fcol: (float) ...
         - beamsize: (float) ...
         - cwave: (float, optional, default=None) central wavelength of
         instrument mode for computing the spectral dispersion. If provided,
         'wave' will be ignored.
         - wave: (array, optional, default=None) wavelength array to compute
         the spectral dispersion.

        returns: dispersion vector.
        """
        if cwave is not None:
            wl = cwave
        else:
            wl = wave
        # camera focal length
        camfl = beamsize * fcam
        # grating angle in radians
        gamma = np.arctan2(1e-3 * self.fiberDmu * self.res, 2*beamsize * fcol)
        # l/mm of grating
        lpmm = 2 * np.sin(gamma) / (1e-7 * wl)
        lpum = 1e-3 * lpmm
        # FOV of camera on detector (in radians)
        # fovcam=atan2(npix*self.pixSize*1e-3,2*camfl)
        # 1 pixel subtends this number of radians
        pixsubt = self.pixSize * 1e-3 / camfl
        # diffracted angle of one pixel (approximation good to very high precision!)
        diffangle = np.arcsin(np.sin(gamma) - 1e-4 * wl * lpum)
        return 1e4 * pixsubt * np.cos(diffangle) / lpum

    def efficiency(self, wl=None, QE=True, fiber=True, spec=True, PF=True,
                   M1=True, obsc=True):
        """Compute the total throughput efficiency of the instrument.

        params
        ------
        - wl: (array, optional, default=None) wavelength points to compute the 
        throughput efficiency.
        - QE: (bool, optional, default=True) Include detector quantum
        efficiency on throuhput budget.
        - fiber: (bool, optional, default=True) Include fiber
        efficiency on throuhput budget.
        - spec: (bool, optional, default=True) Include spectrograph
        efficiency on throuhput budget.
        - PF: (bool, optional, default=True) Include Prime Focus corrector
        efficiency on throuhput budget.
        - M1: (bool, optional, default=True) Include primary mirror reflectivity
        efficiency on throuhput budget.
        - obsc: (bool, optional, default=True) Include obscuration
        effect on throuhput budget.

        returns: total efficiency vector.
        """
        eff = np.ones_like(self.QE)
        if QE:
            eff *= self.QE
        if fiber:
            eff *= self.fiberEff
        if spec:
            eff *= self.specEff
        if PF:
            eff *= self.pfEff
        if M1:
            eff *= self.m1Eff
        if obsc:
            eff *= self.obscEff
        if len(eff.shape) > 1:
            # Combine blue and red arm throughputs
            eff = np.sum(eff, axis=0)
        if self.waveEff is None:
            return eff
        else:
            return np.interp(wl, self.waveEff, eff, right=0, left=0)

    def effectiveArea(self, eff):
        """..."""
        return eff * self.telarea

    def extinction(self, wl, airmass=1.2):
        """Return the extinction for a given wavelength and airmass."""
        ext = self.get_extinction(wl)
        return pow(10, -0.4 * ext * airmass)

    def gaussian_xy(self, y, x, s):
        """..."""
        r = np.sqrt((x - self.offset)**2 + y*y)
        return np.exp(-pow(r, 2)/(2 * s*s))/(2*np.pi * s*s)

    def moffat_xy(self, y, x, alpha, betam):
        """..."""
        r = np.sqrt((x - self.offset)**2 + y*y)
        # new normalization given by integrating Moffat function in Mathematica (15.01.2018)
        if alpha >= 0. and betam > 1.:
            norm = (2 * np.pi * alpha**2)/(2 * (betam - 1))
        else:
            raise ValueError('alpha and/or beta out of bounds')
        return pow(1. + pow(r/alpha, 2), -betam)/norm

    def lightfrac_old(self, seeing, fiberD):
        """..."""
        # note that the integral of a circularly-symmetric gaussian over 0
        # to 2pi and 0 to infinity is just 2 pi sigma^2...
        s = seeing/2.3548
        rfib = fiberD/2.
        lf = quad(lambda x: x * np.exp(-pow(x-self.offset,2)/(2*s*s))/ \
                  (s*(s+self.offset * np.sqrt(np.pi/2.))), 0, rfib)[0]
        return lf

    def lightfrac(self, seeing, fiberD):
        """..."""
        rfib = fiberD/2.
        if self.profile == 'Gaussian':
            s = seeing/2.3548
            lf = dblquad(self.gaussian_xy, -rfib, rfib,
                         lambda x: - np.sqrt(rfib*rfib - x*x),
                         lambda x: np.sqrt(rfib*rfib - x*x), args=(s,))[0]
        elif self.profile == 'Moffat':
            self.alpha = seeing/(2.*np.sqrt(2.**(1./self.betam)-1))
            lf = dblquad(self.moffat_xy, -rfib, rfib,
                         lambda x: - np.sqrt(rfib*rfib - x*x),
                         lambda x: np.sqrt(rfib*rfib - x*x),
                         args=(self.alpha, self.betam))[0]
        return lf

    def objectphotons(self, time, airmass, effArea, mag=None, band=None,
                      wave=None, spectra=None):
        """Total number of object photons collected per pixel in wavelength direction, integrated over slit."""
        if mag is not None:
            photon_flux = self.get_band_photon_flux(mag, band)
            central_wl = self.get_band_central_wl(band)
            atm_extinction = self.extinction(central_wl, airmass)
        elif spectra is not None:
            photon_flux = self.get_spectra_photon_flux(wave, spectra)
            atm_extinction = self.extinction(wave, airmass)
        n_phot = (photon_flux * time * effArea * self.disp
                  * atm_extinction)
        return n_phot

    def skyphotons(self, time, effArea, fiberD, skysb=None, band=None, wave=None):
        """Total number of sky photons collected per pixel in wavelength direction, integrated over slit."""
        if band is not None:
            if self.sky_model:
                wl = self.get_band_central_wl(band)
                photon_flux = self.sky_photon_model(wl)
            else:
                photon_flux = self.get_band_photon_flux(skysb, band)
        elif wave is not None:
            photon_flux = self.sky_photon_model(wave)
        n_phot = (photon_flux * time * effArea * self.disp
                  * np.pi * pow(fiberD/2., 2))
        return n_phot

    def S2N(self, time, mag, band='B', airmass=1.2, seeing=1.0, rn=3.,
            dark=0., eff=None, sb=None, skysb=22.7,
            skyband="B", n_exposures=1):
        """..."""
        if band != skyband:
            print(' WARNING: Sky band {} is different from Source band {}\n'
                  .format(skyband, band)
                  + '  The number of sky photons might be unreliable')

        # number of pixels along the slit subtended by the fiber
        npix_spatial = self.fiberCCD
        # number of pixels in 1 Angstrom
        npix_spectral = 1./self.disp
        pixArea = npix_spatial * npix_spectral
        wl = self.get_band_central_wl(band)
        if not eff:
            eff = self.efficiency(wl=wl)
        effArea = self.effectiveArea(eff)
        if sb:
            # total number of photons per spectral pixel in the fiber, uniform SB
            ophot = (self.objectphotons(mag=mag, band=band, time=time,
                                        airmass=airmass, effArea=effArea)
                     * np.pi * pow(self.fiberD/2., 2))
        else:
            # total number of photons per spectral pixel in the fiber, circular
            # Gaussian
            ophot = (self.objectphotons(mag=mag, band=band, time=time,
                                        airmass=airmass, effArea=effArea)
                     * self.lightfrac(seeing, self.fiberD))
        # number of photons in the sky in the fiber per pixel
        sphot = self.skyphotons(skysb=skysb, time=time,
                                band=skyband, effArea=effArea,
                                fiberD=self.fiberD)
        # full-well depth exceeded?
        if (sphot+ophot)/self.gain > self.fullwell:
            raise FullWellException('exceeded full well depth')
        # S/N in 1 spectral pixel
        SNRpix = ophot / np.sqrt(ophot + sphot
                                 + npix_spatial * (rn * rn + dark*time/3600.))
        # S/N in 1 Angstrom
        SNR = npix_spectral * ophot / np.sqrt(
            npix_spectral * ophot + npix_spectral * sphot
            + pixArea*(rn*rn+dark*time/3600.))
        # S/N in 1 resolution element
        SNRres = self.fiberCCD * ophot / np.sqrt(
            self.fiberCCD * ophot + self.fiberCCD * sphot
            + self.fiberCCD**2 * (rn**2 + dark * time/3600.))
        # Include multiple exposures during OB
        SNRpix *= np.sqrt(n_exposures)
        SNR *= np.sqrt(n_exposures)
        SNRres *= np.sqrt(n_exposures)
        return {'SNR': SNR, 'objectphotons': ophot, 'skyphotons': sphot,
                'efficiency': eff, 'effectivearea': effArea,
                'SNRres': SNRres, 'SNRpix': SNRpix}

    def S2N_spectra(self, time, wave, spectra, airmass=1.2,
                    seeing=1.0, rn=3., dark=0., eff=None, sb=None,
                    n_exposures=1):
        """..."""
        spectra = spectra.copy()
        # number of pixels along the slit subtended by the fiber
        npix_spatial = self.fiberCCD
        # number of pixels in 1 Angstrom
        self.disp = self.dispersion(wave=wave, fcam=self.fcam,
                                    fcol=self.fcol, beamsize=180.)
        npix_spectral = 1./self.disp
        pixArea = npix_spatial * npix_spectral
        if not eff:
            eff = self.efficiency(wave)
        effArea = self.effectiveArea(eff)
        if sb:
            # total number of photons per spectral pixel in the fiber, uniform SB
            ophot = (self.objectphotons(wave=wave, spectra=spectra,
                                        time=time, airmass=airmass,
                                        effArea=effArea)
                     * np.pi * pow(self.fiberD/2., 2))
        else:
            # total number of photons per spectral pixel in the fiber, circular
            # Gaussian
            ophot = (self.objectphotons(wave=wave, spectra=spectra,
                                        time=time, airmass=airmass,
                                        effArea=effArea)
                     * self.lightfrac(seeing, self.fiberD))
        # number of photons in the sky in the fiber per pixel
        sphot = self.skyphotons(wave=wave, time=time, effArea=effArea,
                                fiberD=self.fiberD)
        # full-well depth exceeded?
        if ((sphot+ophot)/self.gain > self.fullwell).any():
            raise FullWellException(
                'exceeded full well depth at some frecuency')
        # S/N in 1 spectral pixel
        SNRpix = ophot / np.sqrt(ophot + sphot
                                 + npix_spatial * (rn * rn + dark*time/3600.))
        # S/N in 1 Angstrom
        SNR = npix_spectral * ophot / np.sqrt(
            npix_spectral * ophot + npix_spectral * sphot
            + pixArea*(rn*rn+dark*time/3600.))
        # S/N in 1 resolution element
        SNRres = self.fiberCCD * ophot / np.sqrt(
            self.fiberCCD * ophot + self.fiberCCD * sphot
            + self.fiberCCD**2 * (rn**2 + dark * time/3600.))
        # Include multiple exposures during OB
        SNRpix *= np.sqrt(n_exposures)
        SNR *= np.sqrt(n_exposures)
        SNRres *= np.sqrt(n_exposures)
        return {'SNR': SNR, 'objectphotons': ophot, 'skyphotons': sphot,
                'efficiency': eff, 'effectivearea': effArea,
                'SNRres': SNRres, 'SNRpix': SNRpix}

    def RVaccuracy(self, snr, scale=0.6):
        """..."""
        # formula taken from Munari et al. (2001)
        # scaling taken from Battaglia et al. (2008)
        # note that this S/N per Ang!
        return (scale*pow(10, 0.6*pow(np.log10(snr), 2) - 2.4*np.log10(snr)
                          - 1.75*np.log10(self.res) + 9.36))

    def time4S2N(self, S2Ndesired, mag, band, airmass=1.2, fiberD=1.3045,
                 seeing=1.0, rn=3., dark=0., eff=None, sb=None, skysb=22.7,
                 skyband="B", snrtype='SNRres'):
        """..."""
        # determine time required to achieve given S/N ratio S2Ndesired
        # bounds: [0.1,72000] s
        args = (mag, band, airmass, fiberD, seeing, rn, dark, eff, sb, skysb,
                skyband)

        def bfunc(x, s2n, args):
            """..."""
            return self.S2N(x, *args)[snrtype]-s2n
        return brentq(bfunc, 0.1, 7.2e4, (S2Ndesired, args))

    def eff4timeS2N(self, timeDesired, S2Ndesired, mag, band, airmass=1.2,
                    fiberD=1.3045, seeing=1.0, rn=3., dark=0., sb=None,
                    skysb=22.7, skyband="B", snrtype='SNRres'):
        """Determine efficiency required to achieve given S/N ratio S2Ndesired."""
        # determine efficiency required to achieve given S/N ratio S2Ndesired
        # in time timeDesired
        # bounds: [0.01,1.00]
        args = vars()
        del args['self']
        del args['timeDesired']
        del args['S2Ndesired']
        del args['snrtype']
        args['time'] = timeDesired

        def bfunc(x, s2n, args):
            """..."""
            args['eff'] = x
            return self.S2N(**args)[snrtype] - s2n
        return brentq(bfunc, 0.01, 1.0, (S2Ndesired, args))

    def mag4timeS2N(self, S2Ndesired, timeDesired, band, airmass=1.2, eff=None,
                    fiberD=1.3045, seeing=1.0, rn=3., dark=0., sb=None,
                    skysb=22.7, skyband="B", snrtype='SNRres'):
        """..."""
        # determine magnitude reached at given S/N ratio S2Ndesired in time
        # in time timeDesired
        # bounds: [0.01,1.00]
        args = vars()
        del args['self']
        del args['timeDesired']
        del args['S2Ndesired']
        del args['snrtype']
        args['time'] = timeDesired

        def bfunc(x, s2n, args):
            """..."""
            args['mag'] = x
            return self.S2N(**args)[snrtype] - s2n
        return brentq(bfunc, 5, 30, (S2Ndesired, args))

    def get_band_transmission(self, band):
        """Return the corresponding transmision curver of an specific photometric band."""
        path = os.path.join(os.path.dirname(__file__), 'data', 'Filters', band + '.dat')
        try:
            filter_wl, filter_trans = np.loadtxt(path, usecols=(0, 1),
                                                 unpack=True)
        except Exception:
            raise NameError('Band filter {} not found at \n{}'.format(
                band, path))
        return filter_wl, filter_trans

    def get_band_central_wl(self, band):
        """Return central wavelength of photometric band."""
        wl, trans = self.get_band_transmission(band)
        return np.sum(wl * trans) / np.sum(trans)

    def get_band_central_frec(self, band):
        """Return central wavelength of photometric band."""
        wl, trans = self.get_band_transmission(band)
        frec = c_linght_ang / wl
        return np.sum(frec * trans) / np.sum(trans)

    def get_band_photon_flux(self, mag, band):
        """Return the number of photons detected for a given magnitude and band."""
        f_nu = 10**(-0.4 * (mag + 48.60))  # erg/s/Hz/cm2/(arcsec2)
        f_nu *= 10000  # erg/s/Hz/m2/(arcsec2)
        central_wl = self.get_band_central_wl(band)  # Angstrom
        f_lambda = f_nu * c_linght_ang / central_wl**2
        n_photons = f_lambda / (planck_cgs * c_linght_ang / central_wl)  # photons/m^2/s/Ang
        return n_photons

    def get_spectra_photon_flux(self, wave, spectra):
        """Return the number of photons detected for a given spectrum.

        params
        ------
        - wave: (array) wavelength vector of spectra.
        - spectra: (array) flux density per unit wavelength in erg/s/AA/cm2/arcsec2

        returns
        -------
        - n_photons: (array) Flux number density of photons per m^2/s/angstrom
        """
        spectra *= 10000  # erg/s/AA/m2/arcsec2
        n_photons = spectra / (planck_cgs * c_linght_ang / wave
                               )  # photons/m^2/s/Ang
        return n_photons

    def get_extinction(self, wl):
        """Return extinction in mags/airmass for a given wavelength."""
        return np.interp(wl, self.extinction_wl, self.extinction_mag)

    def get_throughput_table(self, mode):
        """Set throughput values to LR of HR modes from tabular data.

        Override default efficiency throughput parameters using those provided by WEAVE-SYS-007. Each parameter
        will consist of a (N, M) matrix, with N indicating the number of windows (blue, red for LR / blue, green, red
        for HR) and M the wavelength axis.

        params
        ------
        - mode: (str) Resolution mode of the instrument (LR/HR).
        """
        print('  -> Loading default throughput tables for {} mode'.format(mode))
        mode_windows = {'LR': ['blue', 'red'], 'HR': ['blue', 'green', 'red']}
        # Read table
        path = os.path.join(os.path.dirname(__file__), 'data', 'throughput', 'throughput_{}.csv'.format(mode))
        df = pandas.read_csv(path)
        # Wavelength
        self.waveEff = np.zeros(0)
        wave_windows = {}
        for window in mode_windows[mode]:
            wave_windows[window] = df['wavelength_{}'.format(window)].values * 10  # To angstrom
            self.waveEff = np.concatenate((self.waveEff, wave_windows[window]))
        self.waveEff = (np.unique(self.waveEff))
        # Detector quantum efficiency
        self.QE = np.zeros((len(mode_windows[mode]), self.waveEff.size))
        # Fiber efficiency
        self.fiberEff = np.zeros((len(mode_windows[mode]), self.waveEff.size))
        # Spectrograph throughput
        self.specEff = np.zeros((len(mode_windows[mode]), self.waveEff.size))
        # Primer focus corrector eff
        self.pfEff = np.zeros((len(mode_windows[mode]), self.waveEff.size))
        # Primary mirror M1 reflectivity
        self.m1Eff = np.zeros((len(mode_windows[mode]), self.waveEff.size))
        # Obscuration
        self.obscEff = np.zeros((len(mode_windows[mode]), self.waveEff.size))
        for i, window in enumerate(mode_windows[mode]):
            self.QE[i] = np.interp(self.waveEff, wave_windows[window], df['QE_{}_opt'.format(window)].values,
                                   left=0, right=0)
            self.fiberEff[i] = np.interp(self.waveEff, wave_windows[window],
                                         df['fibre_throughput_{}_opt'.format(window)].values,
                                         left=0, right=0)
            self.specEff[i] = np.interp(self.waveEff, wave_windows[window],
                                        df['spectrograph_throughput_{}_opt'.format(window)].values,
                                        left=0, right=0)
            self.pfEff[i] = np.interp(self.waveEff, wave_windows[window],
                                      df['prime_focus_corrector_{}_opt'.format(window)].values,
                                      left=0, right=0)
            self.m1Eff[i] = np.interp(self.waveEff, wave_windows[window],
                                      df['M1_{}_opt'.format(window)].values,
                                      left=0, right=0)
            self.obscEff[i] = np.interp(self.waveEff, wave_windows[window],
                                        df['obscuration_{}_opt'.format(window)].values,
                                        left=0, right=0)

    def build_sky_model(self,
                        sky_brightness_mag=sky_roque_mag,
                        sky_brightness_mag_zero=sky_roque_zero,
                        sky_brightness_wl=sky_roque_wl, inc_moon=False, moon_phase='new'):
        """Create an interpolator for sky photon flux (photons/AA/m^2/arcsec^2) from input sky fluxes.

        params
        ------
        - sky_brightness_mags: (narray) Sky brightness.
        - sky_brightness_mag_zero: (narray) Zero point for each photometric point.
        - sky_brightness_wl: (narray) wavelength vector in angstrom.
        - inc_mood: (bool, default=False) Whether to include the contribution of the Moon. This only works for default
        sky magnitudes.
        - moon_phase: (str, default="new") Moon phase to compute the moon brightness contribution. The current available
         phases are "new", "crescent", "quarter", "gibbous" and "full".
        """
        if inc_moon:
            sky_mag = (
                sky_brightness_mag + self.sky_roque_moon[moon_phase])
        else:
            sky_mag = sky_brightness_mag
        sky_brightness_flux = mag_to_flux(
            mags=sky_mag, zero=sky_brightness_mag_zero, cgs=True)
        f_phot = (sky_brightness_flux
                  / (planck_cgs * c_linght_ang / sky_brightness_wl))
        # convert to phot/AA/m^2/arcsec^2
        f_phot *= c_linght_ang / sky_brightness_wl**2 * 1e4
        print("[WETC] Building Sky model\n  Moon phase: {}".format(moon_phase))
        self.sky_photon_model = interp1d(sky_brightness_wl, f_phot,
                                         fill_value='extrapolate')

#  \(ﾟ▽ﾟ)/
