import lfa
import math
import numpy
import re

# Import and set up astropy.io.fits or pyfits.  In order to read
# IRAF-style wavelength solutions properly, it needs to be configured
# not to strip header whitespace, which is done by setting the
# variable pyfits.conf.strip_header_whitespace to False.
try:
  import astropy.io.fits as pyfits
except ImportError:
  import pyfits

pyfits.conf.strip_header_whitespace = False

from multispec import *

def espresso_obs():
  # ESO VLT, avg. of 4 UTs, from web page.
  longitude  = -254634.1 * lfa.AS_TO_RAD 
  latitude   = -253456.6 * lfa.AS_TO_RAD
  height     = 2635.43

  obs = lfa.observer(longitude, latitude, height)

  return obs

def espresso_read_bary(hdr, obs=None, src=None):
  # Exposure time.
  exptime = hdr["EXPTIME"]

  # Barycentric date and their vrad corrections.
  bjd = float(hdr["ESO QC BJD"]) - lfa.ZMJD  # UTC
  bvel = float(hdr["ESO QC BERV"])  # km/s

  zb = bvel * 1000 / lfa.LIGHT

  return bjd, zb, exptime

# Use file "ES_S2DA_*.fits" (target, separate orders in original bins)

def espresso_read(thefile, obs=None, src=None):
  if isinstance(thefile, pyfits.HDUList):
    fp = thefile
  else:
    fp = pyfits.open(thefile)

  # Header.
  pri_hdr = fp[0].header
  hdr = fp["SCIDATA"].header

  # Read spectrum, converting to double.
  wave = fp["WAVEDATA_AIR_BARY"].data.astype(numpy.double)
  flux = fp["SCIDATA"].data.astype(numpy.double)
  e_flux = fp["ERRDATA"].data.astype(numpy.double)

  nord, nwave = flux.shape

  # Barycentric correction.
  bjd, zb, exptime = espresso_read_bary(pri_hdr, obs, src)

  # Take it out of wavelengths to match other instruments.
  wave /= (1.0 + zb)

  blaze = None  # for now

  return bjd, zb, exptime, wave, flux, e_flux, blaze

def espresso_orders():
  # UNFINISHED
  return 153, [153]

def espresso_qvalues():
  return [ 0 ]
