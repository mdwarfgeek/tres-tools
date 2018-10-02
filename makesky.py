import numpy

from poly import *

def makesky(wave, flux, deg):
  # x is normalised [-1, 1] wavelength.
  wavemin = wave[0]
  wavemax = wave[-1]

  x = (2*wave - (wavemin + wavemax)) / (wavemax - wavemin)

  # Remove NaNs.
  ww = numpy.isfinite(flux)

  # Legendre fit.
  coef = cliplegfit(x[ww], flux[ww], deg)

  return numpy.polynomial.legendre.legval(x, coef)
