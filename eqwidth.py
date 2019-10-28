import math
import numpy

# Should really do this using the original polynomial wavelength solution,
# where available.

def subpixel(wave):
  # Wavelength interval of each pixel in spectrum.
  # delta_x = x_i+1 - x_i-1
  # do by shifting input array.
  # Extend to ends (where we have no information).
  width = numpy.empty_like(wave)

  npix = len(wave)
  
  width[1:npix-1] = 0.5*(wave[2:npix] - wave[0:npix-2])
  
  width[0] = width[1]
  width[npix-1] = width[npix-2]
  
  # Pixel start and end wavelengths.
  awave = wave - 0.5*width
  bwave = wave + 0.5*width

  return awave, bwave, width

# Equivalent width, using feature window bin_awave-bin_bwave
# and externally computed (e.g. average or median, below)
# continuum level cont (both flambda).  See above routine for
# conversion from pixel centres (normal wavelength array) to
# start, end and width for each pixel.

def eqwidth_sum(awave, bwave, width, flux, e_flux,
                cont, e_cont,
                bin_awave, bin_bwave):
  # Whole pixels.
  ww = numpy.logical_and(awave >= bin_awave,
                         bwave <= bin_bwave)

  ss = numpy.sum((flux[ww]-cont)*width[ww])
  sv = numpy.sum((e_flux[ww]*width[ww])**2)
  sl = numpy.sum(width[ww])

  npix = len(flux)

  # Partial pixels.
  pp, = numpy.where(ww)
  
  if len(pp) < 1:
    return None

  ii = pp[0]-1
  if ii >= 0:
    dl = (bwave[ii] - bin_awave)
    ss += (flux[ii]-cont) * dl
    sv += (e_flux[ii] * dl)**2
    sl += dl
  else:
    return None

  jj = pp[-1]+1
  if jj < npix and jj != ii:
    dl = (bin_bwave - awave[jj])
    ss += (flux[jj]-cont) * dl
    sv += (e_flux[ii] * dl)**2
    sl += dl
  else:
    return None

  return ss, sv, sl

def eqwidth(awave, bwave, width, flux, e_flux,
            cont, e_cont,
            bin_awave, bin_bwave):

  ss, sv, sl = eqwidth_sum(awave, bwave, width, flux, e_flux,
                           cont, e_cont,
                           bin_awave, bin_bwave)

  # Equivalent width.
  ew = ss / cont

  # Relative variance in sum and continuum.
  rvss = sv / (ss*ss)
  rvcont = (e_cont / cont)**2

  # Equivalent width uncertainty.
  e_ew = abs(ew) * math.sqrt(rvss + rvcont)

  return ew, e_ew

# Standard 2-window constant continuum equivalent width.

def eqwidth_2win(awave, bwave, width, flux, e_flux,
                 feat_awave, feat_bwave,
                 cont1_awave, cont1_bwave,
                 cont2_awave, cont2_bwave):

  cont1, e_cont1 = average(awave, bwave, width, flux, e_flux,
                           cont1_awave, cont1_bwave)

  cont2, e_cont2 = average(awave, bwave, width, flux, e_flux,
                           cont2_awave, cont2_bwave)

  cont = 0.5*(cont1+cont2)
  e_cont = 0.5*math.hypot(e_cont1, e_cont2)

  return eqwidth(awave, bwave, width, flux, e_flux,
                 cont*contcorr, e_cont*contcorr,
                 feat_awave, feat_bwave)

# Mean flux in feature window bin_awave-bin_bwave.

def average(awave, bwave, width, flux, e_flux, bin_awave, bin_bwave):
  # Whole pixels.
  ww = numpy.logical_and(awave >= bin_awave,
                         bwave <= bin_bwave)

  ss = numpy.sum(flux[ww])
  sv = numpy.sum(e_flux[ww]**2)
  nn = numpy.sum(ww)

  npix = len(flux)

  # Partial pixels.
  pp, = numpy.where(ww)
  
  ii = pp[0]-1
  if ii >= 0:
    frac = (bwave[ii] - bin_awave) / width[ii]
    ss += flux[ii] * frac
    sv += (e_flux[ii] * frac)**2
    nn += frac

  jj = pp[-1]+1
  if jj < npix and jj != ii:
    frac = (bin_bwave - awave[jj]) / width[jj]
    ss += flux[jj] * frac
    sv += (e_flux[jj] * frac)**2
    nn += frac

  m = ss / nn
  e_m = numpy.sqrt(sv) / nn

  return m, e_m

# Median flux over feature window bin_awave-bin_bwave.

def median(awave, bwave, width, flux, bin_awave, bin_bwave):
  # Whole pixels.
  ww = numpy.logical_and(awave >= bin_awave,
                         bwave <= bin_bwave)

  return numpy.median(flux[ww])
