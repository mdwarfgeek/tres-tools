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

# Return observer structure needed for rest.  It could be recreated
# each time inside the read routine but this is quite inefficient.
# The FITS headers are not used because they can be wrong or not
# very accurate on some instruments.

# Order 45 seems to be missing in the "B" fibre.

def harps_obs():
  # ESO 3.6m, from La Silla web page.
  longitude  = -254634.1 * lfa.AS_TO_RAD  # -70 43 54.1
  latitude   = -105339.5 * lfa.AS_TO_RAD  # -29 15 39.5
  height     = 2400.0

  obs = lfa.observer(longitude, latitude, height)

  return obs

def harps_decode_packed_sexagesimal(thecard):
  # Extract the keyword value.
  mm = re.match(r'^[^\=]+\=\s*([\+\-]?\d*\.?\d*)\s*/?.*$', thecard.image)
  if mm is not None:
    thestr = mm.groups()[0]
  else:
    thestr = str(thecard.value)

  # Deal with sign first to prevent "-0" problems.
  isneg = 0

  if thestr[0] == '-':
    thestr = thestr[1:]
    isneg = 1
  elif thestr[0] == '+':
    thestr = thestr[1:]

  # Chop at decimal point (if there is one).
  ss = thestr.split(".")

  # Convert to hours, minutes, seconds.
  angle = 0

  if ss[0] != "":
    tmp = int(ss[0])

    ih = tmp // 10000
    tmp -= ih * 10000
    im = tmp // 100
    tmp -= im * 100

    angle = (ih * 60 + im) * 60 + tmp

  # Deal with fraction.
  if len(ss) > 1 and ss[1] != "":
    tmp = int(ss[1])

    scl = 1

    for i in range(len(ss[1])):
      scl *= 10

    angle += tmp / float(scl)

  if isneg:
    angle *= -1

  return angle

def harps_read_bary(hdr, obs=None, src=None):
  # Exposure time.
  exptime = hdr["ESO DET WIN1 DIT1"]

  # Simultaneous thorium drift correction.  (NOT YET USED)
  if "ESO DRS DRIFT SPE RV" in hdr:
    zth = float(hdr["ESO DRS DRIFT SPE RV"]) / lfa.LIGHT
  else:
    zth = 0

  # Replace Barycentric correction if observer structure is given.
  if obs is not None:
    if src is None:
      # Decode the broken packed sexagesimal format used for telescope
      # target coordinates in ESO FITS headers.  This encoding looks
      # like a base-10 float to pyfits, but it isn't and must be read
      # as a string.  To do this we retrieve the raw card image and
      # parse it ourselves.
      racrd = hdr.cards["ESO TEL TARG ALPHA"]
      decrd = hdr.cards["ESO TEL TARG DELTA"]

      ra = harps_decode_packed_sexagesimal(racrd) * lfa.SEC_TO_RAD
      de = harps_decode_packed_sexagesimal(decrd) * lfa.AS_TO_RAD

      # Read other quantities (which are already in the right units).
      pma = float(hdr["ESO TEL TARG PMA"])
      pmd = float(hdr["ESO TEL TARG PMD"])
      plx = float(hdr["ESO TEL TARG PARALLAX"])
      vrad = float(hdr["ESO TEL TARG RADVEL"])
      ep = float(hdr["ESO TEL TARG EPOCH"])

      # Check coordinate type is something we understand.
      ctype = hdr["ESO TEL TARG COORDTYPE"].strip()
      epsys = hdr["ESO TEL TARG EPOCHSYSTEM"].strip()
      eq = float(hdr["ESO TEL TARG EQUINOX"])

      if ctype != "M" or epsys != "J" or eq != 2000.0:
        raise RuntimeError("don't know how to deal with {0:s} {1:s} {2:f}".format(ctype, epsys, eq))

      # Make source structure.
      src = lfa.source(ra, de, pma, pmd, plx, vrad, ep)

    dateobs = hdr["DATE-OBS"]

    expcorr = hdr["ESO INS DET1 TMMEAN"]  # science fibre exp meter

    m = re.match(r"^\s*(\d+)\-(\d+)\-(\d+)[Tt](\d+)\:(\d+)\:(\d+\.?\d*)\s*$",
                 dateobs)
    yr, mn, dy, hh, mm, ss = m.groups()

    # Start of exp.
    iutc = lfa.date2mjd(int(yr), int(mn), int(dy))
    futc = ((int(hh)*60 + int(mm)) * 60 + float(ss)) / lfa.DAY
    
    # Mid-exp corrn in days.
    fmidexp = exptime * expcorr / lfa.DAY
    
    # TT-UTC at start.
    ttmutc = obs.dtai(iutc, futc) + lfa.DTT
  
    # Compute time-dependent quantities.
    # Add exp time in here - don't want it included for TT-UTC.
    mjdutc = iutc+futc+fmidexp
    obs.update(mjdutc, ttmutc, lfa.OBSERVER_UPDATE_ALL)

    # Compute current BCRS position.
    (s, dsdt, pr) = obs.place(src, lfa.TR_MOTION)

    # Delay
    delay = obs.bary_delay(s, pr)

    # BJD(TDB)
    bjd = mjdutc + (ttmutc + obs.dtdb + delay) / lfa.DAY

    # Doppler
    zb = obs.bary_doppler(src, s, dsdt, pr)
  else:
    # Barycentric date and their vrad corrections.
    bjd = float(hdr["ESO DRS BJD"]) - lfa.ZMJD  # UTC
    bvel = float(hdr["ESO DRS BERV"])  # km/s

    zb = bvel * 1000 / lfa.LIGHT

  return bjd, zb, exptime

# Use file "e2ds_A.fits" (target, separate orders in original bins)

def harps_read(thefile, obs=None, src=None):
  if isinstance(thefile, pyfits.HDUList):
    fp = thefile
  else:
    fp = pyfits.open(thefile)

  mp = fp[0]

  # Header.
  hdr = mp.header

  # Read spectrum, converting to double.
  flux = mp.data.astype(numpy.double)

  nord, nwave = flux.shape

  # Barycentric correction.
  bjd, zb, exptime = harps_read_bary(hdr, obs, src)

  # Detector parameters.
  readnois = float(hdr["ESO DRS CCD SIGDET"])
  gain = float(hdr["ESO DRS CCD CONAD"])  # NB already applied

  # Wavelength calibration is stored as a polynomial for each order.
  # degree is HIERARCH ESO DRS CAL TH DEG LL
  # coeffs are HIERARCH ESO DRS CAL TH COEFF LL
  wdegree = int(hdr["ESO DRS CAL TH DEG LL"])
  wncoef = wdegree+1

  # Also retrieve spatial FWHM of orders for estimating read noise.
  fdegree = int(hdr["ESO DRS CAL LOC DEG FWHM"])
  fncoef = fdegree+1

  wave = numpy.empty_like(flux, dtype=numpy.double)
  fwhm = numpy.empty_like(flux, dtype=numpy.double)

  x = numpy.arange(0, nwave, dtype=numpy.double)  # XXX - numbers from zero?
  wcoef = numpy.empty([wncoef])
  fcoef = numpy.empty([fncoef])

  for iord in range(nord):
    for i in range(wncoef):
      key = "ESO DRS CAL TH COEFF LL{0:d}".format(i+iord*wncoef)
      wcoef[i] = float(hdr[key])

    wave[iord,:] = numpy.polynomial.polynomial.polyval(x, wcoef)

    for i in range(fncoef):
      key = "ESO DRS CAL LOC FWHM{0:d}".format(i+iord*fncoef)
      fcoef[i] = float(hdr[key])

    fwhm[iord,:] = numpy.polynomial.polynomial.polyval(x, fcoef)

  # FWHM of red orders is underestimated in pipeline for reasons
  # still unknown to me (suspicions are saturation or possibly
  # contamination from the adjacent fibre if they use the more
  # numerous variety of flats where both are illuminated).  They
  # are never less than 3.1 pixels based on my measurements of
  # the flats, so clamp there.
  fwhm = numpy.where(fwhm > 3.1, fwhm, 3.1)

  # Estimate variance for profile weighted extraction assuming
  # the profile was a Gaussian of given FWHM from above.  As
  # far as I can tell based on comments in the change log, this
  # version of the pipeline is using the flat for the weights,
  # but we don't have it and these Gaussians seem to be a decent
  # approximation to the flats based on my tests.  We sum over
  # a window of 25 pixels which is a bit larger than the one I
  # think the pipeline uses but it shouldn't make any difference
  # given the Gaussian weights are very close to zero for all
  # the extra pixels.
  sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
  norm = math.sqrt(2.0 * math.pi) * sigma

  pvar = numpy.where(flux > 0, flux, 0)  # Poisson variance

  s1 = numpy.zeros_like(flux)
  s2 = numpy.zeros_like(flux)
  for ipix in range(25):
    pix = ipix - 12

    pwt = numpy.exp(-0.5*(pix/sigma)**2) / norm

    s1 += pwt
    s2 += pwt*pwt / (pwt*pvar + readnois*readnois)

  e_flux = numpy.sqrt(s1 / s2)

  blaze = None  # for now

  return bjd, zb, exptime, wave, flux, e_flux, blaze

# Use file "s1d_A.fits" (target, combined, rebinned)

def harps_read_s1d(thefile, obs=None, src=None):
  if isinstance(thefile, pyfits.HDUList):
    fp = thefile
  else:
    fp = pyfits.open(thefile)

  mp = fp[0]

  # Header.
  hdr = mp.header

  # Read spectrum, converting to double.
  flux = mp.data.astype(numpy.double)

  nwave, = flux.shape
  nord = 1

  # Barycentric correction.
  bjd, zb, exptime = harps_read_bary(hdr, obs, src)

  # Detector parameters.
  readnois = float(hdr["ESO DRS CCD SIGDET"])
  gain = float(hdr["ESO DRS CCD CONAD"])  # NB already applied
  xwid = 3.1  # a guess, it actually uses profile weighted extraction

  wave = multispec_lambda(hdr, nord, nwave)
  wave = wave[0,:]

  # Uncertainties.  Fluxes already in e-.
  # Spectrum has been renormalized so this won't be correct.
  e_flux = numpy.sqrt(numpy.where(flux > 0, flux, 0) + xwid*readnois*readnois)

  blaze = None  # for now

  return bjd, zb, exptime, wave, flux, e_flux, blaze

def harps_orders():
  # UNFINISHED
#  return 71, [63, 65, 66, 67, 69, 70, 71]
  return 71, [47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
              59, 60, 61, 62, 63, 65, 66, 67, 69, 70, 71]

def harps_qvalues():
  return [ 18157, 16724, 37496, 29679, 21310, 28757, 20969, 16950, 18586, 21980,
           21167, 16613, 12848, 30510, 28067, 28320, 25595, 23399, 27813, 37338, 32964 ]
