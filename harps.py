import fitsio
import lfa
import numpy
import re

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

def harps_decode_packed_sexagesimal(thestr):
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
  if "ESO DRS DRIFT VR" in hdr:
    zth = float(hdr["ESO DRS DRIFT VR"]) / lfa.LIGHT
  else:
    zth = 0

  # Replace Barycentric correction if observer structure is given.
  if obs is not None:
    if src is None:
      # Decode the packed sexagesimal used for target coordinates in
      # FITS header.  Requires dirty trick to force reading as a string,
      # otherwise fitsio converts automatically to a float, which could
      # corrupt it, given that the float conversion assumes the number
      # is base-10 (it's not).
      rastr = hdr._record_map["ESO TEL TARG ALPHA"]["value_orig"].strip()
      destr = hdr._record_map["ESO TEL TARG DELTA"]["value_orig"].strip()

      ra = harps_decode_packed_sexagesimal(rastr) * lfa.SEC_TO_RAD
      de = harps_decode_packed_sexagesimal(destr) * lfa.AS_TO_RAD

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
  if isinstance(thefile, fitsio.FITS):
    fp = thefile
  else:
    fp = fitsio.FITS(thefile)

  mp = fp[0]

  # Header.
  hdr = mp.read_header()

  # Read spectrum, converting to double.
  flux = mp.read().astype(numpy.double)

  nord, nwave = flux.shape

  # Barycentric correction.
  bjd, zb, exptime = harps_read_bary(hdr, obs, src)

  # Detector parameters.
  readnois = float(hdr["ESO DRS CCD SIGDET"])
  gain = float(hdr["ESO DRS CCD CONAD"])  # NB already applied

  # Wavelength calibration is stored as a polynomial for each order.
  # degree is HIERARCH ESO DRS CAL TH DEG LL
  # coeffs are HIERARCH ESO DRS CAL TH COEFF LL
  degree = int(hdr["ESO DRS CAL TH DEG LL"])
  ncoef = degree+1

  wave = numpy.empty_like(flux, dtype=numpy.double)

  x = numpy.arange(0, nwave, dtype=numpy.double)  # XXX - numbers from zero?
  coef = numpy.empty([ncoef])

  for iord in range(nord):
    for i in range(ncoef):
      key = "ESO DRS CAL TH COEFF LL{0:d}".format(i+iord*ncoef)
      coef[i] = float(hdr[key])

    wave[iord,:] = numpy.polynomial.polynomial.polyval(x, coef)

  # Uncertainties.  Fluxes already in e-.
  e_flux = numpy.sqrt(numpy.where(flux > 0, flux, 0) + readnois*readnois)

  blaze = None  # for now

  return bjd, zb, exptime, wave, flux, e_flux, blaze

# Use file "s1d_A.fits" (target, combined, rebinned)

def harps_read_s1d(thefile, obs=None, src=None):
  if isinstance(thefile, fitsio.FITS):
    fp = thefile
  else:
    fp = fitsio.FITS(thefile)

  mp = fp[0]

  # Header.
  hdr = mp.read_header()

  # Read spectrum, converting to double.
  flux = mp.read().astype(numpy.double)

  nwave, = flux.shape
  nord = 1

  # Barycentric correction.
  bjd, zb, exptime = harps_read_bary(hdr, obs, src)

  # Detector parameters.
  readnois = float(hdr["ESO DRS CCD SIGDET"])
  gain = float(hdr["ESO DRS CCD CONAD"])  # NB already applied

  wave = multispec_lambda(hdr, nord, nwave)
  wave = wave[0,:]

  # Uncertainties.  Fluxes already in e-.
  e_flux = numpy.sqrt(numpy.where(flux > 0, flux, 0) + readnois*readnois)

  blaze = None  # for now

  return bjd, zb, exptime, wave, flux, e_flux, blaze

def harps_orders():
  # UNFINISHED
#  return 71, [63, 65, 66, 67, 69, 70, 71]
  return 71, [47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
              59, 60, 61, 62, 63, 65, 66, 67, 69, 70, 71]
