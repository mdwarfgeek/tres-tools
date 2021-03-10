import fitsio
import lfa
import numpy
import os
import re
import string
import warnings

from multispec import *

# Return observer structure needed for rest.  It could be recreated
# each time inside the read routine but this is quite inefficient.
# The FITS headers are not used because they can be wrong or not
# very accurate on some instruments.

def chiron_obs():
  # CTIO 1.5m, from Mamajek 2012.
  longitude  = -254904.44 * lfa.AS_TO_RAD  # -70 48 24.44
  latitude   = -108609.42 * lfa.AS_TO_RAD  # -30 10 09.42
  height     = 2252.0

  obs = lfa.observer(longitude, latitude, height)

  return obs

def chiron_read(thefile, obs=None, src=None):
  if isinstance(thefile, fitsio.FITS):
    fp = thefile
  else:
    fp = fitsio.FITS(thefile)

  mp = fp[0]

  # Header.
  hdr = mp.read_header()

  # Time stamp: use shutter open time and exposure time.
  # Exposure meter is not reliable for faint targets so
  # we don't use it.
  utshut = hdr['UTSHUT']
  exptime = float(hdr['EXPTIME'])

  m = re.match(r"^\s*(\d+)\-(\d+)\-(\d+)[Tt](\d+)\:(\d+)\:(\d+\.?\d*)\s*$",
               utshut)
  yr, mn, dy, hh, mm, ss = m.groups()

  # Start of exp.
  iutc = lfa.date2mjd(int(yr), int(mn), int(dy))
  futc = ((int(hh)*60 + int(mm)) * 60 + float(ss)) / 86400

  # Mid-exp corrn in days.
  fmidexp = exptime * 0.5 / lfa.DAY

  if obs is not None:
    # Telescope position from FITS header.
    rastr = str(hdr["RA"]).strip()
    destr = str(hdr["DEC"]).strip()
    equinox = float(hdr["EPOCH"])

    ra,rv = lfa.base60_to_10(rastr, ":", lfa.UNIT_HR, lfa.UNIT_RAD)
    de,rv = lfa.base60_to_10(destr, ":", lfa.UNIT_DEG, lfa.UNIT_RAD)

    if equinox == 0:
      raise RuntimeError("invalid equinox")

    if equinox < 1984.0:
      raise NotImplementedError("pre-FK5 equinoxes are not supported")

    # Precession matrix ICRS to given equinox.
    pfb = lfa.pfb_matrix(lfa.J2K + (equinox-2000.0) * lfa.JYR)

    # Apply inverse transformation to star coordinates.
    vec_eq = lfa.ad_to_v(ra, de)
    vec_icrs = numpy.dot(pfb.T, vec_eq)  # noting transpose

    # Null velocity vector.
    vel_icrs = numpy.zeros([3])

    # If no override given
    if src is None:
      # Use position from FITS header.  NOT RECOMMENDED!
      src = lfa.source_star_vec(vec_icrs, vel_icrs)

    # TT-UTC at start.
    ttmutc = obs.dtai(iutc, futc) + lfa.DTT
    
    # Compute time-dependent quantities.
    # Add exp time in here - don't want it included for TT-UTC.
    mjdutc = iutc+futc+fmidexp
    obs.update(mjdutc, ttmutc, lfa.OBSERVER_UPDATE_ALL)

    # Compute current BCRS position.
    (s, dsdt, pr) = obs.place(src, lfa.TR_MOTION)

    # Check it against the header for large errors.
    sep = lfa.v_angle_v(s, vec_icrs)

    if abs(sep) > 300.0 * lfa.AS_TO_RAD:
      sra, sde = lfa.v_to_ad(s)
      if sra < 0:
        sra += lfa.TWOPI

      warnings.warn("large coordinate error: cat {0:s} {1:s} vs file {2:s} {3:s} {4:.1f}".format(lfa.base10_to_60(sra, lfa.UNIT_RAD, ":", "", 3, lfa.UNIT_HR), lfa.base10_to_60(sde, lfa.UNIT_RAD, ":", "+", 2, lfa.UNIT_DEG), rastr, destr, equinox), stacklevel=2)

    # Delay
    delay = obs.bary_delay(s, pr)

    # BJD(TDB)
    bjd = mjdutc + (ttmutc + obs.dtdb + delay) / lfa.DAY

    # Doppler
    zb = obs.bary_doppler(src, s, dsdt, pr)
  else:
    bjd = None
    zb = None

  # Gain and read noise.
  if "GAIN" in hdr:  # Yale pipeline
    gain = float(hdr['GAIN'])
    readnois = float(hdr['RON'])
  else:  # Use amp 1 (all the same anyway).
    gain = float(hdr['GAIN11'])
    readnois = float(hdr['RON11'])

  # Get extraction window size.
  if "MODE" in hdr:
    imode = int(hdr['MODE'])
    xwids = hdr['XWIDS']

    if isinstance(xwids, basestring):
      xwids = xwids.split(',')

    xwid = float(xwids[imode])
  else:
    mode = hdr['DECKER'].strip()

    xwid = None

  # Read spectrum, converting to double.
  im = mp.read().astype(numpy.double)

  if im.ndim == 3:  # chiron format
    nord, nwave, nvec = im.shape

    if nvec > 2:  # ch_reduce
      wave = im[:,:,0]
      flux = im[:,:,4]
      snr = im[:,:,2]

      e_flux = numpy.zeros_like(flux)

      ww = numpy.logical_and(numpy.isfinite(snr), snr > 0)
      e_flux[ww] = flux[ww] / snr[ww]

      blraw = im[:,:,5]

      # Normalize it to unit median in each order.
      blmed = numpy.median(blraw, axis=1)
      blnorm = numpy.where(blmed > 0, 1.0/blmed, 0.0)

      blaze = blraw * blnorm[:,numpy.newaxis]

    else:  # Yale
      wave = im[:,:,0]
      flux = im[:,:,1]  # spectrum is in electrons

      e_flux = numpy.sqrt(numpy.where(flux > 0, flux, 0) + xwid*readnois*readnois)

      blaze = None  # for now

  else:  # iraf format (converted)
    nord, nwave = im.shape

    wave = multispec_lambda(hdr, nord, nwave)

    flux = im

    e_flux = numpy.sqrt(numpy.where(flux > 0, flux, 0) + xwid*readnois*readnois)

    blaze = None  # for now

  if blaze is None:
    # Attempt to find extracted flat.  For now, we just use one from
    # the template spectrum night shipped as part of the git repo.
    if imode == 1:  # slicer
      blbase = "chi180421.slicerflat.fits"
    elif imode == 3:  # fiber
      blbase = "chi180421.fiberflat.fits"

    blfile = os.path.join(os.path.dirname(__file__),
                          "blaze", "chiron",
                          blbase)

    if os.path.exists(blfile):
      blfp = fitsio.FITS(blfile)
      blmp = blfp[0]
      
      blimg = blmp.read().astype(numpy.double)

      # Flat function used to normalize it is in the third plane of the
      # file, but both axes are reversed, so correct that.
      blraw = blimg[2,::-1,::-1]

      # Normalize.
      blmed = numpy.median(blraw, axis=1)
      blnorm = numpy.where(blmed > 0, 1.0/blmed, 0.0)

      blaze = blraw * blnorm[:,numpy.newaxis]

  return bjd, zb, exptime, wave, flux, e_flux, blaze

def chiron_orders(imode):
  if imode == 1:  # slicer, best orders 44, 51 (equiv. 41, 45 on TRES)
    return 44, [36, 37, 39, 40, 44, 51]
  elif imode == 3:  # fiber
    return 46, [38, 39, 41, 42, 46, 53]
  elif imode == 0:  # ch_reduce, order layout consistent but more of them
    return 59, [51, 52, 54, 55, 59, 66]

def chiron_qvalues():
  return [18917, 16752, 20230, 25058, 28237, 19470]
