import fitsio
import glob
import lfa
import numpy
import os
import re
import string
import warnings

import bary

from multispec import *

# Return observer structure needed for rest.  It could be recreated
# each time inside the read routine but this is quite inefficient.
# The FITS headers are not used because they can be wrong or not
# very accurate.

def tres_obs():
  # 60", from web page.
  longitude  = -399163.5 * lfa.AS_TO_RAD  # -110 52 43.5
  latitude   =  114051.2 * lfa.AS_TO_RAD  #  +31 40 51.2
  height     =  2345.0

  obs = lfa.observer(longitude, latitude, height)

  return obs

def tres_find_blaze(utc):
  # Longitude as fraction of full circle / day.
  longmjd = -399163.5 / (3600.0 * 360.0)

  # Local time as MJD.
  locmjd = utc + longmjd

  # MJD number of previous mid-day.
  midday = int(locmjd-0.5)

  # Convert to Gregorian.
  yr, mn, dy = lfa.mjd2date(midday)

  # Formatted string as used by pipeline.
  nightstr = "{0:04d}-{1:02d}-{2:02d}".format(yr, mn, dy)

  # Try to locate it.
  path = os.path.join("/home/tres/tred/", nightstr, "repackBlaze")

  gg = glob.glob(os.path.join(path, "*.blaze.spec.fits"))
  
  if len(gg) > 0:
    return gg[0]

  # Use generic one included in git repo.
  gg = glob.glob(os.path.join(os.path.dirname(__file__),
                              "blaze", "tres", "*.blaze.spec.fits"))

  if len(gg) > 0:
    return gg[0]

  return None

def tres_read(thefile, obs=None, src=None):
  if isinstance(thefile, fitsio.FITS):
    fp = thefile
  else:
    fp = fitsio.FITS(thefile)

  mp = fp[0]

  # Header.
  hdr = mp.read_header()

  # Time stamp information.
  dateobs = hdr["DATE-OBS"]
  utopen = hdr["UTOPEN"]
  utend = hdr["UTEND"]

  exptime = float(hdr["EXPTIME"])

  m = re.match(r"^\s*(\d+)\-(\d+)\-(\d+)",
               dateobs)
  yr, mn, dy = m.groups()

  iautc = lfa.date2mjd(int(yr), int(mn), int(dy))

  # Start of exp.
  m = re.match(r"^\s*(\d+)\:(\d+)\:(\d+\.?\d*)\s*$",
               utopen)
  hh, mm, ss = m.groups()

  fautc = ((int(hh)*60 + int(mm)) * 60 + float(ss)) / 86400

  # End of exp.
  m = re.match(r"^\s*(\d+)\:(\d+)\:(\d+\.?\d*)\s*$",
               utend)
  hh, mm, ss = m.groups()

  fbutc = ((int(hh)*60 + int(mm)) * 60 + float(ss)) / 86400

  # Did day roll over?
  if fbutc < fautc:
    ibutc = iautc + 1
  else:
    ibutc = iautc

  # Try to locate blaze.
  blfile = tres_find_blaze(iautc+fautc)

  if obs is not None:
    # Position from FITS header.
    rastr = str(hdr["RA"]).strip()
    destr = str(hdr["DEC"]).strip()
    eqstr = str(hdr["EPOCH"]).strip()

    ra,rv = lfa.base60_to_10(rastr, ":", lfa.UNIT_HR, lfa.UNIT_RAD)
    de,rv = lfa.base60_to_10(destr, ":", lfa.UNIT_DEG, lfa.UNIT_RAD)

    mm = re.search(r'^\s*([jJbB]?)\s*(\d*\.?\d*)', eqstr)
    if mm is None:
      raise RuntimeError("invalid equinox")

    gg = mm.groups()

    if gg[1] == "":
      raise RuntimeError("invalid equinox")

    eqtype = gg[0].upper()
    equinox = float(gg[1])

    if equinox == 0:
      raise RuntimeError("invalid equinox")

    if eqtype == "":
      if equinox > 1984.0:
        eqtype = "J"
      else:
        eqtype = "B"

    if eqtype == "B":
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
    ttmutc = obs.dtai(iautc, fautc) + lfa.DTT

    # TT-UTC at end.
    ttmutcend = obs.dtai(ibutc, fbutc) + lfa.DTT

    # Exposure duration in days.
    expdur = (ibutc-iautc) + (fbutc-fautc) + (ttmutcend-ttmutc) / 86400.0

    # Mid-exposure correction in days.
    fmidexp = 0.5*expdur
    
    # Compute time-dependent quantities.
    # Add exp time in here - don't want it included for TT-UTC.
    mjdutc = iautc+fautc+fmidexp
    obs.update(mjdutc, ttmutc, lfa.OBSERVER_UPDATE_ALL)
    
    # Compute current BCRS position.
    (s, dsdt, pr) = obs.place(src, lfa.TR_MOTION)

    # Check it against the header for large errors.
    sep = lfa.v_angle_v(s, vec_icrs)

    if abs(sep) > 300.0 * lfa.AS_TO_RAD:
      sra, sde = lfa.v_to_ad(s)
      if sra < 0:
        sra += lfa.TWOPI

      warnings.warn("large coordinate error: cat {0:s} {1:s} vs file {2:s} {3:s} {4:s}".format(lfa.base10_to_60(sra, lfa.UNIT_RAD, ":", "", 3, lfa.UNIT_HR), lfa.base10_to_60(sde, lfa.UNIT_RAD, ":", "+", 2, lfa.UNIT_DEG), rastr, destr, eqstr), stacklevel=2)

    # Delay
    delay = obs.bary_delay(s, pr)

    # BJD(TDB)
    bjd = mjdutc + (ttmutc + obs.dtdb + delay) / lfa.DAY

#    # Here's how to do BJD(UTC) if needed
#    bjd = mjdutc + delay / lfa.DAY

    # Doppler
    zb = obs.bary_doppler(src, s, dsdt, pr)

    # TRES zeropoint jumped by 0.1 km/s when new front end and
    # fiber feed were installed 2010-12-18.  Put in an offset
    # to correct older observations onto the current velocity
    # system.
    if mjdutc < 55547:
      zb -= 0.10 * 1000 / lfa.LIGHT

  else:
    bjd = None
    zb = None

  # Gain and read noise.
  gain = 1.06
  readnois = 4.2
  xwid = 3  # a guess, it actually uses profile weighted extraction

  # Read spectrum, converting to double.
  flux = mp.read().astype(numpy.double)

  # Trim off extra dimensions (if extracted using IRAF).
  if flux.ndim > 2:
    flux = flux[1,:,:]

  # Compute wavelengths.
  nord, nwave = flux.shape

  wave = multispec_lambda(hdr, nord, nwave)

  # Compute uncertainties.
  e_flux = numpy.sqrt(numpy.where(flux > 0, flux, 0)*gain + xwid*readnois*readnois) / gain

  # Read blaze.
  if blfile is not None:
    blfp = fitsio.FITS(blfile)
    blmp = blfp[0]

    blaze = blmp.read().astype(numpy.double)

    # Trim off extra dimensions (if extracted using IRAF).
    if blaze.ndim > 2:
      blaze = blaze[1,:,:]
  else:
    blaze = None

  return bjd, zb, exptime, wave, flux, e_flux, blaze

def tres_orders():
  return 41, [36, 38, 39, 41, 43, 45]

def tres_qvalues():
  return [9537, 12339, 13102, 15989, 4486, 9352]

