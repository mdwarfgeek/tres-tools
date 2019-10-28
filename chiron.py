import fitsio
import lfa
import numpy
import re
import string

import matplotlib.pyplot as plt

from multispec import *
from scombine import *

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
    if src is None:
      # Use telescope position from FITS header.  NOT RECOMMENDED!
      rastr = hdr["RA"]
      destr = hdr["DEC"]

      ra,rv = lfa.base60_to_10(rastr, ":", lfa.UNIT_HR, lfa.UNIT_RAD)
      de,rv = lfa.base60_to_10(destr, ":", lfa.UNIT_DEG, lfa.UNIT_RAD)

      src = lfa.source(ra, de)

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

      blaze = im[:,:,5]
      blaze /= numpy.median(numpy.isfinite(blaze))  # normalize

    else:  # Yale
      wave = im[:,:,0]
      flux = im[:,:,1]

      e_flux = numpy.sqrt(numpy.where(flux > 0, flux, 0) + xwid*readnois*readnois) / gain

      blaze = None  # for now

  else:  # iraf format (converted)
    nord, nwave = im.shape

    wave = multispec_lambda(hdr, nord, nwave)

    flux = im

    e_flux = numpy.sqrt(numpy.where(flux > 0, flux, 0) + xwid*readnois*readnois) / gain

    blaze = None  # for now

  return bjd, zb, exptime, wave, flux, e_flux, blaze

def chiron_orders(imode):
  if imode == 1:  # slicer, best orders 44, 51 (equiv. 41, 45 on TRES)
    return 44, [36, 37, 39, 40, 44, 51]
  elif imode == 3:  # fiber
    return 46, [38, 39, 41, 42, 46, 53]
  elif imode == 0:  # ch_reduce, order layout consistent but more of them
    return 59, [51, 52, 54, 55, 59, 66]

def chiron_qvalues():
  return [20656, 18365, 22139, 27416, 30755, 23709]
