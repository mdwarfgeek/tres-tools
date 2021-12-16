import numpy
import os
import string
import warnings

import fitsio
import lfa

from finterp import *

def getcoords(src_str):
  # Convert RA, DEC.  Try : first and then space.
  ss = src_str

  ra, rv = lfa.base60_to_10(ss, ':', lfa.UNIT_HR, lfa.UNIT_RAD)
  if rv < 0:
    ra, rv = lfa.base60_to_10(ss, ' ', lfa.UNIT_HR, lfa.UNIT_RAD)
    if rv < 0:
      raise RuntimeError("could not understand radec: " + src_str)
    else:
      ss = ss[rv:]
      de, rv = lfa.base60_to_10(ss, ' ', lfa.UNIT_DEG, lfa.UNIT_RAD)
      if rv < 0:
        raise RuntimeError("could not understand radec: " + src_str)
      else:
        ss = ss[rv:]
  else:
    ss = ss[rv:]
    de, rv = lfa.base60_to_10(ss, ':', lfa.UNIT_DEG, lfa.UNIT_RAD)
    if rv < 0:
      raise RuntimeError("could not understand radec: " + src_str)
    else:
      ss = ss[rv:]

  # Attempt to split the rest.
  ll = ss.strip().split()

  pmra = 0
  pmde = 0
  plx = 0
  cat_vrad = 0
  cat_epoch = 2000.0

  if len(ll) > 0:
    pmra = float(ll[0])
  if len(ll) > 1:
    pmde = float(ll[1])
  if len(ll) > 2:
    plx = float(ll[2])
  if len(ll) > 3:
    cat_vrad = float(ll[3])
  if len(ll) > 4:
    cat_epoch = float(ll[4])

  src = lfa.source(ra, de, pmra, pmde, plx, cat_vrad, cat_epoch)

  return src

def stripname(filespec):
  l = filespec.split(",")
  s = l[0].strip()

  if len(s) > 1 and s[0] == "@":
    filename = s[1:]
  else:
    filename = s

  basefile, fileext = os.path.splitext(os.path.basename(filename))

  return basefile

# Class to store the spectrum.  This is actually just a generic
# "structure" initialized from the given contents.
class spec:
  def __init__(self, **entries):
    self.__dict__.update(entries)

# Class to read spectra.
class read_spec:
  def __init__(self):
    self.obs = None
    self.read = None
    self.specname = None
    self.singleorder = None
    self.multiorder = None
    self.overrideorder = None
    self.qvalues = None
    self.srclist = None

    if "TRES_TOOLS_SRCLIST" in os.environ:
      # Use environment variable.
      srcfile = os.environ["TRES_TOOLS_SRCLIST"]
      self.read_srclist(srcfile)
    else:
      # Try to find one by walking back up the filesystem from
      # the current directory.
      here = os.path.realpath(os.getcwd())
      srcfile = None

      while True:
        tstfile = os.path.join(here, "source.cat")
        if os.path.exists(tstfile):
          srcfile = tstfile
          break

        head, tail = os.path.split(here)

        # Normal exit condition when we reach filesystem root.
        if tail == "":
          break

        # This shouldn't happen but just in case to prevent
        # infinite loops we check for them here.
        if here == head:
          break

        here = head
        
      if srcfile is not None:
        self.read_srclist(srcfile)
      else:
        warnings.warn("no source list")

  def read_srclist(self, catfile):
    self.srclist = {}

    with open(catfile, "r") as fp:
      for line in fp:
        # Remove comments and trim white space.
        ll = line.split("#", 1)
        ls = ll[0].strip()

        if ls == "":
          continue

        # Extract object name.
        lo = ls.split(None, 1)
        if len(lo) != 2:
          raise RuntimeError("could not understand: " + line)

        objname, rest = lo

        key = objname.upper()
        src = getcoords(rest)

        self.srclist[key] = src

  def read_spec(self, filespec, src=None, src_str=None, istmpl=False, wantstruct=False, doreject=True):
    # Convert file specification from script argument into a file list.
    # The string "filespec" can be a single item or comma-separated
    # list of items.  Each item can be a filename or @list to read an
    # ASCII list of input files, one per line.
    filelist = []

    for s in filespec.split(","):
      thisspec = s.strip()

      if len(thisspec) > 1 and thisspec[0] == "@":  # @list
        listfile = thisspec[1:]
        
        with open(listfile) as fp:
          filelist.extend(list(map(string.strip, fp)))
      else:
        filelist.append(thisspec)

    nspec = len(filelist)

    # Extract source structure from srclist information, if given.
    if src is None and src_str is not None:
      src = getcoords(src_str)

    # Use first file for setup.
    fp = fitsio.FITS(filelist[0], 'r')

    mp = fp[0]
    hdr = mp.read_header()
    
    # Object name.
    if "OBJECT" in hdr:
      objname = str(hdr["OBJECT"]).strip()

      # If we have one, and we don't have coords for this source yet,
      # try to look it up in the source list.
      if src is None and self.srclist is not None:
        key = objname.upper()

        if key in self.srclist:
          src = self.srclist[key]
        else:
          warnings.warn("object " + objname + " not found in srclist")
    else:
      objname = None

    # Try to detect the instrument.
    specname = None
    mode = None

    if "DETECTOR" in hdr and hdr["DETECTOR"].strip() == "tres":
      specname = "tres"
    elif "FPA" in hdr and hdr["FPA"].strip() == "CHIRON":
      specname = "chiron"
      if "MODE" in hdr:
        mode = int(hdr['MODE'])
      else:
        mode = 0
    elif "INSTRUME" in hdr:
      val = hdr["INSTRUME"].strip()
      if val == "HARPS":
        specname = "harps"
      elif val == "MIKE-Red":
        specname = "mike"
      elif val == "IGRINS":
        specname = "igrins"
      elif val == "ELODIE":
        specname = "elodie"
      elif val == "FEROS":
        specname = "feros"
      elif val == "CARMENES":
        specname = "carmenes"

    if specname is None:
      raise RuntimeError("unrecognised instrument")

    if self.specname is None:
      # Import that instrument's library.
      m = __import__(specname)

      self.specname = specname

      # Create an observer instance.
      self.obs = eval("m."+specname + "_obs()")

      # Extract reference to read routine.
      self.read = eval("m."+specname + "_read")

      # Get order list.
      op = eval("m."+specname + "_orders")

      if mode is None:
        self.singleorder, self.multiorder = op()
      else:
        self.singleorder, self.multiorder = op(mode)

      if self.overrideorder is not None:
        self.singleorder = self.overrideorder

      # Q values.
      try:
        self.qvalues = eval("m."+specname + "_qvalues()")
      except:
        self.qvalues = None

    elif specname != self.specname:
      raise RuntimeError("files seem to be from different instruments: previous {0:s} current file {2:s} instrument {3:s}".format(self.specname, filename, specname))

    if istmpl:
      if not "VELOCITY" in hdr:
        warnings.warn("assuming template velocity = 0")
        vrad = 0
      else:
        vrad = float(hdr["VELOCITY"])
    else:
      vrad = 0.0

    mbjd, zb, exptime, wave, flux, e_flux, blaze = self.read(fp, self.obs, src)

    if nspec > 1:
      # Init lists, using the one we already have (reference).
      mbjdlist = numpy.empty([nspec])
      exptimelist = numpy.empty([nspec])
      wtlist = numpy.empty([nspec])

      fluxlist = [None] * nspec
      e_fluxlist = [None] * nspec
      normedlist = [None] * nspec
      e_normedlist = [None] * nspec

      mbjdlist[0] = mbjd
      exptimelist[0] = exptime

      normord = self.singleorder-1
      thisflux = flux[normord,:]
      wtlist[0] = numpy.median(thisflux[numpy.isfinite(thisflux)])

      fluxlist[0] = flux
      e_fluxlist[0] = e_flux
      normedlist[0] = flux
      e_normedlist[0] = e_flux

      nord = wave.shape[0]

      # Read all spectra and interpolate.
      # XXX - blaze currently ignored.
      for i, filename in enumerate(filelist[1:]):
        thismbjd, thiszb, thisexptime, thiswave, thisflux, thise_flux, thisblaze = self.read(filename, self.obs, src)

        mbjdlist[i+1] = thismbjd
        exptimelist[i+1] = thisexptime

        fluxout = numpy.empty_like(flux)
        e_fluxout = numpy.empty_like(flux)

        wtout = None

        for iord in range(nord):
          interp_flux, interp_e_flux = finterp(wave[iord,:], thiswave[iord,:]*(1.0+thiszb)/(1.0+zb), thisflux[iord,:], thise_flux[iord,:])

          fluxout[iord,:] = interp_flux
          e_fluxout[iord,:] = interp_e_flux

          if iord == normord:
            wtout = numpy.median(interp_flux[numpy.isfinite(interp_flux)])
            wtlist[i+1] = wtout

        fluxlist[i+1] = fluxout
        e_fluxlist[i+1] = e_fluxout
        normedlist[i+1] = fluxout * wtlist[0] / wtout
        e_normedlist[i+1] = e_fluxout * wtlist[0] / wtout

      fluxlist = numpy.array(fluxlist)
      e_fluxlist = numpy.array(e_fluxlist)
      normedlist = numpy.array(normedlist)
      e_normedlist = numpy.array(e_normedlist)

      if doreject:
        # Attempt to locate +ve outliers using median and uncertainties.
        # Individual spectra are normalized to the first one using counts
        # in reference order to reduce effect of exposure time variations.
        mednormed = numpy.median(normedlist, axis=0)
        mederrnormed = numpy.median(e_normedlist, axis=0)

        # Per-pixel weights for final combine (0 or 1).
        # Assumes error in median = median error, i.e. we get no noise
        # improvement from taking the median.  This isn't true and
        # becomes increasingly pessimistic as N gets larger, but typically
        # we expect this to be used for very small N where it's important
        # to account for the error in the median and this can't be done
        # empirically.  Cosmics we need to reject are usually very large
        # deviations so I think it should still work okay.
        combmask = normedlist - mednormed < 5*numpy.hypot(e_normedlist, mederrnormed)
      else:
        combmask = numpy.ones_like(fluxlist, dtype=numpy.bool)

      # Weighted mean but scale back to equivalent of sum with no rejects.
      swt = numpy.sum(combmask, axis=0)

      norm = numpy.empty_like(swt, dtype=numpy.double)
      norm.fill(numpy.nan)

      norm[swt > 0] = float(nspec) / swt[swt > 0]

      flux = numpy.sum(fluxlist*combmask, axis=0) * norm
      ssq = numpy.sum((e_fluxlist**2)*combmask, axis=0) * norm

      # Final uncertainty in sum = quadrature sum of uncertainties.
      e_flux = numpy.sqrt(ssq)

      # Combine other quantities.
      wtlist /= numpy.sum(wtlist)

      mbjd = numpy.average(mbjdlist, weights=wtlist)
      exptime = numpy.sum(exptimelist)

    # Make mask.
    msk = numpy.ones_like(wave, dtype=numpy.bool)

    # a band (O2)
    msk[numpy.logical_and(wave >= 6270, wave < 6330)] = 0

    # B band (O2)
    msk[numpy.logical_and(wave >= 6860, wave < 6965)] = 0

    # H2O
    msk[numpy.logical_and(wave >= 6965, wave < 7030)] = 0

    # H2O
    msk[numpy.logical_and(wave >= 7165, wave < 7340)] = 0

    # A band (O2)
    msk[numpy.logical_and(wave >= 7585, wave < 7705)] = 0

    # Barycentric correction.
    wave *= (1.0 + zb)

    # Take off RV for templates.
    wave /= (1.0 + vrad * 1000 / lfa.LIGHT)

    # Keep track of how much adjustment was applied.
    vbcv = (lfa.LIGHT * zb / 1000)

    # Result.
    if wantstruct:
      sp = spec(mbjd=mbjd,
                wave=wave,
                flux=flux,
                e_flux=e_flux,
                msk=msk,
                blaze=blaze,
                vbcv=vbcv,
                vrad=vrad,
                objname=objname,
                specname=specname,
                exptime=exptime,
                istmpl=istmpl)

      return sp
    else:
      return mbjd, wave, flux, e_flux, msk, blaze, vbcv, vrad

