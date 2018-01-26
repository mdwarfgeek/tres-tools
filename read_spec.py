import numpy
import os
import string
import sys

import fitsio
import lfa

from finterp import *

def stripname(filespec):
  l = filespec.split(",")
  s = l[0].strip()

  if len(s) > 1 and s[0] == "@":
    filename = s[1:]
  else:
    filename = s

  basefile, fileext = os.path.splitext(os.path.basename(filename))

  return basefile

class read_spec:
  def __init__(self):
    self.obs = None
    self.read = None
    self.specname = None
    self.singleorder = None
    self.multiorder = None
    self.overrideorder = None

  def read_spec(self, filespec, istmpl=False):
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
          filelist.extend(map(string.strip, fp))
      else:
        filelist.append(thisspec)

    nspec = len(filelist)

    # Use first file for setup.
    fp = fitsio.FITS(filelist[0], 'r')

    mp = fp[0]
    hdr = mp.read_header()
    
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

    if specname is None:
      print >>sys.stderr, "Unrecognised instrument"
      sys.exit(1)

    if self.specname is None:
      # Import that instrument's library.
      m = __import__(specname)

#      print >>sys.stderr, "instrument =", specname

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

    elif specname != self.specname:
      print >>sys.stderr, "Files seem to be from different instruments"
      print >>sys.stderr, "Previous file was", self.specname
      print >>sys.stderr, "Current file", filename, "is", specname
      sys.exit(1)

    if istmpl:
      if not "VELOCITY" in hdr:
        print >>sys.stderr, "ERROR: please set velocity for template"
        sys.exit(1)
        
      vrad = float(hdr["VELOCITY"])
    else:
      vrad = 0.0

    mbjd, zb, exptime, wave, flux, e_flux, blaze = self.read(fp, self.obs)

    if nspec > 1:
      # Init lists, using the one we already have (reference).
      mbjdlist = numpy.empty([nspec])
      zblist = numpy.empty([nspec])
      exptimelist = numpy.empty([nspec])
      wtlist = numpy.empty([nspec])

      mbjdlist[0] = mbjd
      zblist[0] = zb
      exptimelist[0] = exptime

      normord = self.singleorder-1
      wtlist[0] = numpy.median(flux[normord,:])

      # Sum of squares of uncertainties.
      ssq = numpy.zeros_like(e_flux)
      ssq += e_flux*e_flux

      nord = wave.shape[0]

      # Read all spectra and combine.
      # XXX - blaze currently ignored.
      for i, filename in enumerate(filelist[1:]):
        thismbjd, thiszb, thisexptime, thiswave, thisflux, thise_flux, thisblaze = self.read(filename, self.obs)

        mbjdlist[i] = thismbjd
        zblist[i] = thiszb
        exptimelist[i] = thisexptime

        for iord in range(nord):
          interp_flux, interp_e_flux = finterp(wave[iord,:], thiswave[iord,:], thisflux[iord,:], thise_flux[iord,:])

          flux[iord,:] += interp_flux
          ssq[iord,:] += interp_e_flux**2

          if iord == normord:
            wtlist[i] = numpy.median(interp_flux)

      # Final uncertainty in sum = quadrature sum of uncertainties.
      e_flux = numpy.sqrt(ssq)

      # Combine other quantities.
      wtlist /= numpy.sum(wtlist)

      mbjd = numpy.average(mbjdlist, weights=wtlist)
      zb = numpy.average(zblist, weights=wtlist)
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

    return mbjd, wave, flux, e_flux, msk, blaze, vbcv, vrad

