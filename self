#!/usr/bin/env python

import argparse
import math
import os
import re
import sys

import numpy

import lfa

# Stop pyplot trying to use X.
import matplotlib
matplotlib.use('Agg')

import fftrv

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from scipy.interpolate import InterpolatedUnivariateSpline

from medsig import *

from lsd import *
from makesky import *
from prepord import *
from read_spec import *

# Constants used by the script go here.

# Figure size based on pgplot.
figsize = (10.5, 7.8)  # inches

# Velocity range to plot
velrange = 150  # km/s, plots +/- this much

# Regularization constant for LSDs.  Depends on template and possibly
# also on SNR, so may need changing.
kreg = 100

def do_vrad(pdf, tmplname, filename,
            tmpl_mbjd, tmpl_wave, tmpl_flux, tmpl_e_flux, tmpl_msk,
            mbjd, wave, flux, e_flux, msk, order, emchop=True):

  # Extract order and clean.
  thistmpl_wave, thistmpl_flux, thistmpl_e_flux = prepord(order, tmpl_wave, tmpl_flux, tmpl_e_flux, tmpl_msk)
  thiswave, thisflux, thise_flux = prepord(order, wave, flux, e_flux, msk)

  # Take off sky.
  thistmpl_rawflux = numpy.copy(thistmpl_flux)

  ss = makesky(thistmpl_wave, thistmpl_flux, 4)

  thistmpl_flux -= ss

  thisrawflux = numpy.copy(thisflux)

  ss = makesky(thiswave, thisflux, 4)

  thisflux -= ss

  tmpl_emmask = numpy.isfinite(thistmpl_flux)
  emmask = numpy.isfinite(thisflux)

  if emchop:
    medflux, sigflux = medsig(thistmpl_flux[tmpl_emmask])
    tmpl_emmask = numpy.logical_and(tmpl_emmask,
                                    thistmpl_flux < medflux + 5.0*sigflux)
    
    medflux, sigflux = medsig(thisflux[emmask])
    emmask = numpy.logical_and(emmask,
                               thisflux < medflux + 5.0*sigflux)

  # Number of measurements (pixels).
  npix = len(thisflux)

  # Correlate.
  frv = fftrv.fftrv(nbin=32*npix, t_emchop=emchop, s_emchop=emchop)

  z, corr, zbest, hbest, sigt = frv.correlate(thistmpl_wave, thistmpl_flux,
                                              thiswave, thisflux)

  vels = z * lfa.LIGHT / 1000
  vbest = zbest * lfa.LIGHT / 1000

  ww = numpy.abs(vels-vbest) < velrange

  fig = plt.figure(figsize=figsize)

  yofftarg = max(thistmpl_rawflux[tmpl_emmask])

  plt.subplot(2, 1, 1)
  plt.plot(thistmpl_wave, thistmpl_rawflux)
  plt.plot(thiswave, thisrawflux+yofftarg)
  plt.xlim(thiswave[0], thiswave[-1])

  yl = numpy.min(thistmpl_rawflux[tmpl_emmask])
  yh = numpy.max(thisrawflux[emmask]) + yofftarg
  plt.ylim(yl-0.05*(yh-yl), yh+0.05*(yh-yl))

  plt.xlabel("Wavelength (A)")
  plt.ylabel("Counts")
  plt.title("Aperture {0:d}".format(order))
  
  xl, xh = plt.xlim()
  yl, yh = plt.ylim()

  levmin = xh - 0.25*(xh-xl)
  levmax = xh

  wwlev = numpy.logical_and(thistmpl_wave >= levmin,
                            thistmpl_wave <= levmax)
  wwlev = numpy.logical_and(wwlev, tmpl_emmask)

  ytmplmed, ytmplsig = medsig(thistmpl_rawflux[wwlev])
  ytmpl = ytmplmed+3*ytmplsig

  wwlev = numpy.logical_and(thiswave >= levmin,
                            thiswave <= levmax)
  wwlev = numpy.logical_and(wwlev, emmask)

  ytargmed, ytargsig = medsig(thisrawflux[wwlev])
  ytarg = ytargmed+3*ytargsig

  plt.text(xh-0.02*(xh-xl), ytmpl+0.02*(yh-yl),
           tmplname,
           horizontalalignment='right', size='small')

  plt.text(xh-0.02*(xh-xl), yofftarg+ytarg+0.02*(yh-yl),
           filename,
           horizontalalignment='right', size='small')

  plt.subplot(2, 1, 2)
  plt.plot(vels[ww], corr[ww])
  plt.xlim(vels[ww][0], vels[ww][-1])
  plt.xlabel("Velocity (km/s)")
  plt.ylabel("Correlation")
  
  xl, xh = plt.xlim()
  yl, yh = plt.ylim()
  
  plt.text(xh-0.02*(xh-xl), yh-0.1*(yh-yl),
             "Delta RV = {0:.3f} km/s".format(vbest),
             horizontalalignment='right')
  plt.text(xh-0.02*(xh-xl), yh-0.2*(yh-yl),
           "h = {0:.3f}".format(hbest),
           horizontalalignment='right')

  plt.axvline(vbest)
  
  pdf.savefig(fig)
  plt.close()

  return vbest, hbest

def do_lsd(pdf, filename,
           tmpl_mbjd, tmpl_wave, tmpl_flux, tmpl_e_flux, tmpl_msk,
           mbjd, wave, flux, e_flux, msk,
           vrad, orders, savefile=False, emchop=True):
  # Compute at 0.5 km/s intervals.
  vl = vrad-velrange
  vh = vrad+velrange
  nv =  801

  vels, prof = lsd_multiorder(tmpl_wave, tmpl_flux, tmpl_e_flux, tmpl_msk,
                              wave, flux, e_flux, msk,
                              orders,
                              vl, vh, nv,
                              kreg, emchop=emchop)

  # Optionally save LSD to file.
  if savefile:
    basefile = stripname(filename)
    lsdfile = basefile + "_diff_lsd.txt"

    with open(lsdfile, "w") as lfp:
      for imeas, thisvel in enumerate(vels):
        lfp.write(str(thisvel) + " " + str(prof[imeas]) + "\n")

  # Plot.
  fig = plt.figure(figsize=figsize)

  plt.plot(vels, prof)

  plt.xlim(vels[0], vels[-1])

  plt.xlabel("Velocity (km/s)".format(vrad))
  plt.ylabel("Normalized flux")
  plt.title("{0:s} differential LSD".format(filename))

  pdf.savefig(fig)
  plt.close()

def do_multi_vrad(pdf, tmplname, filename,
                  tmpl_mbjd, tmpl_wave, tmpl_flux, tmpl_e_flux, tmpl_msk,
                  mbjd, wave, flux, e_flux, msk,
                  orders, emchop=True):
  l_vrad = numpy.empty_like(orders, dtype=numpy.double)
  l_corr = numpy.empty_like(orders, dtype=numpy.double)

  for ii, order in enumerate(orders):
    # Extract order and clean.
    thistmpl_wave, thistmpl_flux, thistmpl_e_flux = prepord(order, tmpl_wave, tmpl_flux, tmpl_e_flux, tmpl_msk)
    thiswave, thisflux, thise_flux = prepord(order, wave, flux, e_flux, msk)

    # Take off sky.
    ss = makesky(thistmpl_wave, thistmpl_flux, 4)
    
    thistmpl_flux -= ss
    
    ss = makesky(thiswave, thisflux, 4)
    
    thisflux -= ss

    # Number of measurements (pixels).
    npix = len(thisflux)

    # Correlate.
    frv = fftrv.fftrv(nbin=32*npix, t_emchop=emchop, s_emchop=emchop)

    z, corr, zbest, hbest, sigt = frv.correlate(thistmpl_wave, thistmpl_flux,
                                                thiswave, thisflux)

    vels = z * lfa.LIGHT / 1000
    vbest = zbest * lfa.LIGHT / 1000

    l_vrad[ii] = vbest
    l_corr[ii] = hbest

  mean_vrad = numpy.mean(l_vrad)
  sig_vrad = numpy.std(l_vrad)

  if len(l_vrad) > 1:
    e_mean_vrad = sig_vrad / math.sqrt(len(l_vrad)-1)
  else:
    e_mean_vrad = 0

  fig = plt.figure(figsize=figsize)

  plt.subplot(2, 1, 1)

  plt.xlim(orders[0]-1, orders[-1]+1)
  plt.fill_between(plt.xlim(),
                   mean_vrad-sig_vrad, mean_vrad+sig_vrad,
                   color='grey', alpha=0.25)

  plt.plot(orders, l_vrad, 'o', color='black')
  plt.axhline(mean_vrad, color='black')

  plt.xlabel("Aperture number")
  plt.ylabel("Velocity (km/s)")
  plt.title("Multi-order results")

  str = ""

  for ii, order in enumerate(orders):
    str += "Ap {0:3d}  dRV = {1:7.4f} km/s  h = {2:.4f}\n".format(order, l_vrad[ii], l_corr[ii])

  str += "\n"

  str += "RMS = {0:.4f} km/s\n".format(sig_vrad)

  str += "\n"

  str += "Mean dRV = {0:.4f} +/- {1:.4f} km/s\n".format(mean_vrad, e_mean_vrad)

  fig.text(0.1, 0.1, str)

  pdf.savefig(fig)
  plt.close()

  return mean_vrad, e_mean_vrad

ap = argparse.ArgumentParser()
ap.add_argument("template", help="template spectrum file or @list of files to be stacked")
ap.add_argument("filelist", metavar="file", nargs="+", help="spectrum file or @list of files to be stacked")
ap.add_argument("-c", metavar="ra dec pmra pmdec plx vrad epoch", type=str, help="specify catalogue information for target")
ap.add_argument("-E", action="store_true", help="don't remove emission lines from spectrum")
ap.add_argument("-o", type=int, help="override order number used for analysis")
ap.add_argument("-R", action="store_true", help="don't use cosmic rejection when stacking")
args = ap.parse_args()

# New read_spec structure.
rs = read_spec()

if args.o is not None:
  rs.overrideorder = args.o

emchop = not args.E

# Read epoch to use as template.
tmplsp = rs.read_spec(args.template, src_str=args.c, wantstruct=True, doreject=not args.R)

tmplname = re.sub(r'_cb\.spec$', "", stripname(args.template))

filelist = args.filelist
nf = len(args.filelist)

for ifile, filename in enumerate(filelist):
  sp = rs.read_spec(filename, src_str=args.c, wantstruct=True, doreject=not args.R)

  targname = re.sub(r'_cb\.spec$', "", stripname(filename))

  basefile = stripname(filename)
  outfile = basefile + "_self.pdf"

  pdf = PdfPages(outfile)

  # Velocity.
  vrad, hbest = do_vrad(pdf, tmplname, targname,
                        tmplsp.mbjd, tmplsp.wave, tmplsp.flux, tmplsp.e_flux, tmplsp.msk,
                        sp.mbjd, sp.wave, sp.flux, sp.e_flux, sp.msk,
                        order=rs.singleorder, emchop=emchop)

  multiorders = rs.multiorder
  if args.o is not None:
    multiorders = [ args.o ]

  do_lsd(pdf, filename,
         tmplsp.mbjd, tmplsp.wave, tmplsp.flux, tmplsp.e_flux, tmplsp.msk,
         sp.mbjd, sp.wave, sp.flux, sp.e_flux, sp.msk,
         vrad,
         orders=multiorders, emchop=emchop)

  mean_vrad, e_mean_vrad = do_multi_vrad(pdf, tmplname, targname,
                                         tmplsp.mbjd, tmplsp.wave, tmplsp.flux, tmplsp.e_flux, tmplsp.msk,
                                         sp.mbjd, sp.wave, sp.flux, sp.e_flux, sp.msk,
                                         orders=multiorders, emchop=emchop)

  pdf.close()

  print "{0:s} {1:12.4f} {2:9.4f} {3:7.4f} {4:7.2f}".format(filename, lfa.ZMJD+sp.mbjd, mean_vrad, e_mean_vrad, sp.exptime)

sys.exit(0)

