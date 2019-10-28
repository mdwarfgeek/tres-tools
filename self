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

  if pdf is not None:
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

  if pdf is not None:
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
                  orders, qvalues, emchop=True):
  l_vrad = numpy.empty_like(orders, dtype=numpy.double)
  l_e_vrad = numpy.empty_like(orders, dtype=numpy.double)
  l_corr = numpy.empty_like(orders, dtype=numpy.double)

  for ii, order in enumerate(orders):
    # Extract order and clean.
    thistmpl_wave, thistmpl_flux, thistmpl_e_flux = prepord(order, tmpl_wave, tmpl_flux, tmpl_e_flux, tmpl_msk)
    thiswave, thisflux, thise_flux = prepord(order, wave, flux, e_flux, msk)

    emmask = numpy.isfinite(thisflux)
    if emchop:
      medflux, sigflux = medsig(thisflux[emmask])
      emmask = numpy.logical_and(emmask,
                                 thisflux < medflux + 5.0*sigflux)

    if qvalues is not None:
      # Use fixed set of default Q values in case template has low SNR.
      q = qvalues[ii]

      # Estimate of velocity uncertainty using target spectrum SNR.
      ww = numpy.logical_and(emmask, thise_flux > 0)
      snr = thisflux[ww] / thise_flux[ww]

      velrms = lfa.LIGHT / (1000 * q * math.sqrt(numpy.sum(snr*snr)))
    else:
      velrms = 0

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
    l_e_vrad[ii] = velrms
    l_corr[ii] = hbest

  nord = len(l_vrad)

  # Mean of orders, unweighted.
  mean_vrad = numpy.mean(l_vrad)
  sig_vrad = numpy.std(l_vrad)

  if len(l_vrad) > 1:
    e_mean_vrad = sig_vrad / math.sqrt(len(l_vrad)-1)
  else:
    e_mean_vrad = 0

  wt_mean_vrad = None
  wt_e_mean_vrad = None

  if qvalues is not None:
    # Mean of orders, weighted by uncertainties.
    wt = 1.0 / (l_e_vrad*l_e_vrad)
    swt = numpy.sum(wt)
    
    wt_mean_vrad = numpy.sum(l_vrad * wt) / swt
    
    # Correction for overdispersion.  We often operate with small N here
    # so this is not allowed to be less than unity, otherwise I found it
    # sometimes was, but probably just due to statistical fluke.
    chisq = numpy.sum(wt * (l_vrad - wt_mean_vrad)**2)
    errscl = math.sqrt(chisq / (nord - 1))
    if errscl < 1.0:
      errscl = 1.0

    # Resulting error in weighted mean.
    wt_e_mean_vrad = errscl / math.sqrt(swt)

  if pdf is not None:
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

  if wt_mean_vrad is not None:
    return wt_mean_vrad, wt_e_mean_vrad, l_vrad, l_corr
  else:
    return mean_vrad, e_mean_vrad, l_vrad, l_corr

ap = argparse.ArgumentParser()
ap.add_argument("template", help="template spectrum file or @list of files to be stacked")
ap.add_argument("filelist", metavar="file", nargs="+", help="spectrum file or @list of files to be stacked")
ap.add_argument("-c", metavar="ra dec pmra pmdec plx vrad epoch", type=str, help="specify catalogue information for target")
ap.add_argument("-E", action="store_true", help="don't remove emission lines from spectrum")
ap.add_argument("-o", type=int, help="override order number used for analysis")
ap.add_argument("-R", action="store_true", help="don't use cosmic rejection when stacking")
ap.add_argument("-S", action="store_true", help="don't stack epochs to make high SNR template")
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

# Orders.
multiorders = rs.multiorder
if args.o is not None:
  multiorders = [ args.o ]

qvalues = rs.qvalues

# Read spectra.
nspec = nf+1

filenamelist = [None] * nspec
speclist = [None] * nspec

filenamelist[0] = args.template
speclist[0] = tmplsp

for ifile, filename in enumerate(filelist):
  sp = rs.read_spec(filename, src_str=args.c, wantstruct=True, doreject=not args.R)

  filenamelist[ifile+1] = filename
  speclist[ifile+1] = sp

# Stacking, if requested and possible.
is_stack = False

tmplsp_flux = tmplsp.flux
tmplsp_e_flux = tmplsp.e_flux

if nspec > 2 and not args.S:
  for iiter in range(3):
    fluxlist = [None] * nspec
    e_fluxlist = [None] * nspec
    normedlist = [None] * nspec
    e_normedlist = [None] * nspec
    wtlist = numpy.empty([nspec])

    fluxlist[0] = tmplsp.flux
    e_fluxlist[0] = tmplsp.e_flux
    normedlist[0] = tmplsp.flux
    e_normedlist[0] = tmplsp.e_flux
    wtlist[0] = numpy.median(tmplsp.flux[numpy.isfinite(tmplsp.flux)])

    for ifile, filename in enumerate(filelist):
      sp = speclist[ifile+1]

      mean_vrad, e_mean_vrad, l_vrad, l_corr = do_multi_vrad(None, None, None,
                                                             tmplsp.mbjd, tmplsp.wave, tmplsp_flux, tmplsp_e_flux, tmplsp.msk,
                                                             sp.mbjd, sp.wave, sp.flux, sp.e_flux, sp.msk,
                                                             orders=multiorders, qvalues=qvalues, emchop=emchop)

      restwave = sp.wave / (1.0 + mean_vrad * 1000 / lfa.LIGHT)

      nord, npix = restwave.shape

      fluxout = numpy.empty_like(sp.flux)
      e_fluxout = numpy.empty_like(sp.flux)

      wtout = None

      normord = rs.singleorder-1

      for iord in range(nord):
        interp_flux, interp_e_flux = finterp(tmplsp.wave[iord,:], restwave[iord,:], sp.flux[iord,:], sp.e_flux[iord,:])

        fluxout[iord,:] = interp_flux
        e_fluxout[iord,:] = interp_e_flux

        if iord == normord:
          wtout = numpy.median(interp_flux[numpy.isfinite(interp_flux)])
          wtlist[ifile+1] = wtout

      fluxlist[ifile+1] = fluxout
      e_fluxlist[ifile+1] = e_fluxout
      normedlist[ifile+1] = fluxout * wtlist[0] / wtout
      e_normedlist[ifile+1] = e_fluxout * wtlist[0] / wtout

    fluxlist = numpy.array(fluxlist)
    e_fluxlist = numpy.array(e_fluxlist)
    normedlist = numpy.array(normedlist)
    e_normedlist = numpy.array(e_normedlist)

    if not args.R:
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

    # Weighted mean.  Unlike in read_spec we do not scale back to sum equiv.
    swt = numpy.sum(combmask, axis=0)

    norm = numpy.empty_like(swt, dtype=numpy.double)
    norm.fill(numpy.nan)

    norm[swt > 0] = 1.0 / swt[swt > 0]

    stacked_flux = numpy.sum(fluxlist*combmask, axis=0) * norm
    ssq = numpy.sum((e_fluxlist**2)*combmask, axis=0) * norm

    # Final uncertainty in sum = quadrature sum of uncertainties.
    stacked_e_flux = numpy.sqrt(ssq)

    # Replace template.
    tmplsp_flux = stacked_flux
    tmplsp_e_flux = stacked_e_flux

  tmplname = "Stack"
  is_stack = True

# Main correlation loop.
for ispec, sp in enumerate(speclist):
  if ispec == 0 and not is_stack:
    continue

  filename = filenamelist[ispec]

  targname = re.sub(r'_cb\.spec$', "", stripname(filename))

  basefile = stripname(filename)
  outfile = basefile + "_self.pdf"

  pdf = PdfPages(outfile)

  # Velocity.
  vrad, hbest = do_vrad(pdf, tmplname, targname,
                        tmplsp.mbjd, tmplsp.wave, tmplsp_flux, tmplsp_e_flux, tmplsp.msk,
                        sp.mbjd, sp.wave, sp.flux, sp.e_flux, sp.msk,
                        order=rs.singleorder, emchop=emchop)

  do_lsd(pdf, filename,
         tmplsp.mbjd, tmplsp.wave, tmplsp_flux, tmplsp_e_flux, tmplsp.msk,
         sp.mbjd, sp.wave, sp.flux, sp.e_flux, sp.msk,
         vrad,
         orders=multiorders, emchop=emchop)

  mean_vrad, e_mean_vrad, l_vrad, l_corr = do_multi_vrad(pdf, tmplname, targname,
                                                         tmplsp.mbjd, tmplsp.wave, tmplsp_flux, tmplsp_e_flux, tmplsp.msk,
                                                         sp.mbjd, sp.wave, sp.flux, sp.e_flux, sp.msk,
                                                         orders=multiorders, qvalues=qvalues, emchop=emchop)

  pdf.close()

  print "{0:s} {1:12.4f} {2:9.4f} {3:7.4f} {4:7.2f}".format(filename, lfa.ZMJD+sp.mbjd, mean_vrad, e_mean_vrad, sp.exptime),

  for iord in range(len(l_vrad)):
    print " {0:2d} {1:9.4f} {2:8.6f}".format(multiorders[iord], l_vrad[iord], l_corr[iord]),

  print

sys.exit(0)

