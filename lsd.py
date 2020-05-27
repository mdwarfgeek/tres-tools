import math
import numpy

import lfa

from scipy.interpolate import InterpolatedUnivariateSpline

from makesky import *
from prepord import *

def lsd_multiorder(tmpl_wave, tmpl_flux, tmpl_e_flux, tmpl_msk,
                   wave, flux, e_flux, msk,
                   orders,
                   vl, vh, nv,
                   kreg, emchop=True):
  zl = vl * 1000.0 / lfa.LIGHT
  zh = vh * 1000.0 / lfa.LIGHT

  zvals = numpy.linspace(zl, zh, nv)
  vels = zvals * lfa.LIGHT / 1000.0

  AA = numpy.zeros([ nv, nv ])
  bb = numpy.zeros([ nv ])

  for order in orders:
    # Extract order and clean.
    thistmpl_wave, thistmpl_flux, thistmpl_e_flux = prepord(order, tmpl_wave, tmpl_flux, tmpl_e_flux, tmpl_msk)
    thiswave, thisflux, thise_flux = prepord(order, wave, flux, e_flux, msk)

    # Take off sky.
    ss = makesky(thistmpl_wave, thistmpl_flux, 4)

    thistmpl_flux /= ss
    thistmpl_e_flux /= ss

    ss = makesky(thiswave, thisflux, 4)

    thisflux /= ss
    thise_flux /= ss

    tmpl_ww = numpy.isfinite(thistmpl_flux)
    ww = numpy.isfinite(thisflux)

    if emchop:
      # Clip emission lines.
      medflux, sigflux = medsig(thistmpl_flux[tmpl_ww])
      tmpl_ww = numpy.logical_and(tmpl_ww,
                                  thistmpl_flux < medflux + 5*sigflux)

      medflux, sigflux = medsig(thisflux[ww])
      ww = numpy.logical_and(ww,
                             thisflux < medflux + 5*sigflux)

    thistmpl_wave = thistmpl_wave[tmpl_ww]
    thistmpl_flux = thistmpl_flux[tmpl_ww]
    thistmpl_e_flux = thistmpl_e_flux[tmpl_ww]
      
    thiswave = thiswave[ww]
    thisflux = thisflux[ww]
    thise_flux = thise_flux[ww]

    # Figure out which pixels in are always in range.
    wanttmpl = thiswave - zl*thiswave

    inrangel = numpy.logical_and(wanttmpl >= thistmpl_wave[0],
                                 wanttmpl <= thistmpl_wave[-1])

    wanttmpl = thiswave - zh*thiswave

    inrangeh = numpy.logical_and(wanttmpl >= thistmpl_wave[0],
                                 wanttmpl <= thistmpl_wave[-1])

    inrange = numpy.logical_and(inrangel, inrangeh)

    # Restrict to that...
    thiswave = thiswave[inrange]
    thisflux = thisflux[inrange]
    thise_flux = thise_flux[inrange]

#    plt.plot(thistmpl_wave, thistmpl_flux)
#    plt.plot(thiswave, thisflux)
#    plt.show()

    nwave = len(thiswave)

    # Form design matrix.
    A = numpy.empty([ nwave, nv ])

    # Interpolating spline.
    spl = InterpolatedUnivariateSpline(thistmpl_wave, thistmpl_flux, k=3)

    for iz, z in enumerate(zvals):
      wanttmpl = thiswave - z*thiswave

      interp_flux = spl(wanttmpl)

      A[:,iz] = interp_flux

    # Accumulate.
    AA += numpy.dot(A.transpose(), A)
    bb += numpy.dot(A.transpose(), thisflux)

  # Regularization.
  AA += kreg * numpy.identity(nv)  # need to calculate this constant properly

  prof, chisq, rank, s = numpy.linalg.lstsq(AA, bb, rcond=-1)

  return vels, prof
