import math
import numpy

import lfa

def do_vsini_grid(frv,
                  thistmpl_wave, thistmpl_flux,
                  thiswave, thisflux,
                  ald, bld,
                  vsinia, vsinib, nvsini):

  samp = (vsinib - vsinia) / (nvsini-1)
  
  l_vsini = numpy.empty([nvsini])
  l_p = numpy.empty([nvsini])
  
  for ivsini in range(nvsini):
    vsini = vsinia + samp * float(ivsini)

    z, corr, zbest, hbest, sigt = frv.correlate(thistmpl_wave, thistmpl_flux,
                                                thiswave, thisflux,
                                                vsini*1000 / lfa.LIGHT,
                                                ald, bld)
    
    # Normalised (-chisq) like quantity.
    p = hbest*hbest
    
    l_vsini[ivsini] = vsini
    l_p[ivsini] = p
      
  ibest = numpy.argmax(l_p)
  pbest = l_p[ibest]
      
  if ibest > 0 and ibest < len(l_p)-1:
    aa = l_p[ibest]
    bb = 0.5*(l_p[ibest+1] - l_p[ibest-1])
    cc = 0.5*(l_p[ibest+1] + l_p[ibest-1] - 2.0*aa)
    offset = -0.5*bb/cc
  else:
    offset = 0
    
  vsini = vsinia + samp * (ibest+offset)
  if vsini < 0:
    vsini = 0
  
  return vsini, l_vsini, l_p

