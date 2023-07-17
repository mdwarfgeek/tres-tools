import math
import numpy

import matplotlib.pyplot as plt

import lfa

from quad3d import *

class tricor:
  def __init__(self, frv,
               tmpla_wave, tmpla_flux,
               tmplb_wave, tmplb_flux,
               tmplc_wave, tmplc_flux,
               targ_wave, targ_flux,
               vsinia, vsinib, vsinic,
               alpha, beta,
               u1, u2):

    # Decide wavelength sampling.
    wavemin = max(tmpla_wave[0], tmplb_wave[0], tmplc_wave[0], targ_wave[0])
    wavemax = min(tmpla_wave[-1], tmplb_wave[-1], tmplc_wave[-1], targ_wave[-1])

    lwmin = math.log(wavemin)
    lwmax = math.log(wavemax)

    self.lwsamp = (lwmax-lwmin) / (frv.nbin-1)

    # Template x template.
    ll = frv.correlate(tmplb_wave, tmplb_flux,
                       tmpla_wave, tmpla_flux,
                       vsinib*1000/lfa.LIGHT,
                       u1, u2,
                       zbroadt=vsinia*1000/lfa.LIGHT,
                       lwmin=lwmin, lwmax=lwmax)

    ztaxb, corrtaxb, zbesttaxb, hbesttaxb, sigtb = ll

    ll = frv.correlate(tmplc_wave, tmplc_flux,
                       tmpla_wave, tmpla_flux,
                       vsinic*1000/lfa.LIGHT,
                       u1, u2,
                       zbroadt=vsinia*1000/lfa.LIGHT,
                       lwmin=lwmin, lwmax=lwmax)

    ztaxc, corrtaxc, zbesttaxc, hbesttaxc, sigtc = ll

    ll = frv.correlate(tmplc_wave, tmplc_flux,
                       tmplb_wave, tmplb_flux,
                       vsinic*1000/lfa.LIGHT,
                       u1, u2,
                       zbroadt=vsinib*1000/lfa.LIGHT,
                       lwmin=lwmin, lwmax=lwmax)

    ztbxc, corrtbxc, zbesttbxc, hbesttbxc, sigtc = ll

    # Template x target.
    ll = frv.correlate(tmpla_wave, tmpla_flux,
                       targ_wave, targ_flux,
                       vsinia*1000/lfa.LIGHT,
                       u1, u2,
                       lwmin=lwmin, lwmax=lwmax)

    za, corra, zbesta, hbesta, sigta = ll
  
    ll = frv.correlate(tmplb_wave, tmplb_flux,
                       targ_wave, targ_flux,
                       vsinib*1000/lfa.LIGHT,
                       u1, u2,
                       lwmin=lwmin, lwmax=lwmax)

    zb, corrb, zbestb, hbestb, sigtb = ll

    ll = frv.correlate(tmplc_wave, tmplc_flux,
                       targ_wave, targ_flux,
                       vsinic*1000/lfa.LIGHT,
                       u1, u2,
                       lwmin=lwmin, lwmax=lwmax)

    zc, corrc, zbestc, hbestc, sigtc = ll

    self.alp = alpha * sigtb/sigta
    self.btp = beta * sigtc/sigta

    self.corrtaxb = corrtaxb
    self.corrtaxc = corrtaxc
    self.corrtbxc = corrtbxc

    self.corra = corra
    self.corrb = corrb
    self.corrc = corrc

    self.hbin = frv.hbin

  def calc(self,
           iamin, iamax, iastep,
           ibmin, ibmax, ibstep,
           icmin, icmax, icstep):

    inda = numpy.arange(iamin, iamax+1, iastep)
    
    hh = numpy.empty([(iamax-iamin)//iastep+1,
                      (ibmax-ibmin)//ibstep+1,
                      (icmax-icmin)//icstep+1],
                     dtype=numpy.double)

    for ic in range(icmin, icmax+1, icstep):
      for ib in range(ibmin, ibmax+1, ibstep):
        dab = ib - inda
        c_tmplb_tmpla = self.corrtaxb[self.hbin+dab]
      
        dac = ic - inda
        c_tmplc_tmpla = self.corrtaxc[self.hbin+dac]
        
        dbc = ic - ib
        c_tmplc_tmplb = self.corrtbxc[self.hbin+dbc]

        denom = 1.0 + 2*self.alp*c_tmplb_tmpla + 2*self.btp*c_tmplc_tmpla + 2*self.alp*self.btp*c_tmplc_tmplb + self.alp*self.alp + self.btp*self.btp
        rtnum = self.corra[inda] + self.alp * self.corrb[ib] + self.btp * self.corrc[ic]

        hsq = rtnum*rtnum / denom

        hh[:,(ib-ibmin)//ibstep,(ic-icmin)//icstep] = numpy.sqrt(hsq)

    return hh

  def find(self,
           iamin, iamax, iastep,
           ibmin, ibmax, ibstep,
           icmin, icmax, icstep):

    inda = numpy.arange(iamin, iamax+1, iastep)
    
    iabest = None
    ibbest = None
    icbest = None
    hsqbest = None

    for ic in range(icmin, icmax+1, icstep):
      for ib in range(ibmin, ibmax+1, ibstep):
        dab = ib - inda
        c_tmplb_tmpla = self.corrtaxb[self.hbin+dab]
      
        dac = ic - inda
        c_tmplc_tmpla = self.corrtaxc[self.hbin+dac]
        
        dbc = ic - ib
        c_tmplc_tmplb = self.corrtbxc[self.hbin+dbc]

        denom = 1.0 + 2*self.alp*c_tmplb_tmpla + 2*self.btp*c_tmplc_tmpla + 2*self.alp*self.btp*c_tmplc_tmplb + self.alp*self.alp + self.btp*self.btp
        rtnum = self.corra[inda] + self.alp * self.corrb[ib] + self.btp * self.corrc[ic]

        hsq = rtnum*rtnum / denom

        smax = hsq.argmax()
        hsqmax = hsq[smax]

        if hsqbest is None or hsqmax > hsqbest:
          iabest = iamin + smax * iastep
          ibbest = ib
          icbest = ic
          hsqbest = hsqmax

    return iabest, ibbest, icbest, hsqbest

  def vel2ind(self, vel):
    dll = numpy.log1p(vel * 1000 / lfa.LIGHT)
    ill = numpy.rint(dll / self.lwsamp).astype(int)
    ii = self.hbin + ill

    return ii

  def ind2vel(self, ind):
    vel = numpy.expm1((ind-self.hbin)*self.lwsamp) * lfa.LIGHT / 1000

    return vel

  def run(self,
          minvela=-250, maxvela=250,
          minvelb=-250, maxvelb=250,
          minvelc=-250, maxvelc=250,
          fpsamp=16, fpexpand=4, pkfit=1, rpk=2):

    # Perform initial search using coarse grid.
    iamin = self.vel2ind(minvela)
    iamax = self.vel2ind(maxvela)
    ibmin = self.vel2ind(minvelb)
    ibmax = self.vel2ind(maxvelb)
    icmin = self.vel2ind(minvelc)
    icmax = self.vel2ind(maxvelc)

    iabest, ibbest, icbest, hsqbest = self.find(iamin, iamax, fpsamp,
                                                ibmin, ibmax, fpsamp,
                                                icmin, icmax, fpsamp)

    if fpsamp > 1:
      # Refine at full resolution.
      iamin = max(iamin, iabest - fpexpand*fpsamp)
      iamax = min(iamax, iabest + fpexpand*fpsamp)
      
      ibmin = max(ibmin, ibbest - fpexpand*fpsamp)
      ibmax = min(ibmax, ibbest + fpexpand*fpsamp)
      
      icmin = max(icmin, icbest - fpexpand*fpsamp)
      icmax = min(icmax, icbest + fpexpand*fpsamp)
      
      iabest, ibbest, icbest, hsqbest = self.find(iamin, iamax, 1,
                                                  ibmin, ibmax, 1,
                                                  icmin, icmax, 1)

    if pkfit:
      # Refine peak using 3-D quadratic fit.
      iamin = iabest - rpk
      iamax = iabest + rpk
      
      ibmin = ibbest - rpk
      ibmax = ibbest + rpk
      
      icmin = icbest - rpk
      icmax = icbest + rpk
      
      npkfit = 2*rpk+1

      hh = self.calc(iamin, iamax, 1,
                     ibmin, ibmax, 1,
                     icmin, icmax, 1)
    
      idd = numpy.arange(npkfit) - rpk
      dd = idd.astype(numpy.double)

      dcc = numpy.tile(dd, (npkfit, npkfit, 1))
      dbb = numpy.transpose(dcc, (1, 2, 0))
      daa = numpy.transpose(dcc, (2, 0, 1))

      da, db, dc, hbest = quad3d(daa, dbb, dcc, hh)

    else:
      da = 0.0
      db = 0.0
      dc = 0.0
      hbest = math.sqrt(hsqbest)

    # Return result as velocity.
    va = self.ind2vel(iabest + da)
    vb = self.ind2vel(ibbest + db)
    vc = self.ind2vel(icbest + dc)

    return va, vb, vc, hbest
