import math
import numpy

import lfa

from quad2d import *

class todcor:
  def __init__(self, frv,
               tmpla_wave, tmpla_flux,
               tmplb_wave, tmplb_flux,
               targ_wave, targ_flux,
               vsinia, vsinib,
               alpha,
               u1, u2):

    # Decide wavelength sampling.
    wavemin = max(tmpla_wave[0], tmplb_wave[0], targ_wave[0])
    wavemax = min(tmpla_wave[-1], tmplb_wave[-1], targ_wave[-1])

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

    self.alp = alpha * sigtb/sigta

    self.corrtaxb = corrtaxb

    self.corra = corra
    self.corrb = corrb

    self.hbin = frv.hbin

  def calc(self,
           iamin, iamax, iastep,
           ibmin, ibmax, ibstep):

    inda = numpy.arange(iamin, iamax+1, iastep)
    
    hh = numpy.empty([(iamax-iamin)//iastep+1,
                      (ibmax-ibmin)//ibstep+1],
                     dtype=numpy.double)

    for ib in range(ibmin, ibmax+1, ibstep):
      dab = ib - inda
      c_tmplb_tmpla = self.corrtaxb[self.hbin+dab]

      denom = 1.0 + 2*self.alp*c_tmplb_tmpla + self.alp*self.alp
      rtnum = self.corra[inda] + self.alp * self.corrb[ib]

      hsq = rtnum*rtnum / denom

      hh[:,(ib-ibmin)//ibstep] = numpy.sqrt(hsq)

    return hh

  def find(self,
           iamin, iamax, iastep,
           ibmin, ibmax, ibstep,
           restrict=None):

    all_inda = numpy.arange(iamin, iamax+1, iastep)
    
    iabest = None
    ibbest = None
    hsqbest = None

    for ib in range(ibmin, ibmax+1, ibstep):
      if restrict is None:
        inda = all_inda
      else:
        if restrict == "n" or restrict == "s":
          inda = all_inda[all_inda <= ib]
        elif restrict == "p":
          inda = all_inda[all_inda >= ib]
        else:
          inda = all_inda

      dab = ib - inda
      c_tmplb_tmpla = self.corrtaxb[self.hbin+dab]

      denom = 1.0 + 2*self.alp*c_tmplb_tmpla + self.alp*self.alp
      rtnum = self.corra[inda] + self.alp * self.corrb[ib]

      hsq = rtnum*rtnum / denom

      smax = hsq.argmax()
      hsqmax = hsq[smax]

      if hsqbest is None or hsqmax > hsqbest:
        iabest = inda[0] + smax * iastep
        ibbest = ib
        hsqbest = hsqmax

    return iabest, ibbest, hsqbest

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
          fpsamp=1, fpexpand=16, pkfit=1, rpk=2,
          restrict=None):

    # Perform initial search using coarse grid.
    iamin = self.vel2ind(minvela)
    iamax = self.vel2ind(maxvela)
    ibmin = self.vel2ind(minvelb)
    ibmax = self.vel2ind(maxvelb)

    iabest, ibbest, hsqbest = self.find(iamin, iamax, fpsamp,
                                        ibmin, ibmax, fpsamp,
                                        restrict=restrict)

    if fpsamp > 1:
      # Refine at full resolution.
      iamin = max(iamin, iabest - fpexpand*fpsamp)
      iamax = min(iamax, iabest + fpexpand*fpsamp)
      
      ibmin = max(ibmin, ibbest - fpexpand*fpsamp)
      ibmax = min(ibmax, ibbest + fpexpand*fpsamp)
      
      iabest, ibbest, hsqbest = self.find(iamin, iamax, 1,
                                          ibmin, ibmax, 1,
                                          restrict=restrict)

    if pkfit:
      # Refine peak using 2-D quadratic fit.
      iamin = iabest - rpk
      iamax = iabest + rpk
      
      ibmin = ibbest - rpk
      ibmax = ibbest + rpk
      
      npkfit = 2*rpk+1

      hh = self.calc(iamin, iamax, 1,
                     ibmin, ibmax, 1)
    
      idd = numpy.arange(npkfit) - rpk
      dd = idd.astype(numpy.double)

      dbb = numpy.tile(dd, (npkfit, 1))
      daa = numpy.transpose(dbb)

      da, db, hbest = quad2d(daa, dbb, hh)

    else:
      da = 0.0
      db = 0.0
      hbest = math.sqrt(hsqbest)

    # Return result as velocity.
    va = self.ind2vel(iabest + da)
    vb = self.ind2vel(ibbest + db)

    return va, vb, hbest
