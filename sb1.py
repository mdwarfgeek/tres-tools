import math
import numpy
import eb

def semmaj1(esinw, ecosw, period, k):
  # Precompute some quantities.
  omega = eb.TWOPI / (period * eb.DAY)

  omesq = 1.0 - (esinw**2 + ecosw**2)
  roe = numpy.sqrt(omesq)

  return((k*1000) * roe / (omega * eb.AU))

# f1(M) = (M2 sin i)^3 / (M1 + M2)^2
#       = K^3 (1 - e^2)^(3/2) / (G omega)

def massfunc(esinw, ecosw, period, k):
  # Precompute some quantities.
  omega = eb.TWOPI / (period * eb.DAY)

  omesq = 1.0 - (esinw**2 + ecosw**2)
  roe = numpy.sqrt(omesq)

  f1 = ((k*1000)**3 * omesq * roe) / (eb.GMSUN * omega)

  return f1

def massfunce(e, period, k):
  # Precompute some quantities.
  omega = eb.TWOPI / (period * eb.DAY)

  omesq = 1.0 - e*e
  roe = numpy.sqrt(omesq)

  f1 = ((k*1000)**3 * omesq * roe) / (eb.GMSUN * omega)

  return f1

# Solve for mass ratio from SB1 orbital quantities.
# period in days, k in km/s, m1 in Msol.

def qfromsb1(esinw, ecosw, period, k, m1, sini):
  # Precompute some quantities.
  omega = eb.TWOPI / (period * eb.DAY)

  omesq = 1.0 - (esinw**2 + ecosw**2)
  roe = numpy.sqrt(omesq)

  # Kepler 3: a^3 = G M1 (1+q) / omega^2
  # Rad. Vel: K (1+q)/q = a omega sin i / sqrt(1.0 - e^2)
  #
  # So (1+q)^2 / q^3 = (G M1 omega sin^3 i) / (K^3 (1.0 - e^2)^(3/2))
  # Rewrite as t = q^3 / (1+q)^2
  gmw = eb.GMSUN * m1 * omega
  kfac = k*1000 * roe / sini

  t = kfac**3 / gmw

  # Solve cubic equation q^3 = t (1+q)^2 using cubic formula.
  # Rearrange to -q^3 + t q^2 + 2t q + t = 0
  md = 4.0*t + 27.0  # - discriminant / t^2

  delta0 = t + 6.0  # delta0 / t
  delta1 = (2.0*t + 18.0) * t + 27.0  # delta1 / t

  # Cube root argument / kfac^3.  CHECKME: any funny business with signs?
  arg = 0.5 * (delta1 + numpy.sqrt(27.0*md)) / gmw
  C = numpy.cbrt(arg)  # C / kfac

  # One real root.
  q = (t + kfac*(C + kfac*delta0 / (C*gmw))) / 3.0

  return q

# Solve for minimum inclination for eclipse at conjunction.  Uses
# mass-radius relation passed in to get the sum of the radii for
# each value of q.

def eclprob(esinw, ecosw, period, k, m1, mrrel1, mrrel2, wantsec=False): 
  omega = eb.TWOPI / (period * eb.DAY)
  omesq = 1.0 - (esinw**2 + ecosw**2)
 
  # Enhancement factor from eccentricity.
  if wantsec:
    eccfac = (1.0 - esinw) / omesq
  else:
    eccfac = (1.0 + esinw) / omesq

  # Start from 90 degrees.
  cosi = 0.0
  sini = 1.0
  rasum = None
  rr = None

  for i in range(10):
    # Compute mass ratio.
    q = qfromsb1(esinw, ecosw, period, k, m1, sini)
    
    semmaj = (eb.GMSUN * m1 * (1.0 + q) / (omega*omega))**(1.0/3.0)
    
    m2 = m1 * q
    
    # Get radii.
    r1 = mrrel1(m1)
    r2 = mrrel2(m2)

    # Sum of radii / a.
    newrasum = (r1 + r2) * eb.RSUN / semmaj
    newrr = r2 / r1

    # Minimum inclination.
    newcosi = newrasum * eccfac
    newsini = math.sqrt(1.0 - cosi*cosi)

    # Check for convergence.
    delt = newcosi - cosi

    cosi = newcosi
    sini = newsini
    rasum = newrasum
    rr = newrr

    if abs(delt) < 1.0e-15:
      return cosi, rasum, rr

  return None

