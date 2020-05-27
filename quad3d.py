import numpy
import warnings

def quad3d(xx, yy, zz, hh):
  x = xx.flatten()
  y = yy.flatten()
  z = zz.flatten()
  h = hh.flatten()

  # Form design matrix.
  A = numpy.empty([ len(z), 10 ])

  A[:,0] = 1.0
  A[:,1] = x
  A[:,2] = x * x
  A[:,3] = y
  A[:,4] = y * y
  A[:,5] = z
  A[:,6] = z * z
  A[:,7] = x * y
  A[:,8] = x * z
  A[:,9] = y * z

  c, chisq, rank, s = numpy.linalg.lstsq(A, h, rcond=-1)

  # Location of extremum by solving matrix equation.  This is a bit
  # lazy, probably should do the analytic solution, but I doubt it
  # would make much difference.
  M = numpy.array([[2*c[2],   c[7],   c[8]],
                   [  c[7], 2*c[4],   c[9]],
                   [  c[8],   c[9], 2*c[6]]])

  b = numpy.array([-c[1], -c[3], -c[5]])
  v = numpy.linalg.solve(M, b)

  xbest, ybest, zbest = v

  if xbest < numpy.min(x) or xbest > numpy.max(x):
    warnings.warn("x out of range during interpolation")

  if ybest < numpy.min(y) or ybest > numpy.max(y):
    warnings.warn("y out of range during interpolation")

  if zbest < numpy.min(z) or zbest > numpy.max(z):
    warnings.warn("z out of range during interpolation")

#  plt.imshow(zz,
#             interpolation='none',
#             extent=[x[0], x[-1], y[-1], y[0]],
#             cmap='Greys')
#  plt.axvline(xbest)
#  plt.axhline(ybest)
#  plt.show()
#
#  zmod = c[0] + (c[2]*xx + c[1]) * xx + (c[5]*xx + c[4]*yy + c[3]) * yy
#  plt.imshow(zmod,
#             interpolation='none',
#             extent=[x[0], x[-1], y[-1], y[0]],
#             cmap='Greys')
#  plt.axvline(xbest)
#  plt.axhline(ybest)
#  plt.show()

  hbest = c[0] + (c[2]*xbest + c[1]) * xbest + (c[7]*xbest + c[4]*ybest + c[3]) * ybest + (c[8]*xbest + c[9]*ybest + c[6]*zbest + c[5]) * zbest

  return xbest, ybest, zbest, hbest

