import numpy
import warnings

def quad2d(xx, yy, zz):
  x = xx.flatten()
  y = yy.flatten()
  z = zz.flatten()

  # Form design matrix.
  A = numpy.empty([ len(z), 6 ])

  A[:,0] = 1.0
  A[:,1] = x
  A[:,2] = x * x
  A[:,3] = y
  A[:,4] = y * y
  A[:,5] = x * y

  c, chisq, rank, s = numpy.linalg.lstsq(A, z, rcond=-1)

  # Location of extremum.
  xbest = (2*c[1]*c[4] - c[3]*c[5]) / (c[5]*c[5] - 4*c[2]*c[4])
  ybest = (2*c[2]*c[3] - c[1]*c[5]) / (c[5]*c[5] - 4*c[2]*c[4])

  if xbest < numpy.min(x) or xbest > numpy.max(x):
    warnings.warn("x out of range during interpolation")

  if ybest < numpy.min(y) or ybest > numpy.max(y):
    warnings.warn("y out of range during interpolation")

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

  zbest = c[0] + (c[2]*xbest + c[1]) * xbest + (c[5]*xbest + c[4]*ybest + c[3]) * ybest

  return xbest, ybest, zbest

