import numpy

# Flux conserving linear interpolation.

def finterp(xout, xin, yin, e_yin=None, left=None, right=None):
  nin = len(xin)
  nout = len(xout)

  # Wavelength interval of each pixel.
  # delta_x = x_i+1 - x_i-1
  # do by shifting input array.
  # Extend to ends (where we have no information).
  dxin = numpy.empty_like(xin)

  dxin[1:nin-1] = 0.5*(xin[2:nin] - xin[0:nin-2])

  dxin[0] = dxin[1]
  dxin[nin-1] = dxin[nin-2]

  dxout = numpy.empty_like(xout)

  dxout[1:nout-1] = 0.5*(xout[2:nout] - xout[0:nout-2])

  dxout[0] = dxout[1]
  dxout[nout-1] = dxout[nout-2]

  # Interpolate in flux density, and then scale back to flux.
  yout = dxout * numpy.interp(xout, xin, yin/dxin, left, right)

  if e_yin is not None:
    tmp = numpy.interp(xout, xin, (e_yin/dxin)**2, left, right)
    tmp[tmp < 0] = 0
    e_yout = dxout * numpy.sqrt(tmp)
    return yout, e_yout
  else:
    return yout
