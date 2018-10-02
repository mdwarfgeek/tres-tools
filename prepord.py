import numpy
import warnings

def prepord(order, wave, flux, e_flux, msk=None, trim=1):
  iord = order-1

  # Extract this order, take off a few junk pixels.
  if trim:
    thiswave = wave[iord,5:-5]
    thisflux = flux[iord,5:-5]
    thise_flux = e_flux[iord,5:-5]
    thismsk = msk[iord,5:-5]
  else:
    thiswave = wave[iord,:]
    thisflux = flux[iord,:]
    thise_flux = e_flux[iord,:]
    thismsk = msk[iord,:]

  if thismsk is not None:
    # Check mask is contiguous - required for use in correlations.
    n = len(thismsk)

    istart = 0
    while istart < n and not thismsk[istart]:
      istart += 1
    
    if istart >= n:
      raise RuntimeError("order {0:d} is entirely masked out".format(order))
  
    iend = n-1
    while iend >= 0 and not thismsk[iend]:
      iend -= 1

    if iend >= istart:
      if not numpy.all(thismsk[istart:iend+1]):
#        raise RuntimeError("mask is not contiguous in order {0:d}: {1:d} {2:d} {3:d}".format(order, istart, iend, n))

        warnings.warn("mask is not contiguous in order {0:d}: {1:d} {2:d} {3:d}".format(order, istart, iend, n))

        iend = istart+1
        while iend < n and thismsk[iend]:
          iend += 1

        thismsk.fill(0)
        thismsk[istart:iend] = 1

    # Apply mask.
    thiswave = thiswave[thismsk]
    thisflux = thisflux[thismsk]
    thise_flux = thise_flux[thismsk]

  return thiswave, thisflux, thise_flux
