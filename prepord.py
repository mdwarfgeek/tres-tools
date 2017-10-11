import numpy
import sys

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
      print >>sys.stderr, "Order", order, "is entirely masked out"
      sys.exit(1)
  
    iend = n-1
    while iend >= 0 and not thismsk[iend]:
      iend -= 1

    if iend >= istart:
      if not numpy.all(thismsk[istart:iend+1]):
        print >>sys.stderr, "Mask is not contiguous in order", order
        print >>sys.stderr, istart, iend, n
        sys.exit(1)

    # Apply mask.
    thiswave = thiswave[thismsk]
    thisflux = thisflux[thismsk]
    thise_flux = thise_flux[thismsk]

  return thiswave, thisflux, thise_flux
