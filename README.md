tres-tools
==========

A collection of programs originally intended for analysis of
"reconnaissance spectroscopy" observations of M-dwarf planet candidate
host stars using TRES on the 1.5m Tillinghast reflector at FLWO.  It
has since grown into somewhat more than this after being used for
spectroscopic binary orbits and gained support for other instruments,
most notably CHIRON on the 1.5m telescope at CTIO.

The main program for the original application is "inone" and is likely
what most external users will be interested in.  This extracts a
single-order radial velocity and rotational broadening estimate from
the observed spectrum using an observed template spectrum, usually
of Barnard's Star, which we have found to work well for most types of
M-dwarfs.  The order chosen is found in aperture 41 of extracted TRES
spectra and spans a wavelength range of 7065-7165A dominated by
molecular features due to TiO.  A four-page PDF output file is
produced containing these results as well as a least-squares
deconvolution against the template used to assess whether the spectrum
shows evidence for multiple lines, and a page showing various atomic
features used to diagnose activity (Halpha, and Ca II H, K and
infrared triplet lines), youth (Li I) or surface gravity (Na I).

Command-line switches control behaviour but the user is unlikely to
require any of these under normal circumstances and would pass in
the template spectrum as the first argument and a list of target star
spectra for analysis.  Output PDF files are named following the input
file names and saved to the current directory.  We typically take
three subexposures on CHIRON to aid cosmic rejection and these can be
fed in to be stacked and treated as one spectrum using the IRAF-like
@list syntax where command line arguments are "epochs" and the files
listed in each list file are subexposures to be stacked.

Our template spectra for TRES and CHIRON are included in the
"templates" directory and should suffice for analysis of most target
star spectra taken with these instruments.  For user-supplied template
spectra the header keyword VELOCITY is used to specify the Barycentric
radial velocity of the template star.

In addition to the standard "inone" program two much less mature
programs for multi order radial velocity analysis (in a limited sense,
not capable of precise radial velocities on such instruments but
usually adequate for instruments like TRES or CHIRON) are included.
"multi" is intended for analysis using templates and "self" for
performing self-template analysis for relative velocities.  These
programs also attempt to estimate uncertainties although the user is
cautioned that these are known to be flawed and are frequently
underestimated.  In particular at the time of writing none of the
instruments the software ships with support for supply enough
information in the FITS files to calculate the uncertainties in the
spectrum properly so we have to make a fairly crude guess.

The suite of tools also include my own programs for extraction of
radial velocities in multiple lined systems (they are named like the
originals for clarity of method used but I have never seen those
implementations in either case so there are probably considerable
differences), and various orbit fitting programs used in our published
work.  These are currently lacking documentation.

Usage notes
===========

inone Gls699.fits spectrum1 spectrum2 ...

or

inone @BARNARDS @list1 @list2 ...

self spectrum1 spectrum2 ...

multi -r vsini Gls699.fits spectrum1 spectrum2 ...

todcor alpha vsini1 vsini2 template1 template2 spectrum1 spectrum2 ...

tricor alpha beta vsini1 vsini2 vsini3 template1 template2 template3 spectrum1 spectrum2 ...

pixscale spectrum1 spectrum2 ...

snr spectrum1 spectrum2 ...

sb1period vels pmin pmax

sb2period vels pmin pmax

sb1orbit pset vels

sb2orbit pset vels

wilson vels

Installation
============

The software was mostly for internal use so has unfortunately accrued
a few unusual dependencies.  In addition to Python 2.7 with numpy,
scipy and matplotlib the following modules are used:

fitsio version 0.9.7 (others may work but are not recommended due to
API changes), to install this use "pip install fitsio==0.9.7" or
similar.  I'm considering changing this to use astropy.io.fits
although if attempting to do so please note that it's more subtle than
it first appears, whitespace must not be stripped from header values
when reading the IRAF-style wavelength solutions in TRES files.

My "lfa" module available in "lib" on this account.  Notes found there
detail how to set up the ephemeris files for Barycentric corrections.

Support routines found in "pymisc" also on this account (these need to
be available in the Python search path, e.g. PYTHONPATH).

Orbit fitting programs also require my "eb" module, and period finding
my "sfit" module.

For our use we have usually resorted to creating a virtual environment
using "virtualenv" and installing everything inside it to avoid
disturbing the setup of the system or the user's account, especially
for Python 3 users.  I have tried to keep things compatible with
Python 3 (most particularly the C modules which should have the
necessary #ifdefs) but some of the scripts would need to be run
through 2to3 mostly to deal with the changes to "print".

Other notes
===========

Support for additional instruments can be added by altering the
"spectrograph detection" in read_spec.py and adding a new Python
library file for the instrument.  I have not attempted to implement
generic file readers because almost every instrument added so far had
a different file format and there are also usually FITS header issues
to deal with, most notably how to get accurate star coordinates for
Barycentric corrections.

Some limited functionality for override of star coordinates is
included, and a warning is emitted if the source catalogue for doing
this is not found, but it is unlikely the user will need to do this
for reconnaissance or stellar orbits, it is usually only needed for
planetary orbits, which are not really the goal of this software.

