.. phconvert documentation master file, created by
   sphinx-quickstart on Mon Oct 26 09:06:48 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to phconvert's documentation!
=====================================

:Version: |version| (`release notes <https://github.com/Photon-HDF5/phconvert/releases/>`__)

`phconvert <http://photon-hdf5.github.io/phconvert/>`__ is a python 2 & 3 library
which helps writing valid
`Photon-HDF5 <https://www.photon-hdf5.org>`_ files.
This document contains the API documentation for phconvert.

The phconvert library contains two main modules: `hdf5` and `loader`.
The former contains functions to save and validate Photon-HDF5 files.
The latter, contains functions to load other formats to be converted to
Photon-HDF5.

The phconvert repository contains a set the notebooks to convert
existing formats to Photon-HDF5 or to write Photon-HDF5 from scratch:

- `phconvert notebooks <https://github.com/Photon-HDF5/phconvert/blob/master/notebooks/>`_
  (`read online <http://nbviewer.ipython.org/github/Photon-HDF5/phconvert/blob/master/notebooks/>`__).

In particular see notebook `Writing Photon-HDF5 files <https://github.com/Photon-HDF5/phconvert/blob/master/notebooks/Writing%20Photon-HDF5%20files.ipynb>`_
(`read online <http://nbviewer.ipython.org/github/Photon-HDF5/phconvert/blob/master/notebooks/Writing%20Photon-HDF5%20files.ipynb>`__)
as an example of writing Photon-HDF5 files from scratch.

Finally, phconvert repository contains a
`JSON specification <https://github.com/Photon-HDF5/phconvert/blob/master/phconvert/specs/photon-hdf5_specs.json>`_
of the Photon-HDF5 format which lists all the valid field names and
corresponding data types and descriptions.

Contents:

.. toctree::
   :maxdepth: 1

   hdf5
   loader
   pqreader
   bhreader


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
