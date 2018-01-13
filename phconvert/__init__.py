#
# phconvert - Reference library to read and save Photon-HDF5 files
#
# Copyright (C) 2015 Antonino Ingargiola <tritemio@gmail.com>
#

import phconvert.loader
import phconvert.hdf5
import phconvert.v04

has_matplotlib = True
try:
    import matplotlib
except ImportError:
    has_matplotlib = False
if has_matplotlib:
    import phconvert.plotter
    del matplotlib

from phconvert._version import get_versions
__version__ = get_versions()['version']
del get_versions
