#
# phconvert - Reference library to read and save Photon-HDF5 files
#
# Copyright (C) 2015 Antonino Ingargiola <tritemio@gmail.com>
#

from . import loader
from . import hdf5

has_matplotlib = True
try:
    import matplotlib
except ImportError:
    has_matplotlib = False
if has_matplotlib:
    from . import plotter

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
