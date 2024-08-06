#
# phconvert - Reference library to read and save Photon-HDF5 files
#
# Copyright (C) 2015 Antonino Ingargiola <tritemio@gmail.com>
#
from . import loader
from . import hdf5
from . import v04
from . import helperfuncs

has_matplotlib = True
try:
    import matplotlib
except ImportError:
    has_matplotlib = False
if has_matplotlib:
    from . import plotter
    del matplotlib


from phconvert._version import version as __version__
