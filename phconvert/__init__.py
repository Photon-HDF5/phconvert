#
# Copyright (C) 2015 Antonino Ingargiola <tritemio@gmail.com>
#

from . import loader
from . import plotter
from . import hdf5

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
