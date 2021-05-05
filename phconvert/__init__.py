#
# phconvert - Reference library to read and save Photon-HDF5 files
#
# Copyright (C) 2015 Antonino Ingargiola <tritemio@gmail.com>
#
from phconvert._version import get_versions
__version__ = get_versions()['version']
del get_versions

import sys

if sys.version_info < (3,):
    raise ImportError(
    """You are running phconvert %s on Python 2

phconvert 0.9 and above are no longer compatible with Python 2, and you still
ended up with this version installed. That's unfortunate; sorry about that.
It should not have happened. Make sure you have pip >= 9.0 to avoid this kind
of issue, as well as setuptools >= 24.2:

 $ pip install pip setuptools --upgrade

Your choices:

- Upgrade to Python 3.

- Install an older version of phconvert:

 $ pip install 'phconvert<0.9'

It would be great if you can figure out how this version ended up being
installed, and try to check how to prevent that for future users.

Feel free to report the issue to:

https://github.com/Photon-HDF5/phconvert/issues

""" % __version__)

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
