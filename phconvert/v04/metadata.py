# -*- coding: utf-8 -*-
#
# phconvert - Convert files to Photon-HDF5 format
#
# Copyright (C) 2014-2015 Antonino Ingargiola <tritemio@gmail.com>
#
"""
This module defines the string descriptions for all the fields in the
**Photon-HDF5** format.
"""
from sys import version_info as python_version
if python_version.major > 3 or python_version.minor >= 10:
    from importlib.resources import files
else:
    from importlib_resources import files
del python_version

# import pkg_resources
from collections import OrderedDict
import json

FORMAT_VERSION = b'0.4'

_specs_file_fields = 'v04/specs/photon-hdf5_specs.json'


def _get_fields_descr():
    s = files('phconvert').joinpath(_specs_file_fields).read_text(encoding='utf8')
    # s = pkg_resources.resource_string('phconvert',
    #                                   _specs_file_fields).decode('utf8')
    descr = json.loads(s)
    return descr

# Metadata for the HDF5 root node
root_attributes = OrderedDict([
    ('format_name', b'Photon-HDF5'),
    ('format_version', FORMAT_VERSION),
    ('format_url', b'http://photon-hdf5.org/'),
])

official_fields_specs = _get_fields_descr()
