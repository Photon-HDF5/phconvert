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


LATEST_FORMAT_VERSION = b'0.5rc1'

_specs_file = 'specs/photon-hdf5_specs.json'


def _get_fields_descr():
    """
    Build two related dictionaries, one with keys as simple descriptions
    Second builds re.Match objects with keys.
    """
    s = files('phconvert').joinpath(_specs_file).read_text(encoding='utf8')
    specs_dict = json.loads(s)
    return specs_dict

# Metadata for the HDF5 root node
root_attributes = OrderedDict([
    ('format_name', b'Photon-HDF5'),
    ('format_version', LATEST_FORMAT_VERSION),
    ('format_url', b'http://photon-hdf5.org/'),
])

        
        


official_fields_specs = _get_fields_descr()

