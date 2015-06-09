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

import pkg_resources
from collections import OrderedDict
import json

LATEST_FORMAT_VERSION = '0.4dev'

_specs_file_fields = 'specs/photon-hdf5_fields.json'


def _get_fields_descr():
    s = pkg_resources.resource_string('phconvert',
                                      _specs_file_fields).decode('utf8')
    return json.loads(s)

# Metadata for the HDF5 root node
root_attributes = OrderedDict([
    ('format_name', 'Photon-HDF5'),
    ('format_version', LATEST_FORMAT_VERSION),
    ('format_url', 'http://photon-hdf5.readthedocs.org/'),
])


official_fields_descr = _get_fields_descr()
