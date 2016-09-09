Module ``bhreader``
===================

.. automodule:: phconvert.bhreader

List of functions
-----------------

High-level functions to load and decode Becker & Hickl SPC/SET pair of files:

.. autofunction:: phconvert.bhreader.load_spc

.. autofunction:: phconvert.bhreader.load_set


Low-level functions
'''''''''''''''''''

These functions are the building blocks for decoding Becker & Hickl files:

.. autofunction:: phconvert.bhreader.bh_set_identification

.. autofunction:: phconvert.bhreader.bh_set_sys_params

.. autofunction:: phconvert.bhreader.bh_decode

.. autofunction:: phconvert.bhreader.bh_print_sys_params
