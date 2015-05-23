# phconvert

*phconvert* is python 2/3 library for writing and reading 
[Photon-HDF5 files](http://photon-hdf5.readthedocs.org/),
a file format for timestamp-based single-molecule spectroscopy.
*phconvert* is especially useful when saving to Photon-HDF5, because 
it tests the compliance to the Photon-HDF5 specification
and it automatically adds description attributes for each field.

In case you just want to read a Photon-HDF5 file, using phconvert is not 
necessary (although it provides some helper functions). 
Examples on reading Photon-HDF5 directly (without phconvert) 
can be found in [this repository](https://github.com/Photon-Data/photon_hdf5_reading_examples).

*phconvert* contains also functions to load a few common binary formats 
used in in single-molecule spectroscopy such as PicoQuant .ht3, 
Becker & Hickl .spc/.set and usALEX .SM files.

The included notebooks show how to convert these formats to Photon-HDF5.

## Installation

You can install *phconvert* using conda:

    conda install -c tritemio phconvert
    
or PIP:

    pip install phconvert

or by downloading the sources and doing the usual:

    python setup.py build
    python setup.py install
    
## Dependencies

- python 2.7, 3.3 or greater
- future
- numpy >=1.9
- pytables >=3.1
- numba (optional) *to enable a fast HT3 file reader*
    
## License

*phconvert* is released under the license GNU GPL Version 2.

