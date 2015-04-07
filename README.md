# phconvert

*phconvert* is a python library to read from several file formats and to save in [Photon-HDF5 format](http://photon-hdf5.readthedocs.org/).

*phconvert* contains the reference implementation for creating Photon-HDF5 files.

Current formats that can be read are: PicoQuant .ht3, Becker & Hickl .spc/.set and usALEX .SM files.
Example notebooks show how to convert these formats.

## Installation

You can install *phconvert* using conda:

    conda install -c tritemio phconvert
    
or PIP:

    pip install phconvert

or downloading the souces and doing the usual:

    python setup.py build
    python setup.py install
    
## Dependencies

- python 2.7, 3.3 or greater
- future
- numpy
- pytables
- numba (optional)
    
## License

*phconvert* is released under the license GNU GPL Version 2.

