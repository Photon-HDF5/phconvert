# phconvert

*phconvert* is python 2/3 library for writing and reading
<a href="http://photon-hdf5.org/" target="_blank">Photon-HDF5</a>
files, a file format for timestamp-based single-molecule spectroscopy.
Additionally, *phconvert* can convert a few common binary formats
used in in single-molecule spectroscopy (PicoQuant .HT3,
Becker & Hickl .SPC/.SET) to Photon-HDF5.

## Quick-start: Converting files to Photon-HDF5

Converting one of the supported files formats to Photon-HDF5 does not require 
being able to program in python. All you need is running the appropriate notebook
for your input format and follow the instructions therein.

To run a notebook on your machine you need to install the *Jupyter Notebook App*. 
A quick-start guide on installing and running the *Jupyter Notebook App* is available here:

- <a href="http://jupyter-notebook-beginner-guide.readthedocs.org/" target="_blank">Jupyter/IPython Notebook Quick Start Guide</a>

Then you need to install the *phconvert* library with the command (type it in the terminal, cmd on windows):

    conda install -c tritemio phconvert
    
Finally you can download one of the provided notebooks and run it on your machine.
To download all the notebooks in one step you can download the 
[phconvert zip](https://github.com/Photon-HDF5/phconvert/archive/master.zip), 
which contains all the notebooks in the `notebooks` subfolder.

## Project details

*phconvert* repository contains a python packages (library) and a set of
[notebooks](https://github.com/Photon-HDF5/phconvert/tree/master/notebooks) 
([online viewer](http://nbviewer.ipython.org/github/Photon-HDF5/phconvert/tree/master/notebooks/)) 
that show how to convert other formats to Photon-HDF5.

*phconvert* tests the compliance to the Photon-HDF5 specifications
before saving a new files and it automatically adds description 
attributes for each field.

## Read Photon-HDF5 files

In case you just want to read Photon-HDF5 files, phconvert is not
necessary (although it provides some helper functions).
Photon-HDF5 files can be opened with the standard HDF5 viewer
[HDFView](https://www.hdfgroup.org/products/java/hdfview/).
Moreover, we provide code examples on reading Photon-HDF5 files
in multiple languages in 
[this repository](https://github.com/Photon-HDF5/photon_hdf5_reading_examples).

## Installation

The recommended way to install *phconvert* is by using conda (requires installing [Continuum Anaconda](https://store.continuum.io/cshop/anaconda/) first):

    conda install -c tritemio phconvert

You can also install through PIP (requires installing python, numpy and pytables first):

    pip install phconvert

or by downloading the sources and doing the usual (all dependencies need to be installed first):

    python setup.py build
    python setup.py install

## Dependencies

- python 2.7, 3.3 or greater
- future
- numpy >=1.9
- pytables >=3.1
- numba (optional) *to enable a fast HT3 file reader*

> **Note**
> when installing via `conda` all the dependencies are automatically installed.


## The phconvert library (for developers)

The *phconvert* library contains two main modules `hdf5` and `loader`. The former contains 
the function `save_photon_hdf5()` that is used to create Photon-HDF5 files.

The function `save_photon_hdf5()` requires as an argument the data to be saved.
This input data needs to have the hierarchical structure of a Photon-HDF5 file. 
In practice we use a standard python dict: each keys is a Photon-HDF5 field name and
each value contains data (e.g. array, string, etc..) or another dict 
(in this case it represents an HDF5 sub-group). Similarly, sub-dictionaries 
contain data or other dict in order to represent the hierachy of Photon-HDF5 files.

The module `loader` contains loader functions that load data from disk and return a dict
object to be passed to `save_photon_hdf5()`. These functions can be used as examples
when converting a new unsupported file format.

The `loader` module contains high-level functions that "fill" the dict-based
with the appropriate arrays. The actual decoding of the input binary files is perfomed
by low-level functions in other modules (`smreader.py`, `pqreader.py`, `bhreader.py`).
Therefore when trying to decode a new file format you can look at those modules
for examples.

The phconvert repository also contains a JSON specification of the Photon-HDF5 format:

- [photon-hdf5_specs.json](https://github.com/Photon-HDF5/phconvert/blob/master/phconvert/specs/photon-hdf5_specs.json)


## License

*phconvert* is released under the open source MIT license.

