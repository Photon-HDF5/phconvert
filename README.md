# phconvert

*phconvert* is a python 2 & 3 library for writing and reading
<a href="http://photon-hdf5.org/" target="_blank">Photon-HDF5</a>
files, a file format for time stamp-based single-molecule spectroscopy.
Additionally, *phconvert* can convert a few common binary formats
used in in single-molecule spectroscopy (PicoQuant .HT3,
Becker & Hickl .SPC/.SET) to Photon-HDF5.

## Quick-start: Converting files to Photon-HDF5

Converting one of the supported files formats to Photon-HDF5 does not require 
being able to program in python. All you need is running the appropriate "notebook"
corresponding to the source format and follow the instructions therein.

To run a notebook on your machine, you need to install the *Jupyter Notebook App*. 
A quick-start guide on installing and running the *Jupyter Notebook App* is available here:

- <a href="http://jupyter-notebook-beginner-guide.readthedocs.org/" target="_blank">Jupyter/IPython Notebook Quick Start Guide</a>

Next, you need to install the *phconvert* library with the command (typed in a Terminal window for Mac and Linux, or in the cmd prompt on Windows):

    conda install -c tritemio phconvert
    
Finally, you can download one of the provided notebooks and run it on your machine.
To download all notebooks in one step, download the 
[phconvert zip](https://github.com/Photon-HDF5/phconvert/archive/master.zip), 
which contains all the notebooks in the `notebooks` subfolder.

## Project details

*phconvert* repository contains a python package (library) and a set of
[notebooks](https://github.com/Photon-HDF5/phconvert/tree/master/notebooks) 
([online viewer](http://nbviewer.ipython.org/github/Photon-HDF5/phconvert/tree/master/notebooks/)) 
which show how to convert other file formats to Photon-HDF5.

*phconvert* tests the compliance to the Photon-HDF5 specifications
before saving a new files and automatically adds description 
attributes for each field.

## Read Photon-HDF5 files

In case you just want to read Photon-HDF5 files, phconvert is not
necessary (although it provides some helper functions).
Photon-HDF5 files can be opened with a standard HDF5 viewer
[HDFView](https://www.hdfgroup.org/products/java/hdfview/).
Moreover, we provide code examples on reading Photon-HDF5 files
in multiple languages in 
[this repository](https://github.com/Photon-HDF5/photon_hdf5_reading_examples).

## Installation

The recommended way to install *phconvert* is by using conda (which first requires installing the python distribution [Anaconda](https://store.continuum.io/cshop/anaconda/) from Continuum):

    conda install -c tritemio phconvert

You can also install *phconvert* through PIP (which first requires installing python, and the numpy and pytables libraries):

    pip install phconvert

Finally, another way consists in downloading the sources and executing:

    python setup.py build
    python setup.py install

Note: all dependencies need to be installed first.

## Dependencies

- python 2.7, 3.3 or greater
- future
- numpy >=1.9
- pytables >=3.1
- numba (optional) *to enable a fast HT3 file reader*

> **Note**
> when installing via `conda` all the dependencies are automatically installed.


## The phconvert library (for developers)

The *phconvert* library contains two main modules: `hdf5` and `loader`. The former contains 
the function `save_photon_hdf5()` which is used to create Photon-HDF5 files.

The `save_photon_hdf5()` function requires the data to be saved as argument.
The data needs to have the hierarchical structure of a Photon-HDF5 file. 
In practice, we use a standard python dictionary: each keys is a Photon-HDF5 field name and
each value contains data (e.g. array, string, etc..) or another dictionary
(in which case, it represents an HDF5 sub-group). Similarly, sub-dictionaries 
contain data or other dictionaries, as needed to represent the hierachy of Photon-HDF5 files.

The `loader` module contains loader functions which load data from disk and return a dictionary
to be passed to `save_photon_hdf5()`. These functions can be used as examples
when converting a new unsupported file format.

The `loader` module contains high-level functions which "fill" the dictionary
with the appropriate arrays. The actual decoding of the input binary files is perfomed
by low-level functions in other modules (`smreader.py`, `pqreader.py`, `bhreader.py`).
When trying to decode a new file format, these modules can provide useful examples.

The phconvert repository also contains a JSON specification of the Photon-HDF5 format:

- [photon-hdf5_specs.json](https://github.com/Photon-HDF5/phconvert/blob/master/phconvert/specs/photon-hdf5_specs.json)


## License

*phconvert* is released under the open source MIT license.


## Acknowledgements
This work was supported by NIH Grant R01-GM95904.

