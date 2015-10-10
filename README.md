# phconvert

*phconvert* is a python 2 & 3 library that helps writing valid
<a href="http://photon-hdf5.org/" target="_blank">Photon-HDF5</a>
files, a file format for time stamp-based single-molecule spectroscopy.
Additionally, *phconvert* can convert a few common binary formats
used in in single-molecule spectroscopy (PicoQuant .HT3,
Becker & Hickl .SPC/.SET) to Photon-HDF5.

For questions or issues running this software please use the 
[Photon-HDF5 Google Group](https://groups.google.com/forum/#!forum/photon-hdf5)
or open an [issue on GitHub](https://github.com/Photon-HDF5/phconvert/issues).

## Quick-start: Converting files to Photon-HDF5

Converting one of the supported files formats to Photon-HDF5 does not require 
being able to program in python. All you need is running the "notebook"
corresponding to the file format you want to convert from, and follow the instructions therein.

For demonstration purposes, the notebooks can be executed online (courtesy [MyBinder.org](http://mybinder.org/)) 
to convert data files (currently limited to 35MB) to Photon-HDF5 without any installation.
To launch the demo click on the following button:

[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/Photon-HDF5/Photon-HDF5-Converter)

To execute the phconvert notebooks on your machine, you need to install the *Jupyter Notebook App* first. 
A quick-start guide on installing and running the *Jupyter Notebook App* is available here:

- <a href="http://jupyter-notebook-beginner-guide.readthedocs.org/" target="_blank">Jupyter/IPython Notebook Quick Start Guide</a>

Next, you need to install the *phconvert* library with the following command 
(type it in *Terminal* on OS X or Linux, or in the `cmd` prompt on Windows):

    conda install -c tritemio phconvert
    
Finally, you can download one of the provided notebooks and run it on your machine.
Simply, download the 
[phconvert zip](https://github.com/Photon-HDF5/phconvert/archive/master.zip), 
which contains all the notebooks in the `notebooks` subfolder.

###For questions or issues:

- [Open an GitHub issue](https://github.com/Photon-HDF5/phconvert/issues) or 
- Ask a question on the [Photon-HDF5 Google Group](https://groups.google.com/forum/#!forum/photon-hdf5).


## Project details

### What's inside?

*phconvert* repository contains a python package (library) and a set of
[notebooks](https://github.com/Photon-HDF5/phconvert/tree/master/notebooks) 
([online viewer](http://nbviewer.ipython.org/github/Photon-HDF5/phconvert/tree/master/notebooks/)).
Each notebook can convert a different format to Photon-HDF5 using the phconvert library.

If you have a file format that is not yet supported, please [open an new Issue](https://github.com/Photon-HDF5/phconvert/issues).
We are willing add support for as many file formats as possible!

### Why phconvert?

When wrtiting Photon-HDF5 files, phconvert saves you time 
and protects you againsts common programming errors that risk
to make the file not a valid Photon-HDF5.

In principle, Photon-HDF5 files can be written directly using only pytables 
or other HDF5 libraries.
In practice, the process of manually creating each field is tedious and error prone:
it's easy to make spelling errors in field names or to forget a mandatory field.
To help this task, we provide a 
[JSON file](https://github.com/Photon-HDF5/phconvert/blob/master/phconvert/specs/photon-hdf5_specs.json) 
which contains the list Photon-HDF5 field names, types, and descriptions.
*phconvert*, uses this JSON file to make sure that the input field names (and types) 
are valid. Additionally, phconvert adds a description to each field, 
extracting it from the JSON file.

## Read Photon-HDF5 files

In case you just want to read Photon-HDF5 files you don't need to use phconvert.
Photon-HDF5 files can be directly opened with a standard HDF5 viewer
[HDFView](https://www.hdfgroup.org/products/java/hdfview/).
Moreover, we provide code examples of reading Photon-HDF5 files
in multiple languages in 
[this repository](https://github.com/Photon-HDF5/photon_hdf5_reading_examples).

## Installation

The recommended way to install *phconvert* is by using conda (which first 
requires installing the free python distribution 
[Anaconda](https://store.continuum.io/cshop/anaconda/) from Continuum):

    conda install -c tritemio phconvert

Alternatively, you can install *phconvert* in any python installation using PIP:

    pip install phconvert

In this latter case, make sure that numpy and pytables are installed.

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
contain data or other dictionaries, as needed to represent the hierarchy of Photon-HDF5 files.

The `loader` module contains loader functions which load data from disk and return a dictionary
to be passed to `save_photon_hdf5()`. These functions can be used as examples
when converting a new unsupported file format.

The `loader` module contains high-level functions which "fill" the dictionary
with the appropriate arrays. The actual decoding of the input binary files is performed
by low-level functions in other modules (`smreader.py`, `pqreader.py`, `bhreader.py`).
When trying to decode a new file format, these modules can provide useful examples.

The phconvert repository also contains a JSON specification of the Photon-HDF5 format:

- [photon-hdf5_specs.json](https://github.com/Photon-HDF5/phconvert/blob/master/phconvert/specs/photon-hdf5_specs.json)


## License

*phconvert* is released under the open source MIT license.

## Contributing

As with other Photon-HDF5 subprojects, we encourage contributions 
in any form, from simple suggestions, typo fix to the addition of new features. 
Please use GitHub by opening Issues or sending Pull Requests.

All the contributors will be acknowledged in this website, and will included 
as authors in the next software-paper publication.

For more details see our [contribution policy](http://photon-hdf5.readthedocs.org/en/latest/contributing.html).

## Acknowledgements
This work was supported by NIH Grant R01-GM95904.

