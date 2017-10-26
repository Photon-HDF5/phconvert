[![Build status](https://ci.appveyor.com/api/projects/status/l6ncuo9hwqiul36y/branch/master?svg=true)](https://ci.appveyor.com/project/tritemio/phconvert/branch/master)
[![Build Status](https://travis-ci.org/Photon-HDF5/phconvert.svg?branch=master)](https://travis-ci.org/Photon-HDF5/phconvert)

# phconvert

*phconvert* is a python 2 & 3 library that helps writing valid
<a href="http://photon-hdf5.org/" target="_blank">Photon-HDF5</a>
files, a file format for time stamp-based single-molecule spectroscopy.
Additionally, *phconvert* can convert a few common binary formats
used in in single-molecule spectroscopy (PicoQuant .HT3/.PT3/.PTU,.T3R,
Becker & Hickl .SPC/.SET) to Photon-HDF5.

For questions or issues running this software please use the
[Photon-HDF5 Google Group](https://groups.google.com/forum/#!forum/photon-hdf5)
or open an [issue on GitHub](https://github.com/Photon-HDF5/phconvert/issues).

## What's new

**Oct. 2017:** Phconvert 0.8 relased, see the [release notes](https://github.com/Photon-HDF5/phconvert/releases/tag/0.8).


## Quick-start: Converting files to Photon-HDF5

Converting one of the supported files formats to Photon-HDF5 does not require
being able to program in python. All you need is running the "notebook"
corresponding to the file format you want to convert from, and follow the instructions therein.

For demonstration purposes, we provide [a demo service](http://photon-hdf5.github.io/Photon-HDF5-Converter)
to run the notebooks online without any installation.
With this online service, you can convert data files up to 35MB to Photon-HDF5.
To launch the demo click on the following button
(see also [instructions](http://photon-hdf5.github.io/Photon-HDF5-Converter/)):

[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/Photon-HDF5/Photon-HDF5-Converter)

To execute the phconvert notebooks on your machine, you need to install the *Jupyter Notebook App* first.
A quick-start guide on installing and running the *Jupyter Notebook App* is available here:

- <a href="http://jupyter-notebook-beginner-guide.readthedocs.org/" target="_blank">Jupyter/IPython Notebook Quick Start Guide</a>

Next, you need to install the *phconvert* library with the following command
(type it in *Terminal* on OS X or Linux, or in the `cmd` prompt on Windows):

    conda install -c conda-forge phconvert

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

When writing Photon-HDF5 files, phconvert saves you time
and protects you against common errors that risk
to make the file not a valid Photon-HDF5. Also a description
is automatically added to each Photon-HDF5 field.
The descriptions are extracted from a [JSON file](https://github.com/Photon-HDF5/phconvert/blob/master/phconvert/specs/photon-hdf5_specs.json)
which contains the list Photon-HDF5 field names, types, and descriptions.

See also [Writing Photon-HDF5 files](http://photon-hdf5.readthedocs.org/en/latest/writing.html)
in the Photon-HDF5 reference documentation.

## Read Photon-HDF5 files

In case you just want to read Photon-HDF5 files you don't need to use phconvert.
Photon-HDF5 files can be directly opened with a standard HDF5 viewer
[HDFView](https://www.hdfgroup.org/products/java/hdfview/).

See also [Reading Photon-HDF5 files](http://photon-hdf5.readthedocs.org/en/latest/reading.html)
in the Photon-HDF5 reference documentation.

## Installation

The recommended way to install *phconvert* is using conda:

    conda install -c conda-forge phconvert

If you don't have conda installed, please install the free python distribution
[Anaconda](https://store.continuum.io/cshop/anaconda/) choosing the python 3
version. The legacy python 2.7 is also supported, but if you can choose
pick the latest python 3.

Alternatively, you can install *phconvert* in any python installation using PIP:

    pip install phconvert

In this latter case, make sure that numpy and pytables are installed.

See also:

- [phconvert Release Notes](https://github.com/Photon-HDF5/phconvert/releases/).

## Dependencies

- python 2.7 (legacy), 3.4 or greater (recommended)
- future
- numpy >=1.9
- pytables >=3.1
- numba (optional) *to enable a fast HT3 file reader*

> **Note**
> when installing via `conda` all the dependencies are automatically installed.


## The phconvert library documentation (for developers)

The *phconvert* API documentation can be found on ReadTheDocs:

- [phconvert's documentation](http://phconvert.readthedocs.org/)

## License

*phconvert* is released under the open source [MIT license](https://raw.githubusercontent.com/Photon-HDF5/phconvert/master/LICENSE.txt).

## Contributing

As with other Photon-HDF5 subprojects, we encourage contributions
in any form, from simple suggestions, typo fix to the addition of new features.
Please use GitHub by opening Issues or sending Pull Requests.

All the contributors will be acknowledged in this website, and will included
as authors in the next software-paper publication.

For more details see our [contribution policy](http://photon-hdf5.readthedocs.org/en/latest/contributing.html).

## Authors & Contributors

List of contributors (by lines of code):

- Antonino Ingargiola (@tritemio) <tritemio AT gmail.com>
- Ted Laurence (@talaurence) <laurence2 AT llnl.gov>
- Marco Lamperti (@lampo808) <marco.lampo AT gmail.com>
- Xavier Michalet (@smXplorer) <michalet AT chem.ucla.edu>
- Anders Barth (@AndersBarth) <anders.barth AT gmail.com>
- Biswajit Pradhan (@biswajitSM) <biswajitp145 AT gmail.com.

We thank also @ncodina for providing PTU files and helping in testing
the PTU decoder in phconvert.

## Acknowledgements
This work was supported by NIH Grant R01-GM95904.
