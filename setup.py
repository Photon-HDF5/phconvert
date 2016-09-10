from setuptools import setup
import versioneer

project_name = 'phconvert'

## Metadata
long_description = """
phconvert
==========

Convert Beker&Hickl, PicoQuant and other formats used in single-molecule
spectroscopy (e.g. smFRET, FCS, PIFE) to
`Photon-HDF5 <http://photon-hdf5.org/>`_ files.

Easy install via conda with::

    conda install phconvert -c conda-forge

See `Release Notes <https://github.com/Photon-HDF5/phconvert/releases>`__.

"""

setup(name = project_name,
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      author = 'Antonino Ingargiola',
      author_email = 'tritemio@gmail.com',
      url = 'http://photon-hdf5.github.io/phconvert/',
      download_url = 'http://photon-hdf5.github.io/phconvert/',
      install_requires = ['numpy', 'setuptools', 'tables', 'future'],
      include_package_data=True,
      license = 'MIT',
      description = ("Convert Beker&Hickl, PicoQuant and other formats to Photon-HDF5."),
      long_description = long_description,
      platforms = ('Windows', 'Linux', 'Mac OS X'),
      classifiers=['Intended Audience :: Science/Research',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3.4',
                   'Programming Language :: Python :: 3.5',
                   'Topic :: Scientific/Engineering',
                   ],
      packages = ['phconvert'],
      package_data = {'phconvert': ['specs/*.json']},
      keywords = ('single-molecule FRET smFRET biophysics file-format HDF5 '
                  'Photon-HDF5'),
      )
