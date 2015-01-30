from setuptools import setup

## Metadata
long_description = """
phconvert
==========

Converter for `Photon-HDF5 <http://photon-hdf5.readthedocs.org/>`_ file format.

"""

def get_version():
    with open('phconvert/__init__.py') as f:
        for line in f.readlines():
            if line.startswith('__version__'):
                linep = line.strip().replace(' ', '')
                version = linep[len('__version__='):][1:-1]
                return version
        raise ValueError('No version found in phconvert/__init__.py')


setup(name = 'phconvert',
      version = get_version(),
      author = 'Antonino Ingargiola',
      author_email = 'tritemio@gmail.com',
      url          = 'http://github.com/tritemio/phconvert/',
      download_url = 'http://github.com/tritemio/phconvert/',
      install_requires = ['numpy'],
      license = 'GPLv2',
      description = ("Converter for the Photon-HDF5 file format."),
      long_description = long_description,
      platforms = ('Windows', 'Linux', 'Mac OS X'),
      classifiers=['Intended Audience :: Science/Research',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 2.7',
                   'Topic :: Scientific/Engineering',
                   ],
      packages = ['phconvert'],
      keywords = ('single-molecule FRET smFRET biophysics file-format HDF5 '
                  'Photon-HDF5'),
      )

