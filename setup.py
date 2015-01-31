from setuptools import setup
import versioneer

project_name = 'phconvert'

## Configure versioneer
versioneer.VCS = 'git'
versioneer.versionfile_source = project_name + '/_version.py'
versioneer.versionfile_build = project_name + '/_version.py'
versioneer.tag_prefix = '' # tags are like 1.2.0
versioneer.parentdir_prefix = project_name + '-'


## Metadata
long_description = """
phconvert
==========

Converter for `Photon-HDF5 <http://photon-hdf5.readthedocs.org/>`_ file format.

"""

setup(name = project_name,
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
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

