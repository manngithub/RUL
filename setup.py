from __future__ import print_function
import os.path
import sys
import setuptools
from numpy.distutils.core import setup



try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)


DISTNAME = 'SAP'
DESCRIPTION = "Remaining Useful Life Estimation using LSTM Networks"
MAINTAINER = 'Manoj Kumar'
MAINTAINER_EMAIL = 'mann.dhiman@gmail.com'
VERSION = '0.1'
REQUIREMENTS=  [ "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "keras",
        "tensorflow",
        "h5py",
    ],


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)

    config.add_subpackage('SAP')

    return config


if __name__ == '__main__':
    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))

    os.chdir(local_path)
    sys.path.insert(0, local_path)

    setup(configuration=configuration,
          name=DISTNAME,
          maintainer=MAINTAINER,
          include_package_data=True,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          version=VERSION,
          packages=['SAP.Data'],
          package_data={
   	        '': ['*'],     # All files
   		     },
          zip_safe=False,
          install_requires=REQUIREMENTS,
          classifiers=[
              'Intended Audience :: Science/Research',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX', 'Operating System :: Unix',

             ]
    )