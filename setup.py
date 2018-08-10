from setuptools import setup
from setuptools import find_packages

setup(name='drssrn',
      version='0.0.1',
      install_requires=['numpy>=1.9.1',
                        'scipy>=0.14',
                        'six>=1.9.0',
                        'pyyaml',
                        'h5py',
                        'keras_applications==1.0.2',
                        'keras_preprocessing==1.0.1'],
      extras_require={
          'visualize': ['pydot>=1.2.4'],
          'tests': ['pytest',
                    'pytest-pep8',
                    'pytest-xdist',
                    'pytest-cov',
                    'pandas',
                    'requests'],
      },
      classifiers=[
          'Programming Language :: Python :: 3.6',
      ],
      packages=find_packages())
