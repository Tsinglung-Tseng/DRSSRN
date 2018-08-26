from setuptools import setup
from setuptools import find_packages

setup(name='drssrn',
      version='0.0.1',
      install_requires=['numpy>=1.9.1',
                        'scipy>=0.14',
                        'six>=1.9.0',
                        'tables',
                        'h5py'],
      namespace_packages=['DRSSRN'],
      classifiers=[
          'Programming Language :: Python :: 3.6',
      ],
      packages=find_packages())
