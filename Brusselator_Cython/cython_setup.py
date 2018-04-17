#python cython_setup.py build_ext --inplace

filename = 'Circle_cython.pyx'

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


setup(
    ext_modules=cythonize(filename),
    include_dirs=[numpy.get_include()]
)    
