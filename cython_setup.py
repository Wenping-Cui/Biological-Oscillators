#python cython_setup.py build_ext --inplace

filename = 'Oscillation_lib/Activator_inhibitor_cython.pyx'

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


setup(
    ext_modules=cythonize(filename),
    include_dirs=[numpy.get_include()]
)    


filename = 'Oscillation_lib/Circle_cython.pyx'

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


setup(
    ext_modules=cythonize(filename),
    include_dirs=[numpy.get_include()]
)    


filename = 'Oscillation_lib/Brusselator_cython.pyx'

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


setup(
    ext_modules=cythonize(filename),
    include_dirs=[numpy.get_include()]
)    

