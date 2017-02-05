from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
	ext_modules = cythonize("DTW_pyx.pyx")
	,include_dirs=[numpy.get_include()],
)

#type in command line:$ python setup.py build_ext --inplace