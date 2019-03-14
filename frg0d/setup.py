from distutils.core import setup
from Cython.Build import cythonize

setup(name='auxFunctions',ext_modules=cythonize('auxFunctions.pyx'),)
setup(name='interpP1F',ext_modules=cythonize('interpP1F.pyx'),)
