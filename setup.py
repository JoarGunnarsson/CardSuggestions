from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Sparse Vector module',
    ext_modules=cythonize("sparse.pyx"),
)
# python setup.py build_ext --inplace
