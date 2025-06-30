from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [Extension("sparse", ["sparse.pyx"])]

setup(
    ext_modules=cythonize(extensions),
)

# python setup.py build_ext --inplace

