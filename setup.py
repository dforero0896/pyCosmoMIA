# setup.py
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import Cython.Compiler.Options
#Cython.Compiler.Options.annotate = True
import numpy as np

extensions = [
    Extension(
        name="cosmomia.cosmomia",
        sources=["cosmomia/subgrid.pyx"],#, "cosmomia/src/utils.cpp"],#, "cosmomia/src/subgrid.cpp"],
        language="c++",
        include_dirs=[np.get_include()],#, '-I/home/astro/dforero/codes/bosque/examples/c'],
        extra_compile_args=["-std=c++11", '-fopenmp'],
        extra_link_args=['-fopenmp', '-march=native'],#, '-lbosque', '-L/home/astro/dforero/codes/bosque/target/release/'],
    )
]

setup(
    name="cosmomia",
    author = "Daniel Forero",
    ext_modules=cythonize(extensions, annotate = False),
    packages = find_packages(exclude = ['examples', 'tests']),
    install_requires = ['numpy', 'pykdtree', 'jax', 'jaxlib']
)
