# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True
import numpy as np

extensions = [
    Extension(
        name="cosmomia",
        sources=["cosmomia/subgrid.pyx"],#, "cosmomia/src/subgrid.cpp"],
        language="c++",
        include_dirs=[np.get_include(), '-I/home/astro/dforero/codes/bosque/examples/c'],
        extra_compile_args=["-std=c++11", '-fopenmp'],
        extra_link_args=['-fopenmp', '-lbosque', '-L/home/astro/dforero/codes/bosque/target/release/', '-march=native'],
    )
]

setup(
    name="cosmomia",
    ext_modules=cythonize(extensions, annotate = True),
)
