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
        include_dirs=[np.get_include()],
        extra_compile_args=["-std=c++11"]
    )
]

setup(
    name="cosmomia",
    ext_modules=cythonize(extensions, annotate = True),
)
