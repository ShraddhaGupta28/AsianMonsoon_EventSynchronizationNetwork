# -*- coding: utf-8 -*-
# The lower left point in grid cells is the 1st.
import numpy as np
from distutils.core import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension("Event_Sync_Null_Model_Cy", ["Event_Sync_Null_Model_Cy.pyx"],
              include_dirs=[np.get_include()],
              library_dirs=[],
              # windows
              libraries=[],
              extra_compile_args=['/openmp'],
              ),
    Extension("Event_Sync2_Null_Model_Cy", ["Event_Sync2_Null_Model_Cy.pyx"],
              include_dirs=[np.get_include()],
              library_dirs=[],
              # windows
              libraries=[],
              extra_compile_args=['/openmp'],
              ),
    Extension("Event_Sync_Udw_Cy", ["Event_Sync_Udw_Cy.pyx"],
              include_dirs=[np.get_include()],
              library_dirs=[],
              # windows
              libraries=[],
              extra_compile_args=['/openmp'],
              ),
    Extension("Task1_Ud_ES_Construction", ["Task1_Ud_ES_Construction.pyx"],
              include_dirs=[np.get_include()],
              library_dirs=[],
              # windows
              libraries=[],
              extra_compile_args=['/openmp'],
              ),
    Extension("Task1_Ud_ES_Regional_Sync_Corr", ["Task1_Ud_ES_Regional_Sync_Corr.pyx"],
              include_dirs=[np.get_include()],
              library_dirs=[],
              # windows
              libraries=[],
              extra_compile_args=['/openmp'],
              ),
]
setup(
    name="ISM_EASM_IJC",
    ext_modules=cythonize(extensions,
                          language_level='3',
                          build_dir="build",
                          ),
)
