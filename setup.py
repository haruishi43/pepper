#!/usr/bin/env python3

import os.path as osp
from setuptools import find_packages, setup, Extension  # noqa
from Cython.Build import cythonize

import numpy as np

from pepper import __version__


def readme():
    with open("README.md") as f:
        content = f.read()
    return content


def numpy_include():
    try:
        numpy_include = np.get_include()
    except AttributeError:
        numpy_include = np.get_numpy_include()
    return numpy_include


ext_modules = [
    Extension(
        "pepper.core.evaluation.rank_cylib.rank_cy",
        sources=["pepper/core/evaluation/rank_cylib/rank_cy.pyx"],
        include_dirs=[numpy_include()],
    ),
    Extension(
        "pepper.core.evaluation.rank_cylib.roc_cy",
        sources=["pepper/core/evaluation/rank_cylib/roc_cy.pyx"],
        include_dirs=[numpy_include()],
    ),
]


def get_requirements(filename="requirements.txt"):
    here = osp.dirname(osp.realpath(__file__))
    with open(osp.join(here, filename), "r") as f:
        requires = [line.replace("\n", "") for line in f.readlines()]
    return requires


setup(
    name="pepper",
    version=__version__,
    description="Pepper: Yet Another Framework for Image and Video Re-ID",
    author="Haruya Ishikawa",
    license="",
    long_description=readme(),
    url="",
    packages=find_packages(
        exclude=[
            ".readme",
            "configs",
            "data",
            "notebooks",
            "tests",
            "tools",
            "work_dirs",
            "projects",
        ],
    ),
    install_requires=get_requirements(),
    keywords=["Computer Vision", "Re-ID", "Metric Learning"],
    ext_modules=cythonize(
        ext_modules,
        language_level="3",
    ),
)
