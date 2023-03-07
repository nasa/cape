#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import json
import os
import sys

# Third-party
from setuptools import Extension, setup

# Local imports
from cape import setup_py
setup(
    name="cape",
    packages=[
        "cape",
        "cape.attdb",
        "cape.attdb.ftypes",
        "cape.cfdx",
        "cape.cfdx.options",
        "cape.filecntl",
        "cape.optdict",
        "cape.pycart",
        "cape.pycart.options",
        "cape.pyfun",
        "cape.pyfun.options",
        "cape.pyover",
        "cape.pyover.options",
        "cape.tnakit",
        "cape.tnakit.plot_mpl",
        "cape.tnakit.textutils"
    ],
    install_requires=[
        "defusedxml",
        "numpy>=1.4.1",
        "matplotlib>=2",
        "scipy",
        "vendorize",
        "xlrd%i" % PY_MAJOR_VERSION,
        "xlsxwriter"
    ],
    package_data={
        "cape": [
            "templates/paraview/*",
            "templates/tecplot/*"
        ],
        "cape.cfdx": ["templates/*"],
        "cape.cfdx.options": ["*.json"],
        "cape.pycart": ["templates/*"],
        "cape.pycart.options": ["*.json"],
        "cape.pyfun": ["templates/*"],
        "cape.pyfun.options": ["*.json"],
        "cape.pyover": ["templates/*"],
        "cape.pyover.options": ["*.json"],
    },
    scripts=[
        "bin/run_flowCart.py",
        "bin/run_fun3d.py",
        "bin/run_overflow.py"
    ],
    entry_points={
        "console_scripts": [
            "pycart=cape.pycart.cli:main",
            "pyfun=cape.pyfun.cli:main",
            "pyover=cape.pyover.cli:main",
            "dkit=cape.attdb.cli:main",
            "dkit-quickstart=cape.attdb.quickstart:main",
            "dkit-vendorize=cape.attdb.vendorutils:main",
            "dkit-writedb=cape.attdb.writedb:main",
            "cape-writell=cape.writell:main",
            "cape-step2crv=cape.tricli:main_step2crv",
            "cape-steptri2crv=cape.tricli:main_steptri2crv",
            "cape-uh3d2tri=cape.tricli:main_uh3d2tri",
            "cape-tri2uh3d=cape.tricli:main_tri2uh3d",
            "cape-tri2plt=cape.tricli:main_tri2plt",
            "cape-tri2surf=cape.tricli:main_tri2surf",
            "pyfun-plt2triq=cape.pyfun.tricli:main_plt2triq",
        ],
    },
    version="1.1.0-prelim4",
    description="CAPE computational aerosciences package",
    ext_modules=exts)
setup(ext_modules=exts, **setup_py.SETUP_SETTINGS)
