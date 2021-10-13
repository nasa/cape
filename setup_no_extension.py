#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import json
import os
import sys

# Standard library partial imports
from setuptools import setup


# Compile and link
setup(
    name="cape",
    packages=[
        "cape",
        "cape.attdb",
        "cape.attdb.ftypes",
        "cape.cfdx",
        "cape.cfdx.options",
        "cape.filecntl",
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
        "numpy>=1.4.1",
        "matplotlib>=2",
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
        "bin/pc_Step2Crv.py",
        "bin/pc_StepTri2Crv.py",
        "bin/pc_Tri2Surf.py",
        "bin/pf_Plt2Triq.py",
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
            "cape-uh3d2tri=cape.tricli:main_uh3d2tri",
            "cape-tri2uh3d=cape.tricli:main_tri2uh3d",
            "cape-tri2plt=cape.tricli:main_tri2plt",
        ],
    },
    version="1.0a3",
    description="CAPE computational aerosciences package")
