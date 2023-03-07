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

# Python version infor
PY_MAJOR_VERSION = sys.version_info.major
PY_MINOR_VERSION = sys.version_info.minor

# Version-dependent imports
if PY_MAJOR_VERSION == 2:
    # Standard library modules
    from ConfigParser import SafeConfigParser
    # Change name
    ConfigParser = SafeConfigParser
else:
    # Standard library modules
    from configparser import ConfigParser

# Config file
fcfg = "config%i.cfg" % PY_MAJOR_VERSION


# Path to this file
fdir = os.path.dirname(os.path.realpath(__file__))
fcape = os.path.join(fdir, "cape")

# Get a get/set type object
config = ConfigParser()
# Read the configuration options
config.read(os.path.join(fcape, fcfg))

# C compiler flags
cflags = config.get("compiler", "extra_cflags").split()

# Linker options
ldflags = config.get("compiler", "extra_ldflags").split()

# Extra include directories
include_dirs = config.get("compiler", "extra_include_dirs").split()

# Extensions JSON file
extjson = os.path.join(fcape, "extensions.json")
# Read extension settings
extopts = json.load(open(extjson))

# Initialize extensions
exts = []
# Loop through specified extensions
for (ext, opts) in extopts.items():
    # Get sources
    extsources = [str(src) for src in opts["sources"]]
    # Create extension
    _ext = Extension(
        str(ext) + str(PY_MAJOR_VERSION),
        include_dirs=include_dirs,
        extra_compile_args=cflags,
        extra_link_args=ldflags,
        sources=extsources)
    # Add to list
    exts.append(_ext)

# Compile and link
setup(ext_modules=exts, **setup_py.SETUP_SETTINGS)
