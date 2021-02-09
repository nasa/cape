#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import os
import json
import ConfigParser

# Standard library partial imports
from setuptools import Extension, setup


# Path to this file
fdir = os.path.dirname(os.path.realpath(__file__))
fcape = os.path.join(fdir, "cape")

# Get a get/set type object
config = ConfigParser.SafeConfigParser()
# Read the configuration options
config.read(os.path.join(fcape, "config2.cfg"))

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
    # Get sources (ensure not :class:`unicode` objects)
    extsources = [
        str(src) for src in opts["sources"]
    ]
    # Create extension
    _ext = Extension(
        str(ext) + "2",
        include_dirs = include_dirs,
        extra_compile_args = cflags,
        extra_link_args = ldflags,
        sources = extsources)
    # Add to list
    exts.append(_ext)

# Compile and link
setup(
    name="cape",
    version="1.0a1",
    packages=[
        "cape"
    ],
    description="CAPE computational aerosciences package",
    ext_modules=exts)
