#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import os
import json
import configparser
import distutils.core


# Path to this file
fpwd = os.path.dirname(os.path.realpath(__file__))

# Get a get/set type object
config = configparser.SafeConfigParser()
# Read the configuration options
config.read(os.path.join(fpwd, "config3.cfg"))

# C compiler flags
cflags = config.get("compiler", "extra_cflags").split()

# Linker options
ldflags = config.get("compiler", "extra_ldflags").split()

# Extra include directories
include_dirs = config.get("compiler", "extra_include_dirs").split()

# Extensions JSON file
extjson = os.path.join(fpwd, "extensions.json")
# Read extension settings
extopts = json.load(open(extjson))

# Initialize extensions
exts = []
# Loop through specified extensions
for (ext, opts) in extopts.items():
    # Get sources
    extsources = [str(src) for src in opts["sources"]]
    # Create extension
    _ext = distutils.core.Extension(
        str(ext) + "3",
        include_dirs = include_dirs,
        extra_compile_args = cflags,
        extra_link_args = ldflags,
        sources = extsources)
    # Add to list
    exts.append(_ext)

# Compile and link
distutils.core.setup(
    name="cape",
    packages=["cape"],
    package_dir={"cape": "."},
    version="1.0",
    description="CAPE computational aerosciences package",
    ext_modules=exts)
