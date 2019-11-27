#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import os
import ConfigParser
import distutils.core


# Path to this file
fpwd = os.path.dirname(os.path.realpath(__file__))

# Get a get/set type object
config = ConfigParser.SafeConfigParser()
# Read the configuration options
config.read(os.path.join(fpwd, "config.cfg"))

# C compiler flags
cflags = config.get("compiler", "extra_cflags").split()

# Linker options
ldflags = config.get("compiler", "extra_ldflags").split()

# Extra include directories
include_dirs = config.get("compiler", "extra_include_dirs").split()

# Assemble the information for the module.
_ftypes = distutils.core.Extension(
    "_ftypes",
    include_dirs = include_dirs,
    extra_compile_args = cflags,
    extra_link_args = ldflags,
    sources = [
        "_ftypesmodule.c",
        "src/capec_BaseFile.c",
        "src/cape_CSVFile.c"
    ])


# Compile and link
distutils.core.setup(
    name="cape-ftypes-extension",
    packages=["ftypes"],
    package_dir={"ftypes": "."},
    version="1.0",
    description="This package provides data file interfaces",
    ext_modules=[_ftypes])
