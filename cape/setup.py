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

# Assemble the information for the ftypes extension module
_ftypes = distutils.core.Extension(
    "_ftypes",
    include_dirs = include_dirs,
    extra_compile_args = cflags,
    extra_link_args = ldflags,
    sources = [
        "src/_ftypesmodule.c",
        "src/capec_BaseFile.c",
        "src/cape_CSVFile.c"
    ])

# Assemble the information for the CAPE extension module
_cape = distutils.core.Extension("_cape",
    include_dirs = include_dirs,
    extra_compile_args = cflags,
    extra_link_args = ldflags,
    sources = [
        "src/_capemodule.c",
        "src/pc_io.c",
        "src/pc_Tri.c"
    ])


# Compile and link
distutils.core.setup(
    name="cape",
    packages=["cape"],
    package_dir={"cape": "."},
    version="1.0",
    description="CAPE computational aerosciences package",
    ext_modules=[_ftypes, _cape])
