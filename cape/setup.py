# Packages
from distutils.core import setup, Extension
import ConfigParser
import os
import os.path as op


# Get a get/set type object
config = ConfigParser.SafeConfigParser()
# Read the configuration options
config.read("../config.cfg")

# Path to this file
fpwd = op.dirname(op.realpath(__file__))

# C compiler flags
cflags = config.get("compiler", "extra_cflags").split()

# Linker options
ldflags = config.get("compiler", "extra_ldflags").split()

# Extra include directories
include_dirs = config.get("compiler", "extra_include_dirs").split()

# Assemble the information for the module.
_pycart = Extension("_cape",
    include_dirs = include_dirs,
    extra_compile_args = cflags,
    extra_link_args = ldflags,
    sources = [
        "_capemodule.c",
        "pc_Tri.c"])


# Compile and link
setup(
    name="python-cart3d",
    version="0.1",
    description="This package provides some fast methods for CAPE",
    ext_modules=[_pycart])
