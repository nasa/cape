# Packages
from distutils.core import setup, Extension
import ConfigParser
import os.path as op


# Get a get/set type object
config = ConfigParser.SafeConfigParser()
# Read the configuration options
config.read("../config.cfg")

# Path to this file
directory = op.dirname(op.realpath(__file__))

# Compiler and linker options
cflagstrs = config.get("compiler", "extra_cflags")
cflags = [str(x) for x in cflagstrs.split(' ')]

ldflagstrs = config.get("compiler", "extra_ldflags")
ldflags = [str(x) for x in ldflagstrs.split(' ')]


# Assemble the information for the module.
_pycart = Extension("_pycart",
    include_dirs = [],
    sources = [
        "_pycartmodule.c",
        "pc_Tri.c"])


# Compile and link
setup(
    name="python-cart3d",
    version="1.0",
    description="This package provides some fast methods for pyCart",
    ext_modules=[_pycart])
