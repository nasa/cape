#!/usr/bin/env python

# Standard library modules
import os
import glob
import shutil
import ConfigParser
import subprocess as sp


# Path to this file
fpwd = os.path.dirname(os.path.realpath(__file__))

# Get a get/set type object
config = ConfigParser.SafeConfigParser()
# Read the configuration options
config.read(os.path.join(fpwd, "config.cfg"))

# Python command, in cases of potential ambiguity.
pythonexec = config.get("python", "exec")

# Status update
print("Building compiled functions for ftypes")

# Clean-up the existing build directory
if os.path.isdir("build"):
    shutil.rmtree("build", ignore_errors=True)

# Compile
print("Executing setup...")
sp.call([pythonexec, "setup.py", "build"])
# Status update
print("Moving the module into place...")

# Find all build folders
dirs = glob.glob("build/lib*")
# There can be only one
if len(dirs) > 1:
    raise ValueError("More than one build directory found.")
    
# Check for an existing _pycart.so object.
if os.path.isfile("_ftypes.so") or os.path.islink("_ftypes.so"):
    # Delete it!
    os.remove("_ftypes.so")
    
# Form the path to the relevant library file.
libdir = dirs[0]
lib = os.path.join(libdir, "_ftypes.so")

# Move it into the main folder
shutil.move(lib, ".")

#print("Removing the build directory...")
#shutil.rmtree("build")
