#!/usr/bin/env python2

import ConfigParser
import subprocess as sp
import os, shutil, glob

config = ConfigParser.SafeConfigParser()
config.read("../config.cfg")

# Python command, in cases of potential ambiguity.
pythonexec = config.get("python", "exec")

# Status update.
print("Building Cart3D tools for CAPE and pyCart")

# Clean-up the existing build directory
if os.path.isdir("build"):
    shutil.rmtree("build", ignore_errors=True)

print("Executing setup...")
sp.call([pythonexec, "setup.py", "build"])

print("Moving the module into place...")
# Find all build folders.
dirs = glob.glob("build/lib*")
# There can be only one.
if len(dirs) > 1:
    raise ValueError("More than one build directory found.")
    
# Check for an existing _pycart.so object.
if os.path.isfile("_cape.so") or os.path.islink("_cape.so"):
    # Delete it!
    os.remove("_cape.so")
    
# Form the path to the relevant library file.
libdir = dirs[0]
lib = os.path.join(libdir, "_cape.so")
# Move it into the pyCart/ folder
shutil.move(lib, ".")

print "Removing the build directory..."
shutil.rmtree("build")
