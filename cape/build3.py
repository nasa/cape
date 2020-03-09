#!/usr/bin/env python3

# Standard library modules
import os
import sys
import json
import glob
import shutil
import configparser
import subprocess as sp


# Path to this file
fpwd = os.path.dirname(os.path.realpath(__file__))

# Extensions JSON file
extjson = os.path.join(fpwd, "extensions.json")
# Read extension settings
extopts = json.load(open(extjson))

# Get a get/set type object
config = configparser.SafeConfigParser()
# Read the configuration options
config.read(os.path.join(fpwd, "config3.cfg"))

# Python command, in cases of potential ambiguity.
pythonexec = config.get("python", "exec")

# Status update
print("Building compiled functions for ftypes")

# Clean-up the existing build directory
if os.path.isdir("build"):
    shutil.rmtree("build", ignore_errors=True)

# Compile
print("Executing setup...")
sp.call([pythonexec, "setup3.py", "build"])
# Status update
print("Moving the module into place...")

# Find all build folders
dirs = glob.glob("build/lib*3.6*")
# There can be only one
if len(dirs) > 1:
    raise ValueError("More than one build directory found.")
# Compilation folder
flib = dirs[0]

# Loop through extensions
for (ext, opts) in extopts.items():
    # Destination folder
    fdest = opts["destination"].replace("/", os.sep)
    # File name for compiled module
    fname = "%s3.so" % ext
    # Final location for module
    fmod = os.path.join(fpwd, fdest, fout)
    # Expected build location
    fbld = os.path.join(fpwd, flib, fname)
    # Exit if no build
    if not os.path.isfile(fbld):
        print("Build of extension '%s' failed" % ext)
        sys.exit(1)
    # Check for existing object
    if os.path.isfile(fmod):
        os.remove(fmod)
    # Move the result to the destination folder
    shutil.move(fbld, fmod)

#print("Removing the build directory...")
#shutil.rmtree("build")
