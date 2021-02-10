#!/usr/bin/env python2.7
#-*- coding: utf-8 -*-

# Standard library modules
import ConfigParser
import glob
import json
import os
import platform
import sys
import shutil
import subprocess as sp

# Standard library direct imports
from distutils import sysconfig


# System configuration variables
# sysplatform = sysconfig.get_platform()
syssystem = platform.system().lower()
sysmachine = platform.machine()
sysplatform = "%s-%s" % (syssystem, sysmachine)
syspyversion = sysconfig.get_python_version()
# Suffix for build folders
libext = "%s-%s" % (sysplatform, syspyversion)
# Library folder
flib = os.path.join("build", "lib.%s" % libext)
ftmp = os.path.join("build", "temp.%s" % libext)

# Path to this file
fdir = os.path.dirname(os.path.realpath(__file__))
# Module file
fmod = os.path.join(fdir, "cape")

# Extensions JSON file
extjson = os.path.join(fmod, "extensions.json")
# Read extension settings
extopts = json.load(open(extjson))

# Get a get/set type object
config = ConfigParser.SafeConfigParser()
# Read the configuration options
config.read(os.path.join(fmod, "config2.cfg"))

# Python command, in cases of potential ambiguity.
pythonexec = config.get("python", "exec")

# Compile
print("Building extensions...")
sp.call([pythonexec, "setup2.py", "build"])

# Check for build
if not os.path.isdir(flib):
    print("Extension build FAILED:")
    print("  No build folder '%s' found" % flib)
    sys.exit(1)

# Creating wheel
print("Building wheel...")
sp.call([pythonexec, "setup2.py", "bdist_wheel"])

# Status update
print("Moving the extensions into place...")
# Loop through extensions
for (ext, opts) in extopts.items():
    # Destination folder
    fdest = opts["destination"].replace("/", os.sep)
    # File name for compiled module
    fname = "%s2.so" % ext
    # Final location for module
    fout = os.path.join(fmod, fdest, fname)
    # Expected build location
    fbld = os.path.join(fdir, flib, fname)
    # Exit if no build
    if not os.path.isfile(fbld):
        print("Build of extension '%s' failed" % ext)
        sys.exit(1)
    # Check for existing object
    if os.path.isfile(fout):
        os.remove(fout)
    # Move the result to the destination folder
    shutil.move(fbld, fout)

# Remove that "egg" thing
print("Removing the egg-info folder")
shutil.rmtree("cape.egg-info")
