#!/usr/bin/env python
#-*- coding: utf-8 -*-

# Standard library modules
import json
import os
import platform
import sys
import shutil
import subprocess as sp

# Standard library direct imports
from distutils import sysconfig


# Python version infor
PY_MAJOR_VERSION = sys.version_info.major
PY_MINOR_VERSION = sys.version_info.minor

# Version-dependent imports
if PY_MAJOR_VERSION == 2:
    # Standard library modules
    from ConfigParser import SafeConfigParser

    # Hack for getting suffix of build/lib.* folder
    syssystem = platform.system().lower()
    sysmachine = platform.machine()
    sysplatform = "%s-%s" % (syssystem, sysmachine)
    # File extension for the binary extension modules
    if syssystem == "windows":
        # Alternate extension
        ext_suffix = ".pyd"
    else:
        # Normally it's a .so file
        ext_suffix = ".so"
else:
    # Standard library modules
    from configparser import SafeConfigParser

    # Suffix for build/lib.* folder
    sysplatform = sysconfig.get_platform()
    # Extension binary file extension
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")


# Config file
fcfg = "config%i.cfg" % PY_MAJOR_VERSION

# System configuration variables
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
config = SafeConfigParser()
# Read the configuration options
config.read(os.path.join(fmod, fcfg))

# Python command, in cases of potential ambiguity.
pythonexec = config.get("python", "exec")

# Compile
print("Building extensions...")
sp.call([pythonexec, "setup.py", "build"])

# Check for build
if not os.path.isdir(flib):
    print("Extension build FAILED:")
    print("  No build folder '%s' found" % flib)
    sys.exit(1)

# Creating wheel
print("Building wheel...")
sp.call([pythonexec, "setup.py", "bdist_wheel"])

# Status update
print("Moving the extensions into place...")
# Loop through extensions
for (ext, opts) in extopts.items():
    # Destination folder
    fdest = opts["destination"].replace("/", os.sep)
    # File name for compiled module
    fname = "%s%i%s" % (ext, int(PY_MAJOR_VERSION), ext_suffix)
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
    shutil.copy(fbld, fout)

# Remove that "egg" thing
print("Removing the egg-info folder")
shutil.rmtree("cape.egg-info")
