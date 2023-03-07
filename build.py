#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library imports
import os
import sys
import shutil
import subprocess as sp

# Local imports
from cape.setup_py.extensions import (
    EXT_SUFFIX,
    EXTENSION_OPTS,
    LIB_DIR,
    PY_MAJOR_VERSION)

# Path to this folder
THIS_DIR = os.path.dirname(__file__)

# Compile
print("Building extensions...")
sp.call([sys.executable, "setup_with_extension.py", "build"])

# Check for build
if not os.path.isdir(LIB_DIR):
    print("Extension build FAILED:")
    print("  No build folder '%s' found" % LIB_DIR)
    sys.exit(1)

# Creating wheel
print("Building wheel...")
ierr = sp.call([sys.executable, "setup_with_extension.py", "bdist_wheel"])

# Status update
print("Moving the extensions into place...")
# Loop through extensions
for (ext, opts) in EXTENSION_OPTS.items():
    # File name for compiled module
    fname = "%s%i%s" % (ext, PY_MAJOR_VERSION, EXT_SUFFIX)
    # Final location for module
    fout = os.path.join(THIS_DIR, fname)
    # Expected build location
    fbld = os.path.join(THIS_DIR, LIB_DIR, fname)
    # Exit if no build
    if not os.path.isfile(fbld):
        print("Build of extension '%s' failed" % ext)
        sys.exit(1)
    # Check for existing object
    if os.path.isfile(fout):
        os.remove(fout)
    # Status update
    print(
        "copying file '%s' -> '%s'" % (
            os.path.relpath(fbld, THIS_DIR),
            os.path.relpath(fout, THIS_DIR)))
    # Move the result to the destination folder
    shutil.copy(fbld, fout)

# Remove that "egg" thing
print("Removing the egg-info folder")
shutil.rmtree("cape.egg-info")
