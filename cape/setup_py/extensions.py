#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import importlib
import json
import os
import sys

# Third-party
from setuptools import Extension


# Python version infor
PY_MAJOR_VERSION = sys.version_info.major
PY_MINOR_VERSION = sys.version_info.minor

# Version-dependent imports
if PY_MAJOR_VERSION == 2:
    # Config parser module
    mod = importlib.import_module("SafeConfigParser")
    # Get parser class
    ConfigParser = mod.SafeConfigParser
else:
    # Config parser module
    mod = importlib.import_module("configparser")
    # Get parser class
    ConfigParser = mod.ConfigParser


# Config file
CONFIG_FILE = "config%i.cfg" % PY_MAJOR_VERSION
EXTENSION_FILE = "extensions.json"

# This folder
THIS_DIR = os.path.dirname(__file__)
CAPE_DIR = os.path.dirname(THIS_DIR)
REPO_DIR = os.path.dirname(CAPE_DIR)

# Path to this file
fdir = os.path.dirname(os.path.realpath(__file__))
fcape = os.path.join(fdir, "cape")

# Get a get/set type object
config = ConfigParser()
# Read the configuration options
config.read(os.path.join(THIS_DIR, CONFIG_FILE))

# C compiler flags
cflags = config.get("compiler", "extra_cflags").split()

# Linker options
ldflags = config.get("compiler", "extra_ldflags").split()

# Extra include directories
include_dirs = config.get("compiler", "extra_include_dirs").split()

# Extensions JSON file
extjson = os.path.join(THIS_DIR, EXTENSION_FILE)
# Read extension settings
extopts = json.load(open(extjson))

# Initialize extensions
EXTENSIONS = []
# Loop through specified extensions
for (ext, opts) in extopts.items():
    # Get sources
    extsources = [str(src) for src in opts["sources"]]
    # Create extension
    _ext = Extension(
        str(ext) + str(PY_MAJOR_VERSION),
        include_dirs=include_dirs,
        extra_compile_args=cflags,
        extra_link_args=ldflags,
        sources=extsources)
    # Add to list
    EXTENSIONS.append(_ext)

