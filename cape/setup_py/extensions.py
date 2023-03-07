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
    mod = importlib.import_module("ConfigParser")
    # Get parser class
    ConfigParser = mod.SafeConfigParser
else:
    # Config parser module
    mod = importlib.import_module("configparser")
    # Get parser class
    ConfigParser = mod.ConfigParser


# Config file
_CONFIG_FILE = "config%i.cfg" % PY_MAJOR_VERSION
_EXTENSION_FILE = "extensions.json"

# This folder
THIS_DIR = os.path.dirname(__file__)
CAPE_DIR = os.path.dirname(THIS_DIR)
REPO_DIR = os.path.dirname(CAPE_DIR)

# Absolute paths
CONFIG_FILE = os.path.join(THIS_DIR, _CONFIG_FILE)
EXTENSION_FILE = os.path.join(THIS_DIR, _EXTENSION_FILE)

# Get a get/set type object
CONFIG = ConfigParser()
# Read the configuration options
CONFIG.read(CONFIG_FILE)

# C compiler flags
cflags = CONFIG.get("compiler", "extra_cflags").split()

# Linker options
ldflags = CONFIG.get("compiler", "extra_ldflags").split()

# Extra include directories
include_dirs = CONFIG.get("compiler", "extra_include_dirs").split()

# Read extension settings
with open(EXTENSION_FILE, 'r') as fp:
    EXTENSION_OPTS = json.load(fp)

# Initialize extensions
EXTENSIONS = []
# Loop through specified extensions
for (ext, opts) in EXTENSION_OPTS.items():
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

