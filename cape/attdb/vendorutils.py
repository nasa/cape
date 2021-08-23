#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================
:mod:`cape.attdb.vendorutils`: Package Vendorization Utilities
================================================================

This module provides some CAPE-specific alterations to the
:mod:`vendorize` package available from PyPI

"""

# Standard library modules
import json
import os

# Third-party modules
import setuptools
import vendorize


# Default *includes*
DEFAULT_FIND_WHERE = "."
DEFAULT_FIND_EXCLUDE = ()
DEFAULT_FIND_INCLUDE = ("*",)


# Find vendors
def find_vendors(where=".", **kw):
    # Options for find_packages()
    o_exclude = kw.pop("exclude", DEFAULT_FIND_EXCLUDE)
    o_include = kw.pop("include", DEFAULT_FIND_INCLUDE)
    # Replace any Nones
    if where is None:
        where = DEFAULT_FIND_WHERE
    if o_exclude is None:
        o_exclude = DEFAULT_FIND_EXCLUDE
    if o_include is None:
        o_include = DEFAULT_FIND_INCLUDE
    # Generate base path (absolute)
    absdir = os.path.realpath(where)
    # Find packages
    pkg_list = setuptools.find_packages(
        where, exclude=o_exclude, include=o_include)
    # Initialize list of packages with vendorize inputs
    pkgs = []
    # Loop through found packages
    for pkg in pkg_list:
        # Get folder name for package
        pkgdir = pkg.replace(".", os.sep)
        # Full path to vendorize input files
        fjson = os.path.join(pkgdir, "vendorize.json")
        ftoml = os.path.join(pkgdir, "vendorize.toml")
        # Check for either file
        if os.path.isfile(fjson):
            # Found JSON file
            pkgs.append(pkg)
        elif os.path.isfile(ftoml):
            # Found TOML file
            pkgs.append(pkg)
    # Sort the package list
    pkgs.sort()
    # Output
    return pkgs

    