# -*- coding: utf-8 -*-
r"""
:mod:`cape.tnakit.pkgutils`: Tools for creating DataKit packages
==================================================================

This module provides a handle to :func:`setuptools.setup` with useful
defaults. The goal is for most datakit packages to just be able to call
:func:`setup` with no arguments.

"""

# Standard library
import os

# Third-party modules
import setuptools

# CAPE modules
from ..tnakit import rstutils
from ..tnakit.metautils import ModuleMetadata


# Setup wrapper
def setup(**kw):
    # Get options
    pkg = kw.pop("name", None)
    pkgs = kw.pop("packages", None)
    title = kw.pop("title", None)
    descr = kw.pop("description", None)
    version = kw.pop("version", "1.0")
    pkgdata = kw.pop("package_data", {})
    # Options not for setup()
    pkg_db = kw.pop("db", True)
    pkg_meta = kw.pop("meta", True)
    pkg_rawdata = kw.pop("rawdata", False)
    # Default list of packages
    if pkgs is None:
        pkgs = setuptools.find_packages()
    # Default package name
    if pkg is None:
        pkg = pkgs[0]
    # Read meta data
    meta = read_metadata(pkg)
    # Default title
    if title is None:
        title = meta.get("title", pkg)
    # Default description
    if descr is None:
        descr = title
    # Default package data
    if pkgdata is None:
        pkgdata = {}
    if pkgdata.get(pkg) is None:
        pkgdata[pkg] = find_package_data(
            pkg, meta=pkg_meta, db=pkg_db, rawdata=pkg_rawdata)
    # Call the main setup function
    setuptools.setup(
        name=pkg,
        packages=pkgs,
        package_data=pkgdata,
        description=descr,
        version=version,
        **kw)
    

# Find data files (package_data)
def find_package_data(pkg, meta=True, db=True, rawdata=False, **kw):
    r"""Find

    :Call:
        >>> pkg_files = find_package_data(pkg, **kw)
    :Inputs:
        *pkg*: :class:`str`
            Name of package
        *meta*: {``True``} | ``False``
            Flag to include ``meta.json``
        *db*: {``True``} | ``False``
            Flag to include processed ``db/`` files
        *rawdata*: ``True`` | {``False``}
            Flg to include source data in ``rawdata/`` folder
    :Outputs:
        *pkg_files*: :class:`list`\ [:class:`str`]
            List of files to include
    :Versions:
        * 2021-10-21 ``@ddalle``: Version 1.0
    """
    # Turn *pkg* into a path
    fpkg = pkg.replace(".", os.sep)
    # Initialize list of files
    pkg_files = set()
    # Enter folder to simplify relative paths
    os.chdir(fpkg)
    # Check for metadata
    if meta and os.path.isfile("meta.json"):
        pkg_files.add("meta.json")
    # Loop through db/ folder
    for dirpath, _, fnames in os.walk("db"):
        # Check flag
        if not db:
            break
        # Loop through files
        for fname in fnames:
            # Full path
            fabs = os.path.join(dirpath, fname)
            # Check pattern
            if fname.startswith("."):
                # Skip dotfiles
                continue
            elif fname.endswith(".dvc"):
                # DVC file: use w/o suffix
                pkg_files.add(fabs[:-4])
            else:
                # Some other file
                pkg_files.add(fabs)
    # Loop through rawdata/ folder
    for dirpath, _, fnames in os.walk("rawdata"):
        # Check flag
        if not rawdata:
            break
        # Loop through files
        for fname in fnames:
            # Full path
            fabs = os.path.join(dirpath, fname)
            # Check pattern
            if fname.startswith("."):
                # Skip dotfiles
                continue
            elif fname.endswith(".dvc"):
                # DVC file: use w/o suffix
                pkg_files.add(fabs[:-4])
            else:
                # Some other file
                pkg_files.add(fabs)
    # Go back up
    os.chdir("..")
    # Output
    return list(pkg_files)


# Read metadata
def read_metadata(pkg):
    r"""Read metadata for a package

    :Call:
        >>> meta = read_metadata(pkg)
    :Inputs:
        *pkg*: :class:`str`
            Name of package
    :Outputs:
        *meta*: :class:`ModuleMetadata`
            :class:`dict`-like container for ``meta.json``
    :Versions:
        * 2021-10-21 ``@ddalle``: Version 1.0
    """
    # Try to read file or use empty instance
    try:
        return ModuleMetadata(pkg)
    except Exception:
        return ModuleMetadata()
