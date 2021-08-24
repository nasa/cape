#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import json
import os
import sys

# Third-party
import setuptools

# Local modules
from .. import argread
from .. import text as textutils
from ..tnakit import promptutils


# Docstring for CLI
HELP_QUICKSTART = r"""
--------------------------------------------------------------
``dkit-quickstart``: Create template for a new datakit package
--------------------------------------------------------------

Create several files and folders to create a basic template for a
DataKit 3.0 package.

:Usage:
    .. code-block:: bash

        $ dkit-quickstart [PKG] [WHERE] [OPTIONS]

:Arguments:
    * *MODNAME1*: name of first module to process
    * *MODNAMEN*: name of *n*\ th module to process

:Options:

    -h, --help
        Display this help message and quit

    -f, --force
        Overwrite any existing database files (only for *MODNAMES*)

    -F, --force-all
        Overwrite all database files including added dependencies

    --no-reqs, --no-dependencies
        Don't read requirements; just process *MODNAMES*

    --prefix PREFIX
        Specify prefix which may be left off of *MODNAMES*

    --no-write
        Don't actually write databases (just print dependencies)

    --write_func FUNC
        Function name in modules to process datakits {"write_db"}

:Versions:

    * 2021-08-24 ``@ddalle``: Version 1.0
"""


# Defaults
DEFAULT_TITLE = "DataKit quickstart package"


# Main function
def main():
    r"""Main command-line interface function

    :Call:
        >>> main()
    :Versions:
        * 2021-08-24 ``@ddalle``: Version 1.0
    """
    # Process command-line arguments
    a, kw = argread.readkeys(sys.argv)
    # Real main function
    quickstart(*a, **kw)


# Primary API function
def quickstart(*a, **kw):
    # Check for help flag
    if kw.get('h') or kw.get('help'):
        # Display help message and quit
        print(textutils.markdown(HELP_QUICKSTART))
        return
    # Process positional args
    if len(a) > 2:
        print("Too many inputs!")
        print(textutils.markdown(HELP_QUICKSTART))
        return
    # Process "where"
    if len(a) > 1:
        # Get the "where" from *kw*, but default to second arg
        where = kw.pop("where", a[1])
    else:
        # Get "where", and default to current $PWD
        where = kw.pop("where", ".")
    # Process package name
    if len(a) > 0:
        # Get the package
        pkg = a[0]
    else:
        # Look for an existing package folder
        pkgs = setuptools.find_packages(where)
        # Get a default
        if len(pkgs) > 0:
            # Default to the first package found
            pkg = pkgs[0]
        else:
            # No default
            pkg = None
        # Prompt for a package
        pkg = promptutils.prompt("Python package name", vdef=pkg)
    # Path to default settings JSON file
    fjson = os.path.join(os.path.realpath(where), "datakit.json")
    # Load default settings
    if os.path.isfile(fjson):
        # Read options from it
        with open(fjson) as f:
            opts = json.load(f)
    else:
        # No defaults
        opts = {}
    # Apply default metadata
    for k, v in opts.get("meta", {}).keys():
        # Full CLI name
        col = "meta." + k
        # Save it
        kw[col] = v
    # Set default target
    kw.setdefault("target", opts.get("target"))
    # Create folder
    create_pkgdir(pkg, where, **kw)
    # Prompt for a title
    kw["title"] = _prompt_title(**kw)
    # Write metadata
    create_metadata(pkg, where, **kw)


# Create the metadata
def create_metadata(pkg, where=".", **kw):
    r"""Write ``meta.json`` template in package folder

    :Call:
        >>> create_metadata(pkg, **kw)
    :Inputs:
        *pkg*: :class:`str`
            Name of Python module/package to create
        *title*: {``None``} | :class:`str`
            Title to use for this package (not module name)
        *where*: {``"."``} | :class:`str`
            Path from which to begin
        *t*, *target*: {``None``} | :class:`str`
            Optional subdir of *where* to put package in
        *meta.key*: :class:`str`
            Save this value as *key* in ``meta.json``
    :Versions:
        * 2021-08-24 ``@ddalle``: Version 1.0
    """
    # Get the path to the package
    basepath = expand_target(where)
    pkgpath = get_pkgdir(pkg, **kw)
    # Absolute path
    pkgdir = os.path.join(basepath, pkgpath)
    # Path to metadata file
    fjson = os.path.join(pkgdir, "meta.json")
    # Check if file exists
    if os.path.isfile(fjson):
        return
    # Get title
    title = kw.get("title", DEFAULT_TITLE)
    # Initialize metadata
    metadata = {
        "title": title,
    }
    # Add anyother metadata
    for k in kw:
        # Check if it's a metadata key
        if k.startswith("meta."):
            # Normalized metadata key name
            col = k[5:].replace("_", "-")
            # Save value as shortened key
            metadata[col] = kw[k]
    # Write the JSON file
    with open(fjson, "w") as f:
        json.dump(metadata, f, indent=4)


# Ensure a title is present
def _prompt_title(**kw):
    # Get existing title from kwargs
    title = kw.get("title", kw.get("Title"))
    # If non-empty, return it
    if title:
        return title
    # Otherwise prompt one
    title = promptutils.prompt("One-line title for package")
    # Check for an answer again
    if title is None:
        return DEFAULT_TILE
    else:
        return title


# Ensure folder
def create_pkgdir(pkg, where=".", **kw):
    r"""Create folder(s) for a package

    :Call:
        >>> create_pkgdir(pkg, where=".", **kw)
    :Inputs:
        *pkg*: :class:`str`
            Name of Python module/package to create
        *where*: {``"."``} | :class:`str`
            Path from which to begin
        *t*, *target*: {``None``} | :class:`str`
            Optional subdir of *where* to put package in
    :Examples:
        This would create the folder ``c008/f3d/db001/`` if needed:
        
            >>> create_pkgdir("c008.f3d.db001")

        This would create the folder ``att_vm_clvtops3/db001":

            >>> create_pkgdir("db001", t="att_vm_clvtops3")

    :Versions:
        * 2021-08-24 ``@ddalle``: Version 1.0
    """
    # Get absolute path to target
    basepath = expand_target(where)
    # Get relative folder
    path = get_pkgdir(pkg, **kw)
    # Create folders as needed
    mkdirs(basepath, path)


# Get relative path to package folder
def get_pkgdir(pkg, **kw):
    r"""Get relative folder to a package

    :Call:
        >>> pkgdir = get_pkgdir(pkg, **kw)
    :Inputs:
        *pkg*: :class:`str`
            Name of Python module/package to create
        *t*, *target*: {``None``} | :class:`str`
            Optional subdir of *where* to put package in
    :Versions:
        * 2021-08-24 ``@ddalle``: Version 1.0
    """
    # Get target if any
    target = kw.get("t", kw.get("target"))
    # Convert package name to folder (mypkg.mymod -> mypkg/mymod/)
    pkgdir = pkg.replace(".", os.sep)
    # Total path
    if target:
        # Combine two-part package path
        return os.path.join(target.replace("/", os.sep), pkgdir)
    else:
        # No target; just the package
        return pkgdir



# Create folders
def mkdirs(basepath, path):
    r"""Create one or more folders within fixed *basepath*

    :Call:
        >>> mkdirs(basepath, path)
    :Inputs:
        *basepath*: :class:`str`
            Absolute folder in which to start creating folders
        *path*: :class:`str`
            Path to folder to create relative to *basepath*
    :Versions:
        * 2021-08-24 ``@ddalle``: Version 1.0
    """
    # Ensure *basepath* exists
    if not os.path.isdir(basepath):
        raise SystemError("basepath '%s' is not a folder" % basepath)
    # Loop through remaining folders
    for pathj in path.split(os.sep):
        # Append to path
        basepath = os.path.join(basepath, pathj)
        # Check if it exists
        if not os.path.isdir(basepath):
            # Create it
            os.mkdir(basepath)


# Expand "target"
def expand_target(target="."):
    r"""Expand a relative path

    :Call:
        >>> cwd = expand_target(target=".")
    :Inputs:
        *target*: {``"."``} | ``None`` | :class:`str`
            Relative or absolute path
    :Outputs:
        *cwd*: :class:`str`
            Absolute path
    :Versions:
        * 2021-08-24 ``@ddalle``: Version 1.0
    """
    # Get absolute path
    if target is None:
        # Use current path
        return os.getcwd()
    else:
        # Expand
        return os.path.realpath(target)


# Write the starter for a module
def write_init_py(pkg, target="."):
    pass