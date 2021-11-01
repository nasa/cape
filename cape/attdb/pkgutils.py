# -*- coding: utf-8 -*-
r"""
:mod:`cape.tnakit.pkgutils`: Tools for creating DataKit packages
==================================================================

This module provides a handle to :func:`setuptools.setup` with useful
defaults. The goal is for most datakit packages to just be able to call
:func:`setup` with no arguments.

"""

# Standard library
import ast
import os

# Third-party modules
import setuptools

# CAPE modules
from ..tnakit import rstutils
from ..tnakit.metautils import ModuleMetadata


# Default values for various options
DEFAULT_FIND_EXCLUDE = ()
DEFAULT_FIND_INCLUDE = ("*",)
DEFAULT_FIND_REGEX = None
DEFAULT_FIND_WHERE = "."

# Number of criteria required in find_packages()
N_CRITERIA_FIND_PKG = 2


# Setup wrapper
def setup(**kw):
    r"""Create a package using :mod:`setuptools` and some extra defaults

    :Call:
        >>> setup(**kw)
    :Inputs:
        *name*: {``None``} | :class:`str`
            Name of the package (default from :func:`find_packages`)
        *packages*: {``None``} | :class:`list`\ [:class:`str`]
            Packages to include (default from :func:`find_packages`)
        *title*: {``None``} | :class:`str`
            Short description (default from ``meta.json``)
        *description*: {*title*} | :class:`str`
            Description
        *package_data*: {``{}``} | :class:`dict`
            Extra files to include for *name*
        *db*: {``True``} | ``False``
            Search for ``db/`` data files in *package_data*
        *meta*: {``True``} | ``False``
            Automatically include ``meta.json`` in *package_data*
        *rawdata*: ``True`` | {``False``}
            Search for ``rawdata/`` files in *package_data*
    :Versions:
        * 2021-10-21 ``@ddalle``: Version 1.0
    """
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


# Find datakit packages
def find_packages(where=".", **kw):
    r"""Find packages that appear to be datakit packages

    In order for a package to qualify as a datakit package, it must
    satisfy 2 of the following 4 criteria:

    1. The package contains the file ``meta.json``
    2. The ``__init__.py`` has a function :func:`read_db`
    3. The ``__init__.py`` has a function :func:`write_db`
    4. The ``__init__.py`` imports :mod:`datakitloader`

    (A package created by ``dkit-quickstart`` will satisfy all of
    these criteria.)

    :Call:
        >>> pkgs = find_packages(where=".", **kw)
    :Inputs:
        *where*: {``"."``} | :class:`str`
            Location from which to search for packages
        *exclude*: {``()``} | :class:`tuple`\ [:class:`str`]
            List of globs to exclude from package search
        *include*: {``("*",)``} | :class:`tuple`\ [:class:`str`]
            List of globs to include during package search
        *re*, *regex*, {``None``} | :class:`str`
            Only include packages including regular expression *regex*
    :Outputs:
        *pkgs*: :class:`list`\ [:class:`str`]
            List of packages that look like datakit packages
    :Versions:
        * 2021-08-23 ``@ddalle``: Version 1.0
    """
    # Options for find_packages()
    o_exclude = kw.pop("exclude", DEFAULT_FIND_EXCLUDE)
    o_include = kw.pop("include", DEFAULT_FIND_INCLUDE)
    # Other options
    o_regex = kw.pop("re", kw.pop("regex", DEFAULT_FIND_REGEX))
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
    # Also look in the current folder
    pkg_list.insert(0, "")
    # Initialize list of packages with vendorize inputs
    pkgs = []
    # Loop through found packages
    for pkg in pkg_list:
        # Check for regex
        if o_regex:
            # Check for regular expression match
            if re.search(o_regex, pkg) is None:
                continue
        # Number of criteria met
        n = 0
        # Get folder name for package
        pkgdir = pkg.replace(".", os.sep)
        # Absolutize
        pkgdir = os.path.join(absdir, pkgdir)
        # File names
        fmeta = os.path.join(pkgdir, "meta.json")
        finit = os.path.join(pkgdir, "__init__.py")
        # Check for source file
        if not os.path.isfile(finit):
            # (This should not happen)
            continue
        # Check for package metadata file
        if os.path.isfile(fmeta):
            n += 1
        # Parse the source code file
        try:
            modtree = ast.parse(open(finit).read())
        except SyntaxError:
            # Package is broken
            continue
        # Check for datakitloader
        if _check_datakitloader(modtree):
            n += 1
            # Check criteria
            if n >= N_CRITERIA_FIND_PKG:
                # Found TOML file
                pkgs.append(pkg)
                continue
        # Check for read_db()
        if _check_read_db(modtree):
            n += 1
            # Check criteria
            if n >= N_CRITERIA_FIND_PKG:
                # Found TOML file
                pkgs.append(pkg)
                continue
        # Check for write_db()
        if _check_write_db(modtree):
            n += 1
            # Check criteria
            if n >= N_CRITERIA_FIND_PKG:
                # Found TOML file
                pkgs.append(pkg)
                continue
    # Sort the package list
    pkgs.sort()
    # Output
    return pkgs


# Find data files (package_data)
def find_package_data(pkg, meta=True, db=True, rawdata=False, **kw):
    r"""Find extra data files for a datakit package

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


# Check for datakitloader import in module's syntax tree
def _check_datakitloader(modtree):
    r"""Check for :mod:`datakitloader` import in module

    :Call:
        >>> q = _check_datakitloader(modtree)
    :Inputs:
        *modtree*: :class:`ast.Module`
            Syntax tree of a parsed Python module
    :Outputs:
        *q*: ``True`` | ``False``
            Whether :mod:`datakitloader` is imported
    :Versions:
        * 2021-11-01 ``@ddalle``: Version 1.0
    """
    # Check type
    if not isinstance(modtree, ast.Module):
        return False
    # Loop through elements
    for elem in modtree.body:
        # Check type
        if not isinstance(elem, (ast.Import, ast.ImportFrom)):
            continue
        # Get the package name
        name = elem.names[0].name
        # Success if it ends with "datakitloader"
        if name.endswith("datakitloader"):
            return True
    # No such import found
    return False


# Check for read_db() module's syntax tree
def _check_read_db(modtree):
    r"""Check if module contains :func:`read_db`

    :Call:
        >>> q = _check_read_db(modtree)
    :Inputs:
        *modtree*: :class:`ast.Module`
            Syntax tree of a parsed Python module
    :Outputs:
        *q*: ``True`` | ``False``
            Whether module defines :func:`read_db`
    :Versions:
        * 2021-11-01 ``@ddalle``: Version 1.0
    """
    # Check type
    if not isinstance(modtree, ast.Module):
        return False
    # Loop through elements
    for elem in modtree.body:
        # Check type
        if not isinstance(elem, ast.FunctionDef):
            continue
        # Check the function name
        if elem.name == "read_db":
            return True
    # No such import found
    return False


# Check for write_db() in module's syntax tree
def _check_write_db(modtree):
    r"""Check if module contains :func:`write_db`

    :Call:
        >>> q = _check_write_db(modtree)
    :Inputs:
        *modtree*: :class:`ast.Module`
            Syntax tree of a parsed Python module
    :Outputs:
        *q*: ``True`` | ``False``
            Whether module defines :func:`write_db`
    :Versions:
        * 2021-11-01 ``@ddalle``: Version 1.0
    """
    # Check type
    if not isinstance(modtree, ast.Module):
        return False
    # Loop through elements
    for elem in modtree.body:
        # Check type
        if not isinstance(elem, ast.FunctionDef):
            continue
        # Check the function name
        if elem.name == "write_db":
            return True
    # No such import found
    return False

