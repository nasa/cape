#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
:mod:`cape.attdb.cli`: Command-Line Interface to datakit tools
=================================================================


"""

# Standard library modules
import importlib
import os
import setuptools
import sys

# Local modules
from . import datakitloader
from . import pkgutils
from .. import argread
from .. import text as textutils


# Docstring for CLI
HELP_WRITEDB = r"""
-----------------------------------------------------------------
``dkit-writedb``: Read raw data to create formatted datakit files
-----------------------------------------------------------------

Any database can be processed by this script when given its module name.
The revision argument can be something like **c008.f3d.db001** or a full
module name like **sls10afa.c008.f3d.db001**.

Folder names, e.g. ``sls10afa/c008/f3d/db001``, are also allowed.

:Usage:
    .. code-block:: bash

        $ dkit-writedb [MODNAMES] [OPTIONS]

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

    * 2017-07-13 ``@ddalle``: Version 1.0
    * 2020-07-06 ``@ddalle``: Version 1.1; update docstring
    * 2021-07-17 ``@ddalle``: Version 2.0; process dependencies
    * 2021-07-19 ``@ddalle``: Version 2.1; add ``--no-write``
    * 2021-08-20 ``@ddalle``: Version 3.0; generalize for ``cape``
    * 2021-09-15 ``@ddalle``: Version 3.1; more DVC support
"""


# Main writer
def main():
    r"""Main command-line interface function

    :Call:
        >>> main()
    :Versions:
        * 2021-07-15 ``@ddalle``: Version 1.0
        * 2021-07-17 ``@ddalle``: Version 2.0
            - Move to :func:`write_dbs`
            - Add dependency tracking
            - Add ``-F`` option

        * 2021-08-20 ``@ddalle``: Version 3.0
            - Generalize for :mod:`cape`
            - Add *write_func* option
    """
    # Process command-line arguments
    a, kw = argread.readkeys(sys.argv)
    # Real main function
    write_dbs(*a, **kw)


# API to module writer
def write_dbs(*a, **kw):
    r"""Write one or more datakit modules, with dependencies

    :Call:
        >>> write_dbs(*modnames, **kw)
    :Inputs:
        *modnames*: :class:`tuple`\ [:class:`str`]
            Names of modules to write, like ``"db0001"``
        *prefix*: {``None``} | :class:`str`
            Optional user-specified prefix
        *f*, *force*: ``True`` | {``False``}
            Overwrite any modules in *a*
        *F*, *force_all*, *force-all*: ``True`` | {``False``}
            Overwrite all modules, including dependencies
        *reqs*, *dependencies*: {``True``} | ``False``
            Also process any modules listed in *REQUIREMENTS* in each
            modul in *a*
        *write*: {``True``} | ``False``
            Flag to write databases (otherwise just print dependencies)
        *write_func*, *func*: {``"write_db"``} | :class:`str`
            Name of function to use to write formatted files
    :Versions:
        * 2021-07-17 ``@ddalle``: Version 1.0
        * 2021-07-19 ``@ddalle``: Version 1.1; add *write* option
        * 2021-08-20 ``@ddalle``: Version 1.2; generalize *prefix*
    """
    # Check for help flag
    if (len(a) == 0) or kw.get('h') or kw.get('help'):
        # Display help message and quit
        print(textutils.markdown(HELP_WRITEDB))
        return
    # Change '/' to '.'
    a_normalized = tuple(ai.replace(os.sep, '.') for ai in a)
    # Initialize packages
    pkgs = []
    # Check inputs against list
    for j, aj in enumerate(a_normalized):
        # Find matching packages
        pkgsj = pkgutils.find_packages(regex=aj)
        # Check for errors
        if len(pkgsj) == 0:
            print("Found no packages for name %i, '%s'" % (j+1, aj))
            continue
        # Avoid duplicates
        for pkg in pkgsj:
            if pkg in pkgs:
                continue
            # Add to global list
            pkgs.append(pkg)
    # Process other options
    force_all = kw.pop("force-all", kw.pop("force_all", kw.pop("F", False)))
    force_last = kw.pop("force", kw.pop("f", False))
    write = kw.pop("write", True)
    # Process original module names
    anames, _ = genr8_modsequence(pkgs, reqs=False)
    # Process all other requirements
    dbnames, modnames = genr8_modsequence(pkgs, **kw)
    # Status update
    if force_all:
        print("Rewriting databases from source:")
    elif force_last:
        print("(Re)writing databases from source:")
    else:
        print("Writing databases from source (no overwrite):")
    # Loop through modules in order
    for dbname, modname in zip(dbnames, modnames):
        # Get overwrite option
        if force_all:
            # Overwrite all modules
            f = True
        elif force_last and dbname in anames:
            # Overwrite specified modules
            f = True
        else:
            # Don't overwrite
            # 1. User specified no -f or -F option
            # 2. User specified -f but this is an extra mod from reqs
            f = False
        # Print module and name being processed
        if f:
            print("%s (%s) overwrite=True" % (dbname, modname))
        else:
            print("%s (%s)" % (dbname, modname))
        # Check for simulate option
        if not write:
            continue
        # Individual revision
        write_db(modname, f=f, **kw)


# Write the DB file(s) for one or more DBs
def write_db(modname, **kw):
    r"""Convert source data to formatted datakit files

    :Call:
        >>> write_db(modname, **kw)
    :Inputs:
        *modnames*: :class:`tuple`\ [:class:`str`]
            Names of module names to process
        *prefix*: {``None``} | :class:`str`
            Optional user-specified prefix
        *f*: ``True`` | {``False``}
            Overwrite existing data files
        *write_func*, *func*: {``"write_db"``} | :class:`str`
            Name of function to use to write formatted files
    :Versions:
        * 2017-07-13 ``@ddalle``: Version 1.0
        * 2018-12-27 ``@ddalle``: Version 2.0; using :mod:`importlib`
        * 2021-07-15 ``@ddalle``: Version 2.1; Generalize for TNA/S-53
        * 2021-08-20 ``@ddalle``: Version 3.0
            - move to :mod:`cape` from ``ATT-VM-CLVTOPS-003``
            - generalize prefix using :func:`setuptools.find_packages`
            - add *prefix*, *write_func* kwargs
    """
    # Get prefix
    prefix = kw.pop("prefix", None)
    # Get non-default function name
    func = kw.pop("write_func", kw.pop("func", None))
    # Remove __replaced__
    kw.pop("__replaced__", None)
    # Read module
    mod = import_module(modname, prefix)
    # Call the template function from
    if func is None:
        # Use default function
        mod.write_db(**kw)
    else:
        # Get the function attribute
        fn = getattr(mod, func)
        # Call it
        fn(**kw)


# Read each module and process *REQUIREMENTS*
def genr8_modsequence(modnames, **kw):
    r"""Create a sequence of modules that satisfy all requirements

    :Call:
        >>> dbnames, modnames = genr8_modsequence(modnames, **kw)
    :Inputs:
        *modnames*: :class:`tuple`\ [:class:`str`]
            Names of module names to process
        *reqs*, *dependencies*: {``True``} | ``False``
            Also process any modules listed in *REQUIREMENTS* in each
            module in *a*
        *prefix*: {``None``} | :class:`str`
            Optional user-specified prefix
    :Outputs:
        *dbnames*: :class:`list`\ [:class:`str`]
            List of *DB_NAME* for modules corresponding to *modnames*
        *modnames*: :class:`list`\ [:class:`str`]
            Modules in order that satisfies all dependencies
    :Versions:
        * 2021-07-17 ``@ddalle``: Version 1.0
        * 2021-08-20 ``@ddalle``: Version 1.1; *prefix* option
    """
    # Check for --no-dependencies
    qreq = kw.get("requirements", kw.get("dependencies", kw.get("reqs", True)))
    # Prefix
    prefix = kw.get("prefix")
    # Process case where now dependencies will be processed
    if not qreq:
        # Just create a list of names in order given
        dbnames = []
        # Loop through names
        for modname in modnames:
            # Import module
            mod = import_module(modname, prefix)
            # Get name
            dbname = get_dbname(mod)
            # Append
            dbnames.append(dbname)
        # Output
        return dbnames, list(modnames)
    # Make list of unchecked modules
    unprocessed_modnames = list(modnames)
    # Save original inputs
    orig_modnames = modnames
    # Initialize final module load sequence
    modnames = []
    dbnames = []
    # Status update
    print("Processing requirements:")
    # Loop through specified modules
    while len(unprocessed_modnames) > 0:
        # Take first module name
        modname = unprocessed_modnames.pop(0)
        # Read the module
        mod = import_module(modname, prefix)
        # Get name
        dbname = get_dbname(mod)
        # Status update for original requests (even if it has no reqs)
        if modname in orig_modnames:
            print("  %s (%s)" % (dbname, modname))
        # Append to sequence if not already present
        if dbname not in dbnames:
            modnames.append(modname)
            dbnames.append(dbname)
        # Get index
        i = dbnames.index(dbname)
        # Get requirements
        reqdbnames = mod.__dict__.get("REQUIREMENTS")
        # Check if no requirements
        if not reqdbnames:
            continue
        # Status update for implied modules (only if *they* have reqs)
        if modname not in orig_modnames:
            print("  %s (%s)" % (dbname, modname))
        # Loop through them (in reverse order)
        for reqdbname in reversed(reqdbnames):
            # Read module
            reqmod = import_dbname(mod, reqdbname)
            # Get module name
            reqmodname = reqmod.__name__
            # Check if it's in the sequence
            if reqdbname in dbnames:
                # Get index to make sure it's before *i*
                j = dbnames.index(reqdbname)
                # Check if later than *i*
                if j > i:
                    print("    Detected possible circular requirements")
                # Nothing to add
                continue
            # Status update
            print("    > %s (%s)" % (reqdbname, reqmodname))
            # Insert it
            modnames.insert(i, reqmodname)
            dbnames.insert(i, reqdbname)
            # Otherwise call this dependency "unprocessed"
            unprocessed_modnames.insert(0, reqmodname)
    # Output
    return dbnames, modnames


# Import a child
def import_dbname(mod, dbname, **kw):
    r"""Import a module by *DB_NAME* instead of module spec

    :Call:
        >>> mod2 = import_dbname(mod, dbname)
    :Inputs:
        *mod*: :class:`module`
            Parent module
        *dbname*: :class:`str`
            DB name like ``"SLS-10-D-AFA-004"``
        *prefix*: {``None``} | :class:`str`
            Optional user-specified prefix
    :Outputs:
        *mod2*: :class:`module`
            Second module, having DB name matching *dbname*
    :Versions:
        * 2021-07-16 ``@ddalle``: Version 1.0
        * 2021-08-20 ``@ddalle``: Version 1.1; *prefix* option
    """
    # Get DataKitLoader
    dkl = mod.__dict__.get("DATAKIT_LOADER")
    # Check for valid loader
    if isinstance(dkl, datakitloader.DataKitLoader):
        # Import child
        return dkl.import_db_name(dbname)
    else:
        # Try to use the name directly
        return import_module(dbname, prefix=kw.get("prefix"))


# Read a single module
def import_module(modname=None, prefix=None, **kw):
    r"""Import module from (possibly abbrev.) name

    :Call:
        >>> mod = import_module(modname=None)
    :Inputs:
        *modname*: ``None`` | :class:`str`
            Module name, possibly missing top-level prefix
        *prefix*: {``None``} | :class:`str`
            Optional user-specified prefix
    :Outputs:
        *mod*: :class:`module`
            Module with (possibly prefixed) name *rev*
    :Versions:
        * 2021-07-16 ``@ddalle``: Version 1.0 (``ATT-VM-CLVTOPS-003``)
        * 2021-08-20 ``@ddalle``: Version 1.1
            - automated default *PREFIX*
            - support empty *modname*
    """
    # Append current path
    sys.path.insert(0, os.path.realpath("."))
    # Prepend module name if necessary
    modname = get_fullmodname(modname, prefix)
    # Attempt import
    try:
        # Use import library
        mod = importlib.import_module(modname)
        # Clean up path
        sys.path.pop(0)
        # Output
        return mod
    except ImportError:
        # Clean up path
        sys.path.pop(0)
        # Unknown version
        raise ValueError("Failed to import '%s'" % modname)


# Prefix module name
def get_fullmodname(modname, prefix=None, **kw):
    r"""Append prefix to module name if necessary

    For example ``"v004"`` might become ``"sls10afa.v004"``

    :Call:
        >>> modname = get_fullmodname(modname, prefix=None)
    :Inputs:
        *modname*: :class:`str`
            Module name, possibly missing top-level prefix
        *prefix*: {``None``} | :class:`str`
            Optional user-specified prefix
    :Outputs:
        *modname*: :class:`str`
            Full module name for import, possibly prepended
    :Versions:
        * 2021-08-20 ``@ddalle``: Version 1.0
        * 2021-09-15 ``@ddalle``: Version 1.1; better *prefix* check
    """
    # Get prefix
    prefix = get_prefix(prefix=prefix)
    # Check if name starts with the prefix
    if not modname:
        # Use the prefix as the module name
        modname = prefix
    elif os.path.isdir(modname.replace(".", os.sep)):
        # Valid module name even if has different prefix
        modname = modname
    elif modname.split(".")[0] != prefix:
        # Prepend
        modname = prefix + "." + modname
    # Output
    return modname


# Get prefix
def get_prefix(prefix=None, **kw):
    r"""Determine module name prefix based on current folder

    :Call:
        >>> prefix = get_prefix(prefix=None, **kw)
    :Inputs:
        *prefix*: {``None``} | :class:`str`
            Optional user-specified prefix
    :Outputs:
        *prefix*: :class:`str`
            User-specified prefix or package from
            :func:`setuptools.find_packages`
    :Versions:
        * 2021-08-20 ``@ddalle``: Version 1.0
    """
    # Check for input
    if prefix is not None:
        return prefix
    # Find packages
    pkgs = setuptools.find_packages()
    # Return the first one
    return pkgs[0]


# Create a name for a module
def get_dbname(mod):
    r"""Get database name from a module

    :Call:
        >>> dbname = get_dbname(mod)
    :Inputs:
        *mod*: :class:`module`
            DataKit module
    :Outputs:
        *dbname*: :class:`str`
            Database name, from *mod.DATAKIT_LOADER* or *mod.__name__*
    :Versions:
        * 2021-08-20 ``@ddalle``: Version 1.0
    """
    # Get DataKitLoader
    dkl = mod.__dict__.get("DATAKIT_LOADER")
    # Get name if possible
    if isinstance(dkl, datakitloader.DataKitLoader):
        # Get name like "ATT-VM-CLVTOPS-003-0101"
        return dkl.get_option("DB_NAME")
    else:
        # Use full module name like "att_vm_clvtops3.db0101.datakit"
        return mod.__name__

