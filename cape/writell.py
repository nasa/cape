#!/usr/bin/env python3
r"""
:mod:`cape.writell`: Collect CAPE line load databooks
======================================================

This module provides tools to combine the individual-case line load
files from a CAPE "databook" (which are extracted using a command like
``pyfun --ll``) into a single ``.mat`` file.

:Versions:
    * 2021-02-10 ``@ddalle``: Version 1.0 (executable)
    * 2021-09-30 ``@ddalle``: Version 1.0 (module)
"""

# Standard library
import fnmatch
import os
import sys

# Third-party modules
import numpy as np

# CAPE modules
from . import argread
from . import text as textutils
from .attdb import rdb as rdb
from .attdb import dbll as dbll
from .cntl import Cntl


# Help message for executable
HELP_WRITELL = r"""
``cape-writell``: Combine CAPE line load data into MAT file
=====================================================================

This tool combines the individual-case line load data files from a CAPE
"databook" (which are extracted using a command like ``pyfun --ll``)
into a single ``.mat`` file.

That ``.mat`` file (or files, if multiple line load components are
written) is placed in the databook folder.

:Usage:
    .. code-block:: console

        $ cape-writell [PAT1 PAT2 ...] [OPTIONS]

:Example:

    Write all the DataBook components in ``pyFun.json`` with the type
    ``"LineLoad"``.

        .. code-block:: console

            $ cape-writell -f pyFun.json

    Only write the line load component ``"CORE_LL"``.

        .. code-block:: console

            $ cape-writell CORE_LL

    Write any LineLoad component starting with "CORE" or "BODY".

        .. code-block:: console

            $ cape-writell "CORE*" "BODY*"

:Inputs:
    *PAT1*: Optional pattern to subset available line load components
    *PAT2*: Second pattern, use comps matching *PAT1* or *PAT2*

    .. note::

        If no subsets are given, all available line load components will
        be processed.

:Outputs:

    -h, --help
        Display this help message and quit

    -f FNAME
        Use CAPE input file *FNAME* (by default checks for
        ``'cape.json'``, ``'pyFun.json'``, etc.)

    --ll COMP
        Only consider components matching *COMP* from "DataBook" section
        of *FNAME* (by default apply no filters) (combines with *PAT1*,
        *PAT2*, etc.)

:Versions:
    * 2021-02-10 ``@ddalle``: Version 1.0
"""

# Standard file names for JSON input files
_JSON_FNAMES = [
    "cape.json",
    "pyFun.json",
    "pyfun.json",
    "pyCart.json",
    "pycart.json",
    "pyOver.json",
    "pyover.json"
]


# Write the datakit
def write_ll_datakit(cntl, comp):
    r"""Write ``.mat`` file of combined line loads

    The output file is in the ``"DataBook"`` folder of *cntl* with a name
    like ``lineload/lineload_%(comp)s.mat``.

    :Call:
        >>> db = genr8_ll_datakit(cntl, comp)
    :Inputs:
        *cntl*: :class:`cape.Cntl`
            CAPE control class instance
        *comp*: :class:`str`
            Name of ``"LineLoad"`` component to read using *cntl*
    :Outputs:
        *db*: ``None`` | :class:`cape.attdb.dbll.DBLL`
            Line load database read from ``pyfun --ll`` data if possible
    :Versions:
        * 2021-02-10 ``@ddalle``: Version 1.0
    """
    # Generate the datakit
    db = genr8_ll_datakit(cntl, comp)
    # Check for result
    if db is None:
        return
    # DataBook dir
    fdb = cntl.opts.get_DataBookDir()
    # Name of output file
    fout = os.path.join("lineload", "lineload_%s.mat" % comp)
    # Absolute path
    fmat = os.path.join(fdb, fout)
    # Status update
    print("Writing '%s' combined line load data kit" % comp)
    print("  File:")
    print("    %s" % fout)
    print("  Folder:")
    print("    %s" % fdb)
    # Write it
    db.write_mat(fmat)


# Function to create datakit from line loads
def genr8_ll_datakit(cntl, comp):
    r"""Create datakit from run matrix of CAPE line load files

    :Call:
        >>> db = genr8_ll_datakit(cntl, comp)
    :Inputs:
        *cntl*: :class:`cape.Cntl`
            CAPE control class instance
        *comp*: :class:`str`
            Name of ``"LineLoad"`` component to read using *cntl*
    :Outputs:
        *db*: ``None`` | :class:`cape.attdb.dbll.DBLL`
            Line load database read from ``pyfun --ll`` data if possible
    :Versions:
        * 2021-02-10 ``@ddalle``: Version 1.0
    """
    # Status update
    print("Reading databook component '%s'" % comp)
    # Databook folder
    fdb = cntl.opts.get_DataBookDir()
    # Check it
    if not os.path.isdir(fdb):
        print("  No databook folder found:")
        print("    %s" % fdb)
        return
    # Run matrix interface
    x = cntl.x
    # Initialize datakit with whole run matrix
    db = dbll.DBLL(Values=x)
    # Indices of matches
    matches = []
    # Get entire list of (candidate) runs
    fruns = x.GetFullFolderNames()
    # Number of candidates
    n = len(fruns)
    # Maximum length
    lmax_name = max([len(frun) for frun in fruns])
    lmax_case = int(np.log10(n-1)) + 1
    # STDOUT format
    fmt = "index=%%%ii n=%%%ii case=%%-%is" % (lmax_case, lmax_case, lmax_name)
    # Name for line load datat files in databook case folders
    fllcsv = "LineLoad_%s.csv" % comp
    # Seam files
    fsmy = "LineLoad_%s.smy" % comp
    fsmz = "LineLoad_%s.smz" % comp
    # Absolute seam files
    fsmy = os.path.join(fdb, "lineload", fsmy)
    fsmz = os.path.join(fdb, "lineload", fsmz)
    # Current index of matched case
    j = -1
    # Loop trhough cases
    for i, frun in enumerate(fruns):
        # Folder for those line loads
        frundb = os.path.join(fdb, "lineload", frun)
        # Check if it exists
        if not os.path.isdir(frundb):
            continue
        # Path to file
        flli = os.path.join(frundb, fllcsv)
        # Check if file exists
        if not os.path.isfile(flli):
            continue
        # Try to read it
        try:
            # Found a case
            dbi = rdb.DataKit(csv=flli)
        except Exception:
            # Couldn't read the case
            continue
        # Update matches
        j += 1
        matches.append(i)
        # Status update regarding match
        print(fmt % (i, j, frun))
        # Get column names for *db*
        if j == 0:
            # Get combined-datakit column names
            dbcols = genr8_dbcolnames(dbi)
            # Get line load cols from *dbi*
            llcols = genr8_llcolnames(dbi)
        # Save columns
        for k, col in enumerate(dbi.cols):
            # Translated column name
            dbcol = dbcols.get(col, col)
            # Initialize line load cols in *db* for first match
            if j == 0:
                # Get column size from *dbi*
                # (These should all be equal.)
                nk = dbi[col].size
                # Add a 2D array with maximum size
                db.save_col(dbcol, np.zeros((nk, n)))
                # Save columns
            # Save the data
            db[dbcol][:, i] = dbi[col]
    # Check for no matches
    if j < 0:
        return
    # Convert matches to array
    I = np.array(matches)
    # Trim the *xcols* to only those with matches
    for col in cntl.x.cols:
        # Get values and remove definition
        v = db.burst_col(col)
        # Check type
        if db.get_col_type(col) == "str":
            # Reprocess as new list
            vo = [v[j] for j in matches]
        else:
            # Direct subset
            vo = v[I]
        # Resave column
        db.save_col(col, vo)
    # Trim the line load columns
    for col in dbi.cols:
        # Line load col name
        dbcol = dbcols.get(col, col)
        # Get data from *db*
        v = db.burst_col(dbcol)
        # Check for 1D
        if v.ndim == 1:
            # Just save *matches*
            v = v[I]
        elif v.ndim == 2:
            # Get min/max of each *row*
            vmin = np.min(v, axis=1)
            vmax = np.max(v, axis=1)
            # Delta across rows
            vdiff = np.max(vmax - vmin)
            # Scaling
            vabs = max(0.01,
                max(np.max(np.abs(vmin)), np.max(np.abs(vmax))))
            # Check for repeated columns
            if np.max(vdiff) / vabs <= 1e-4:
                # 1D array; use first column
                v = v[:, 0]
            else:
                # Nontrivial 2D array
                v = v[:, I]
        else:
            # Filter
            print("Encountered unexpected %iD col '%s'" % (v.ndim, dbcol))
        # Resave the column
        db.save_col(dbcol, v)
    # Check for y=0 seam curves
    if os.path.isfile(fsmy):
        # Read data but don't set as default
        db.make_seam("smy", fsmy, "smy.x", "smy.z", [])
    # Check for z=0 seam curves
    if os.path.isfile(fsmz):
        # Read data and set as default
        db.make_seam("smz", fsmz, "smz.x", "smz.y", llcols)
    # Output
    return db


# Get translated column names
def genr8_dbcolnames(dbi, comp=None):
    r"""Translate column names from one case line load

    :Call:
        >>> dbcols = genr8_llcolnames(dbi, comp=None)
    :Inputs:
        *dbi*: :class:`DataKit`
            DataKit read from single-case CAPE line load data file
        *comp*: {``None``} | :class:`str`
            Optional prefix for combined line load column names (e.g.
            shift ``"CN"`` to ``"CORE.dCN"`` instead of just ``"dCN"``)
    :Outputs:
        *dbcols*: :class:`dict`\ [:class:`str`]
            List of prefixed line load column names
    :Versions:
        * 2021-02-10 ``@ddalle``: Version 1.0
    """
    # Initialize column names
    dbcols = {}
    # Loop through columns
    for k, col in enumerate(dbi.cols):
        # Prefix column name if necessary
        if k == 0:
           # Use first column ("x") as-is
           dbcol = col
        else:
            # Prepend with "d"
            if col.startswith("d"):
                # In this case it's already there
                dbcol = col
            else:
                # Add the prefix
                dbcol = "d" + col
        # Check for a prefix
        if comp is not None:
            # Prepend the component name
            dbcol = "%s.%s" % (comp, dbcol)
        # Save it
        dbcols[col] = dbcol
    # Output
    return dbcols


# Get translated line load column names
def genr8_llcolnames(dbi, comp=None):
    r"""Generate names of line load columns from one case line load

    :Call:
        >>> llcols = genr8_llcolnames(dbi, comp=None)
    :Inputs:
        *dbi*: :class:`DataKit`
            DataKit read from single-case CAPE line load data file
        *comp*: {``None``} | :class:`str`
            Optional prefix for combined line load column names (e.g.
            shift ``"CN"`` to ``"CORE.dCN"`` instead of just ``"dCN"``)
    :Outputs:
        *llcols*: :class:`list`\ [:class:`str`]
            List of prefixed line load column names
    :Versions:
        * 2021-02-10 ``@ddalle``: Version 1.0
    """
    # Initialize column names
    llcols = []
    # Loop through columns
    for k, col in enumerate(dbi.cols):
        # Prefix column name if necessary
        if k == 0:
           # Not a line load column
           continue
        else:
            # Prepend with "d"
            if col.startswith("d"):
                # In this case it's already there
                dbcol = col
            else:
                # Add the prefix
                dbcol = "d" + col
        # Check for a prefix
        if comp is not None:
            # Prepend the component name
            dbcol = "%s.%s" % (comp, dbcol)
        # Save it
        llcols.append(dbcol)
    # Output
    return llcols


# Main function
def main():
    r"""Command-line interface to ``cape-writell``

    :Call:
        >>> main()
    :Versions:
        * 2021-02-10 ``@ddalle``: Version 1.0
    """
    # Process command-line parameters
    a, kw = argread.readkeys(sys.argv)
    # Check for "help" option
    if kw.get("h") or kw.get("help"):
        print(textutils.markdown(HELP_WRITELL))
        return
    # Read the datakit
    fname = kw.get("f")
    # Check for defaults
    if fname is None:
        # Loop through available defaults
        for fjson in _JSON_FNAMES:
            # Check if it exists
            if os.path.isfile(fjson):
                # Use it
                fname = fjson
                break
        else:
            # No match found
            print("No CAPE input JSON file found; use '-f' to specify one")
            return
    # Read the datakit
    cntl = Cntl(fname)
    # Get list of line load components
    llcomps = cntl.opts.get_DataBookByType("LineLoad")
    # Check for filters
    if len(a) == 0:
        # No positional args
        pats = []
    else:
        pats = list(a)
    # Check for "--ll" command-line optin
    kwll = kw.get("ll")
    # Append nontrivial option to patterns
    if kwll:
        pats.append(kwll)
    # Loop through candidate line load comps
    for comp in llcomps:
        # Check for pattern submset
        if pats:
            # Loop through patterns
            for pat in pats:
                # Check for a match
                if fnmatch.fnmatch(comp, pat):
                    # Found a match; do this one by exiting pat search
                    break
            else:
                # No pattern match found; go to next *comp*
                continue
        # Write it
        write_ll_datakit(cntl, comp)

