#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`cape.testutils.testopts`: Options module for testing utilities
======================================================================

This module primarily provides a simple class, :class:`TestOpts`, that reads a
JSON file for a small number of options the dictate how a test will be run.


The allowable options and their default values are contained in the variable
*rc* from this module.  They are tabulated below:

    =============  ===============  ========================================
    Option         Default Value    Description
    =============  ===============  ========================================
    *stdout*       ``"STDOUT"``     Name of file to capture standard out
    *stderr*       ``"STDERR"``     Name of file to capture standard error
    *CopyFiles*    ``[]``           Files to copy into test folder
    *LinkFiles*    ``[]``           Files to link into test folder
    *Commands*     ``[]``           List of commands to run during test
    =============  ===============  ========================================
"""

# Standard library modules
import os
import json


# Local relative imports
from .verutils import ver


# Create :class:`unicode` class for Python 3+
if ver > 2:
    unicode = str


# Default attributes
rc = {
    "CopyFiles": [],
    "LinkFiles": [],
    "Commands": [],
    "STDOUT": "test.%02i.out",
    "STDERR": "test.%02i.err",
    "MaxTime": None,
    "MaxTimeCheckInterval": None,
    "ReturnCode": 0,
    "TargetSTDOUT": None,
    "TargetSTDERR": None,
    "RootLevel": None,
    "DocFolder": "doc/test",
    "DocTitle": None,
    "DocIntroFile": None,
    "MAXLINES": 10000,
    "NORMALIZE_WHITESPACE": False,
    "REGULAR_EXPRESSION": False,
    "VALUE_INTERVAL": True,
    "ELLIPSIS": True,
}


# Read JSON file
def read_json(fname):
    """Read options settings from a JSON file
    
    :Call:
        >>> optsd = read_json(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of JSON file to read
    :Outputs:
        *optsd*: :class:`dict`
            Contents of JSON file converted to dictionary
    :Versions:
        * 2019-06-28 ``@ddalle``: First version
    """
    # Check if file exists
    if not os.path.isfile(fname):
        raise SystemError("No test settings JSON file '%s'" % fname)
    # Open file
    with open(fname) as f:
        # Read as JSON
        dopts = json.load(f)
    # Output
    return dopts


# Convert to list
def enlist(v):
    """Convert item to list, if necessary
    
    :Call:
        >>> V = enlist(v)
    :Inputs:
        *v*: :class:`list` | :class:`str` | :class:`float`
            Any type
    :Outputs:
        *V*: :class:`list` | :class:`dict`
            Single-element list if *v* is not a :class:`list`
    :Versions:
        * 2019-07-05 ``@ddalle``: First version
    """
    # Check type
    if isinstance(v, list):
        # Already a list
        return v
    elif isinstance(v, dict):
        # Cannot convert
        return v
    elif isinstance(v, tuple):
        # Convert to list
        return list(v)
    elif isinstance(v, (str, unicode, int, float, bool)):
        # Singleton list
        return [v]
    else:
        # Unknown
        raise TypeError("Cannot convert type '%s'" % v.__class__.__name__)


# Get *n*th element of list, repeating last entry
def getel(V, i=None):
    """Get an element from a list, repeating last entry
    
    This will repeat the *first* entry if needed when *i* is negative
    
    :Call:
        >>> V = getel(V)
        >>> v = getel(V, i)
    :Inputs:
        *V*: :class:`list`
            List of items
        *i*: {``None``} | :class:`int`
            Index
    :Outputs:
        *V*: :class:`list`
            Entire input list if *i* is ``None``
        *v*: :class:`any`
            * ``V[i]``  if possible
            * ``V[-1]`` if ``i >= len(V)``
            * ``V[0]``  if ``i < -len(V)``
    :Versions:
        * 2019-07-05 ``@ddalle``: First version
    """
    # Check input types
    if not isinstance(V, (tuple, list)):
        # Cannot index
        return V
    elif i is None:
        # Return entire list
        return V
    # Length of *V*
    n = len(V)
    # Check cases
    if n == 0:
        # No entries
        return None
    elif i >= n:
        # Repeat last entry
        return V[-1]
    elif i < -n:
        # Repeat first entry
        return V[0]
    else:
        # Use common indexing if applicable
        return V[i]


# Options class
class TestOpts(dict):
    """Simple options class for :mod:`cape.testutils`
    
    :Call:
        >>> opts = TestOpts(fname="cape-test.json")
    :Inputs:
        *fname*: {``"cape-test.json"``} | :class:`str`
            Path to JSON file containing options
    :Outputs:
        *opts*: :class:`TestOpts`
            Options class based on :class:`dict`
    :Versions:
        * 2019-06-29 ``@ddalle``: First version
    """
    # Initialize
    def __init__(self, fname="cape-test.json"):
        """Initialization file
        
        :Versions:
            * 2019-06-28 ``@ddalle``: JSON only
        """
        # Read the JSON file
        dopts = read_json(fname)
        # Process JSON options as keywords (avoids changes to *dopts*)
        self.process_kwargs(**dopts)
        # Get commands
        cmds = enlist(self.get("Commands", []))
        # Save list form of commands
        self["Commands"] = cmds
        # Save number of commands
        self.n = len(cmds)
        
    # Process the options
    def process_kwargs(self, **kw):
        """Loop through known options and check for unknown keywords
        
        :Call:
            >>> opts.process_kwargs(**kw)
        :Inputs:
            *opts*: :class:`TestOpts`
                Test options class based on :class:`dict`
        :Versions:
            * 2019-06-29 ``@ddalle``: First version
        """
        # Loop through known parameters
        for (k, vdef) in rc.items():
            # Get value from *kw* if present
            v = kw.pop(k, vdef)
            # Save value
            self[k] = v
        # Exit if *kw* is drained
        if not kw:
            return
        # Initialize error text
        msg = "TestOpts received unrecognized options:\n"
        # Loop through any remaining optoins
        for (k, v) in kw.items():
            # Append to error message
            msg += "  '%s'\n" % k
        # Raise exception using this message message
        raise ValueError(msg)
        
    # Get entry
    def getel(self, k, i=None, vdef=None):
        """Get an element from a list, repeating last entry
    
        This will repeat the *first* entry if needed when *i* is negative
        
        :Call:
            >>> V = opts.getel(k)
            >>> v = opts.getel(k, i, vdef=None)
        :Inputs:
            *opts*: :class:`TestOpts`
                Test options class based on :class:`dict`
            *k*: :class:`str`
                Name of parameter
            *i*: {``None``} | :class:`int`
                Index
            *vdef*: {``None``} | :class:`any`
                Default value(s) for *k*
        :Outputs:
            *V*: :class:`list`
                Entire input list if *i* is ``None``
            *v*: :class:`any`
                * ``V[i]``  if possible
                * ``V[-1]`` if ``i >= len(V)``
                * ``V[0]``  if ``i < -len(V)``
        :Versions:
            * 2019-07-05 ``@ddalle``: First version
        """
        # Get values of option
        V = self.get(k, vdef)
        # Use indexing function
        return getel(V, i)
        
    # Get STDOUT and prepare it
    def get_STDOUT(self, i):
        """Get STDOUT option for case *i*, creating file if needed
        
        :Call:
            >>> fnout, fout = opts.get_STDOUT(i)
        :Inputs:
            *opts*: :class:`TestOpts`
                Test options class based on :class:`dict`
            *i*: {``None``} | :class:`int`
                Index
        :Outputs:
            *fnout*: ``None`` | :class:`str`
                Output file name, if appropriate
            *fout*: ``None`` | :class:`int` | :class:`file`
                File handle or identifier, if any
        :Versions:
            * 2019-07-05 ``@ddalle``: First version
        """
        # Get option for *i*
        fnout = self.getel("STDOUT", i, vdef=rc["STDOUT"])
        # Check type
        if fnout is None:
            # No STDOUT option
            return None, None
        elif isinstance(fnout, int):
            # File identifier but no name
            return None, fnout
        elif isinstance(fnout, (str, unicode)):
            # Check for '%' sign
            if '%' in fnout:
                # Use the index (1-based)
                fnouti = fnout % (i+1)
            else:
                # Fixed STDOUT file
                fnouti = fnout
            # Open the file name
            fout = open(fnouti, 'a+')
            # Output
            return fnouti, fout
        else:
            raise TypeError(
                "STDOUT has unrecognized type '%s'" %
                fnout.__class__.__name__)
        
    # Get STDERR and prepare it
    def get_STDERR(self, i, fout=None):
        """Get STDERR option for case *i*, creating file if needed
        
        :Call:
            >>> fnerr, ferr = opts.get_STDERR(i, fout=None)
        :Inputs:
            *opts*: :class:`TestOpts`
                Test options class based on :class:`dict`
            *i*: {``None``} | :class:`int`
                Index
            *fout*: ``None`` | :class:`int` | :class:`file`
                STDOUT File handle or identifier, if any
        :Outputs:
            *fnerr*: ``None`` | :class:`str`
                Output file name, if appropriate
            *ferr*: ``None`` | :class:`int` | :class:`file`
                STDERR File handle or identifier, if any
        :Versions:
            * 2019-07-05 ``@ddalle``: First version
        """
        # Get option for *i*
        fnout = self.getel("STDOUT", i, vdef=rc["STDOUT"])
        fnerr = self.getel("STDERR", i, vdef=rc["STDERR"])
        # Check type
        if fnerr is None:
            # No STDOUT option
            return None, None
        elif isinstance(fnerr, int):
            # File identifier but no name
            return None, fnerr
        elif isinstance(fnerr, (str, unicode)):
            # Check for '%' sign
            if '%' in fnerr:
                # Use the index (1-based)
                fnerri = fnerr % (i+1)
            else:
                # Fixed STDOUT file
                fnerri = fnerr
            # Check for '%' sign in STDOUT file name
            if isinstance(fnout, (str, unicode) ) and ('%' in fnout):
                # Use the index (1-based)
                fnouti = fnout % (i+1)
            else:
                # Fixed STDOUT file
                fnouti = fnout
            # Check for matching STDOUT and STDERR files
            if (fnouti == fnerri) and (fout is not None):
                # Reuse file handle
                ferr = fout
            else:
                # Open the file name
                ferr = open(fnerri, 'a+')
            # Output
            return fnerri, ferr
        else:
            raise TypeError(
                "STDERR has unrecognized type '%s'" %
                fnerr.__class__.__name__)
        
    # Get STDOUT comparison file
    def get_TargetSTDOUT(self, i):
        """Get target STDOUT file for case *i*
        
        :Call:
            >>> fnout = opts.get_TargetSTDOUT(i)
        :Inputs:
            *opts*: :class:`TestOpts`
                Test options class based on :class:`dict`
            *i*: {``None``} | :class:`int`
                Index
        :Outputs:
            *fnout*: ``None`` | :class:`str`
                Output file name, if appropriate
        :Versions:
            * 2019-07-05 ``@ddalle``: First version
        """
        # Get option for *i*
        fnout = self.getel("TargetSTDOUT", i, vdef=rc["TargetSTDOUT"])
        # Check type
        if fnout is None:
            # No target STDOUT option
            return None
        elif isinstance(fnout, (str, unicode)):
            # Check for '%' sign
            if '%' in fnout:
                # Use the index (1-based)
                fnouti = fnout % (i+1)
            else:
                # Fixed STDOUT file
                fnouti = fnout
            # Output
            return fnouti
        else:
            raise TypeError(
                "Target STDOUT has unrecognized type '%s'" %
                fnout.__class__.__name__)
        
    # Get STDOUT comparison file
    def get_TargetSTDERR(self, i):
        """Get target STDERR file for case *i*
        
        :Call:
            >>> fnerr = opts.get_TargetSTDERR(i)
        :Inputs:
            *opts*: :class:`TestOpts`
                Test options class based on :class:`dict`
            *i*: {``None``} | :class:`int`
                Index
        :Outputs:
            *fnerr*: ``None`` | :class:`str`
                Output file name, if appropriate
        :Versions:
            * 2019-07-05 ``@ddalle``: First version
        """
        # Get option for *i*
        fnerr = self.getel("TargetSTDERR", i, vdef=rc["TargetSTDERR"])
        # Check type
        if fnerr is None:
            # No target STDOUT option
            return None
        elif isinstance(fnerr, (str, unicode)):
            # Check for '%' sign
            if '%' in fnout:
                # Use the index (1-based)
                fneri = fnerr % (i+1)
            else:
                # Fixed STDOUT file
                fnerri = fnerr
            # Output
            return fnerri
        else:
            raise TypeError(
                "Target STDERR has unrecognized type '%s'" %
                fnerr.__class__.__name__)

    # Get options for file comparison
    def get_FileComparisonOpts(self, i=0):
        """Get options for file comparison tests
        
        :Call:
            >>> kw_comp = opts.get_FileComparisonOpts(i=0)
        :Inputs:
            *opts*: :class:`TestOpts`
                Test options class based on :class:`dict`
            *i*: {``0``} | :class:`int`
                Index
        :Outputs:
            *kw_comp*: :class:`dict`
                Options for text file comparison
        :See also:
            * :func:`cape.testutils.fileutils.compare_files`
            * :func:`cape.testutils.fileutils.compare_lines`
        :Versions:
            * 2019-07-05 ``@ddalle``: First version
        """
        # Initialize options
        kw_comp = {}
        # Loop through comparison options
        for k in [
            "MAX_LINES",
            "NORMALIZE_WHITESPACE",
            "REGULAR_EXPRESSION",
            "VALUE_INTERVAL",
            "ELLIPSIS"
        ]:
            # Save option
            kw_comp[k] = self.getel(k, i, vdef=rc[k])
        # Output
        return kw_comp
