#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`cape.testutils.testopts`: Options module for testing utilities
======================================================================

This module primarily provides a simple class, :class:`TestOpts`, that reads a
JSON file for a small number of options the dictate how a test will be run.

"""

# Standard library modules
import os
import json


# Default attributes
rc = {
    "stdout": "STDOUT",
    "stderr": "STDERR",
    "CopyFiles": [],
    "LinkFiles": [],
    "Commands": [],
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
        
    # Process the options
    def process_kwargs(self, **kw):
        """Loop through known options and check for unknown keywords
        
        :Call:
            >>> opts.process_kwargs(**kw)
        :Inputs:
            
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
        msg = "TestOptions received unrecognized options:\n"
        # Loop through any remaining optoins
        for (k, v) in kw.items():
            # Append to error message
            msg += "  '%s'\n" % k
        # Raise exception using this message message
        raise ValueError(msg)
