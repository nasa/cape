#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`cape.testutils.crawleropts`: Options for CAPE test crawler
=================================================================

Process options for the test crawler, which by default attempts to run
a test in each sub folder of the current directory.  Users may instead
opt to run tests only in specified folders or run tests in any folders
matching one or more specified patterns.
"""

# Standard library modules
import os
import json

# Local partial imports
from .testopts import read_json


# Default options
rc = {
    "Glob": "*",
}

# Options calss
class TestCrawlerOpts(dict):
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
        # Process options (if any)
        if os.path.isfile(fname) or (fname != "cape-test.json"):
            # Read the JSON file or fail to read non-default file
            dopts = read_json(fname)
        else:
            # Use empty options
            dopts = {}
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
        msg = "TestCrawlerOpts received unrecognized options:\n"
        # Loop through any remaining optoins
        for (k, v) in kw.items():
            # Append to error message
            msg += "  '%s'\n" % k
        # Raise exception using this message message
        raise ValueError(msg)

