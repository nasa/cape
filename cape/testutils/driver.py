#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`cape.testutils.driver`: CAPE's main test case driver
===========================================================

This module contains the underlying functions that operate the CAPE
test driver for individual tests.


The test crawler is initiated using the command:

    .. code-block:: console
    
        $ pc_TestCase.py

This calls the :func:`cli` command from this module.

Options are processed using the :mod:`cape.testutils.crawleropts`
module, which looks for the ``cape-test.json`` file to process any
options to the test crawler.

:See Also:
    * :mod:`cape.testutils.testopts`
    * :mod:`cape.testutils.crawleropts`
"""

# Standard library modules
import os
import time
import shutil

# Local modules
from . import fileutils
from . import testshell
from . import testopts


# Crawler class
class TestDriver(object):
    """Test driver class
    
    :Call:
        >>> driver = TestDriver(**kw)
    :Inputs:
        *f*, *json*: {``"cape-test.json"``} | :class:`str`
            Name of JSON settings file
    :Outputs:
        *driver*: :class:`cape.testutils.driver.TestDriver`
            Test driver controller
    :Versions:
        * 2019-07-03 ``@ddalle``: Started
    """
    
    # Standard attributes
    fname = "cape-test.json"
    RootDir = None
    opts = {}
    
    # Initialization method
    def __init__(self, *a, **kw):
        """Initialization method
        
        :Versions:
            * 2019-07-03 ``@ddalle``: First version
        """
        # Process options file name
        fname = kw.pop("f", kw.pop("json", "cape-test.json"))
        # Save name of file
        self.fname = os.path.split(fname)[1]
        # Save current directory
        self.RootDir = os.getcwd()
        # Process options
        self.opts = testopts.TestOpts(fname)
        
    # String method
    
    
    # Representation method
    
    
    # Execute test
    def run_commands(self):
        """Execute tests in the current folder
        
        :Call:
            >>> statuses = driver.run_commands()
        :Inputs:
            *driver*: :class:`cape.testutils.driver.TestDriver`
                Test driver controller
        :Outputs:
        :Versions:
            * 2019-07-03 ``@ddalle``: First version
        """
        # Get commands to run
        cmds = self.opts.get("Commands", [])
        # Get output file names
        fnout = self.opts.get("stdout", "STDOUT")
        fnerr = self.opts.get("stderr", "STDERR")
        # Maximum allowed time
        
        # Open output files
        if fnout is None:
            # No output file
            fout = None
        elif isinstance(fnout, int):
            # Identifier; leave as is
            fout = fnout
        else:
            # Open file
            fout = open(fnout, 'w')
        # Open STDERR file
        if fnerr is None:
            # No error file
            ferr = None
        elif isinstance(fnerr, int):
            # Identifier; leave as is
            ferr = fnerr
        elif fnerr == fnout:
            # Directing both streams to same file
            ferr = fout
        else:
            # Open STDERR file
            ferr = open(fnerr, 'w')
        # Loop through commands
        for i, cmd in enumerate(cmds):
            # Call the command 
            t, ierr, out, err = testshell.comm(
                cmd, maxtime=tmax, dt=dt, stdout=stdout, stderr=stderr)
        # Close files
        if isinstance(fout, file):
            fout.close()
        if isinstance(ferr, file):
            ferr.close()
            
    
    
    # Prepare a test
    def prepare_files(self):
        """Prepare test folder for execution
        
        :Call:
            >>> driver.prepare_files()
        :Inputs:
            *driver*: :class:`cape.testutils.driver.TestDriver`
                Test driver controller
        :Versions:
            * 2019-07-03 ``@ddalle``: First version
        """
        # Name of container folder
        fwork = self.opts.get("ContainerName", "work")
        # Delete contents if present
        if os.path.isdir(fwork):
            shutil.rmtree(fwork)
        # Create folder
        os.mkdir(fwork)
        # Get files to copy/link
        fcopy = self.opts.get("CopyFiles", [])
        flink = self.opts.get("LinkFiles", [])
        dcopy = self.opts.get("CopyDirs", [])
        dlink = self.opts.get("LinkDirs", [])
        # Copy files
        for fname in fileutils.expand_file_list(fcopy, typ="f"):
            # Double-check for file
            if not os.path.isfile(fname):
                continue
            # Copy it
            shutil.copy(fname, os.path.join(fwork, fname))
        # Link files
        for fname in fileutils.expand_file_list(flink, typ="f"):
            # Double-check for file
            if not os.path.isfile(fname):
                continue
            # Link it
            os.symlink(fname, os.path.join(fwork, fname))
        # Copy dirs
        for fname in fileutils.expand_file_list(dcopy, typ="d"):
            # Double-check for dir
            if not os.path.isdir(fname):
                continue
            # Copy folder and its contents
            shutil.copytree(fname, os.path.join(fwork, fname))
        # Link dirs
        for fname in fileutils.expand_file_list(dlink, typ="d"):
            # Double-check for dir
            if not os.path.isdir(fname):
                continue
            # Create link to folder and its contents
            os.symlink(fname, os.path.join(fwork, fname))
# class TestDriver

