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
import shlex
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
    
    
    # Run the main test
    def exec_test(self):
        """Execute the test controlled by the driver
        
        :Call:
            >>> ierr, ttot = driver.exec_test()
        :Inputs:
            *driver*: :class:`cape.testutils.driver.TestDriver`
                Test driver controller
        :Outputs:
            *ierr*: :class:`int`
                Exit status from last command or first to fail
            *ttot*: :class:`float`
                Total time used 
        :Versions:
            * 2019-07-05 ``@ddalle``: First version
        """
        # Go to home folder
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Prepare files (also enters working folder)
        self.prepare_files()
        # Run any commands
        ierr, ttot = self.run_commands()
        
        # Return to original location
        os.chdir(fpwd)
        # Output
        return ierr, ttot
    
    
    # Execute test
    def run_commands(self):
        """Execute tests in the current folder
        
        :Call:
            >>> ierr, ttot = driver.run_commands()
        :Inputs:
            *driver*: :class:`cape.testutils.driver.TestDriver`
                Test driver controller
        :Outputs:
            *ierr*: :class:`int`
                Exit status from last command or first to fail
            *ttot*: :class:`float`
                Total time used 
        :Versions:
            * 2019-07-05 ``@ddalle``: First version
        """
        # Get commands to run
        cmds = self.opts.get("Commands", [])
        # Get output file names
        fnout = self.opts.get("STDOUT", "STDOUT")
        fnerr = self.opts.get("STDERR", "STDERR")
        # Maximum allowed time
        tmax = self.opts.get("MaxTime", None)
        tstp = self.opts.get("MaxTimeCheckInterval", None)
        # Target exit status
        sts = self.opts.get("ExitStatus", 0)
        # Total Time used
        ttot = 0.0
        # Number of commands
        ncmd = len(cmds)
        # Loop through commands
        for i, cmd in enumerate(cmds):
            # Break command into parts
            cmdi = shlex.split(cmd)
            # Get handles
            fnout, fout = self.opts.get_STDOUT(i)
            fnerr, ferr = self.opts.get_STDERR(i, fout)
            # Target exit status
            stsi = self.opts.getel("ExitStatus", i, vdef=0)
            # Call the command
            t, ierr, out, err = testshell.comm(
                cmdi, maxtime=tmax, dt=tstp, stdout=fout, stderr=ferr)
            # Update time used
            ttot += t
            # Check for nonzero exit status
            if ierr != stsi:
                break
            # Process maximum time consideration
            if tmax:
                # Update time available
                tmax -= t
                # Check for expiration
                if tmax <= 0:
                    break
        # Close files
        if isinstance(fout, file):
            fout.close()
        # (No concern about closing same file twice if STDERR==STDOUT)
        if isinstance(ferr, file):
            ferr.close()
        # return exit status and total time used
        return ierr, ttot

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
        # Enter the folder
        os.chdir(fwork)
# class TestDriver

