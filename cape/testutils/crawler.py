#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`cape.testutils.crawler`: Functionality for CAPE's test crawler
=====================================================================

This module contains the underlying functions that operate the CAPE
test crawler.  This is a simple task that enters zero or more folders
and runs the main CAPE test utility.  This list of folders is set by
the user either directly or by pattern.  The default is to run a test
in each subfolder that exists.

The test crawler is initiated using the command:

    .. code-block:: console
    
        $ pc_TestCrawler.py

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
import glob

# Local modules
from . import testshell
from . import crawleropts


# Crawler class
class TestCrawler(object):
    
    # Standard attributes
    fname = "cape-test.json"
    RootDir = None
    opts = {}
    testdirs = []
    
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
        self.opts = crawleropts.TestCrawlerOpts(fname)
        # Get list of tests
        self.get_test_dirs()
        
    # Representation
    def __repr__(self):
        """Representation method
        
        :Versions:
            * 2019-07-03 ``@ddalle``: <TestCrawler(ntest=4)>
        """
        return "<%s(ntest=%i)>" % (
            self.__class__.__name__, len(self.testdirs))
        
    # String
    def __str__(self):
        """String method
        
        :Versions:
            * 2019-07-03 ``@ddalle``: <TestCrawler('/root/', [])>
        """
        return "<%s('%s', %s)>" % (
            self.__class__.__name__,
            self.RootDir, self.testdirs)
        
    # Process test list
    def get_test_dirs(self):
        """Process list of test directories from options
        
        This process the option *Glob* in *crawler.opts*
        
        :Call:
            >>> testdirs = crawler.get_test_dirs()
        :Inputs:
            *crawler*: :class:`TestCrawler`
                Test crawler controller
        :Attributes:
            *crawler.testdirs*: :class:`list`\ [:class:`str`]
                List of folders in which to conduct tests
        :Versions:
            * 2019-07-03 ``@ddalle``: First version
        """
        # Safely change to root folder
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Get option for which folders to enter
        o_glob = opts.get("Glob")
        # Convert to list
        if isinstance(o_glob, (list, tuple)):
            # Already a list; pass variable
            L_glob = o_glob
        elif o_glob is None:
            # Check all folders
            L_glob = []
        else:
            # Convert singleton to list
            L_glob = [o_glob]
        # Initialize list of tests
        testdirs = []
        # Loop through candidate globs
        for pattern in L_glob:
            # Find the matches
            fglob = glob.glob(pattern)
            # Sort these matches (no global sort in case user gives order)
            fglob.sort()
            # Loop through matches to add them to overall list
            for fdir in fglob:
                # Only process true folders
                if os.path.islink(fdir):
                    continue
                elif not os.path.islink(fdir):
                    continue
                # Check if it's already in the combined list
                if fdir not in testdirs:
                    testdirs.append(fdir)
        # Save the tests
        self.testdirs = testdirs
        # Return to original location
        os.chdir(fpwd)
        # Output
        return self.testdirs
        
    # Primary function
    def crawl(self, **kw):
        """Execute tests
        
        :Call:
            >>> crawler.crawl()
        :Inputs:
            *crawler*: :class:`TestCrawler`
                Test crawler controller
        :Versions:
            * 2019-07-03 ``@ddalle``: First version
        """
        # Update test list if inecessary
        self.get_test_dirs()
        # Safely change to root folder
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Loop through the tests
        for fdir in self.testdirs:
            
            pass
        # Go back to original location
        os.chdir(fpwd)
# class TestCrawler
    
    
# Command-line interface
def cli(*a, **kw):
    """Test crawler command-line interface
    
    :Call:
        >>> cli(*a, **kw)
    :Inputs:
        *f*, *fname*: {``"cape-test.json"``} | :class:`str`
            Name of JSON settings file for crawler
    :Versions:
        * 2019-07-03 ``@ddalle``: First version
    """
    # Get an instance of the crawler class
    crawler = TestCrawler(**kw)
    # Run the crawler
    crawler.crawl()


