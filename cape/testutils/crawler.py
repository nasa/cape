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
import sys
import glob
import math
import traceback

# Local modules
from . import crawleropts
from . import driver
from . import fileutils


# Crawler class
class TestCrawler(object):
    """Test crawler class
    
    :Call:
        >>> crawler = TestCrawler(**kw)
    :Inputs:
        *f*, *json*: {``"cape-test.json"``} | :class:`str`
            Name of JSON settings file
    :Outputs:
        *crawler*: :class:`cape.testutils.crawler.TestCrawler`
            Test crawler controller
    :Attributes:
        *fname*: :class:`str`
            Name of JSON file actually read
        *RootDir*: :class:`str`
            Absolute path to directory where instance is created
        *opts*: :class:`TestCrawlerOpts`
            Options for the test crawler
        *testdirs*: :class:`list`\ [:class:`str`]
            List of test folders
        *crawldirs*: :class:`list`\ [:class:`str`]
            List of folders to recurse crawler in
    :Versions:
        * 2019-07-03 ``@ddalle``: First version
    """
    
    # Standard attributes
    fname = "cape-test.json"
    RootDir = None
    opts = {}
    testdirs = []
    crawldirs = []
    results = {}
    
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
        o_glob = self.opts.get("Glob")
        # By default, check all folders
        if o_glob is None:
            o_glob = ["*"]
        # Get list of folders
        testdirs = fileutils.expand_file_list(o_glob, typ="d", error=False)
        # Save the tests
        self.testdirs = testdirs
        # Return to original location
        os.chdir(fpwd)
        # Output
        return self.testdirs
        
    # Process recursive crawl list
    def get_crawl_dirs(self):
        """Process list of test directories to recurse crawler in
        
        This process the option *CrawlGlob* in *crawler.opts*
        
        :Call:
            >>> crawldirs = crawler.get_crawl_dirs()
        :Inputs:
            *crawler*: :class:`TestCrawler`
                Test crawler controller
        :Attributes:
            *crawler.crawldirs*: :class:`list`\ [:class:`str`]
                List of folders in which to recurse crawler
        :Versions:
            * 2019-07-05 ``@ddalle``: First version
        """
        # Safely change to root folder
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Get option for which folders to enter
        o_glob = self.opts.get("CrawlGlob")
        # By default, check all folders
        if o_glob is None:
            o_glob = []
        # Get list of folders
        crawldirs = fileutils.expand_file_list(o_glob, typ="d", error=False)
        # Save the tests
        self.crawldirs = crawldirs
        # Return to original location
        os.chdir(fpwd)
        # Output
        return self.crawldirs
        
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
            * 2019-07-05 ``@ddalle``: Added recursion
        """
        # Update test list if necessary
        self.get_test_dirs()
        self.get_crawl_dirs()
        # Save current location
        fpwd = os.getcwd()
        # First status update
        print("Test folder '%s':" % os.path.split(self.RootDir)[1])
        # Number of tests
        ntest = len(self.testdirs)
        # Number of digits
        if ntest == 0:
            itest = 1
        else:
            itest = int(math.floor(math.log10(ntest))) + 1
        # Format string for status updates
        fmt1 = "  Test %%%ii: %%s ...\n" % itest
        fmt2 = "    PASS (%.4g seconds)\n"
        fmt3 = "    FAIL (command %i, %s) (%.4g seconds)\n"
        # Loop through the tests
        for (i, fdir) in enumerate(self.testdirs):
            # Status update
            sys.stdout.write(fmt1 % (i+1, fdir))
            sys.stdout.flush()
            # Enter the test folder
            os.chdir(self.RootDir)
            os.chdir(fdir)
            # Create a driver
            try:
                # Create the driver
                testd = driver.TestDriver()
            except Exception as e:
                # Get the message
                fmt = "%s: %s\n" % (e.__class__.__name__, e.message)
                # Indent it
                fmt = "".join(
                    ["    " + line + "\n" for line in fmt.split("\n")])
                # Show the STDERR output
                sys.stderr.write(fmt.rstrip() + "\n")
                sys.stderr.flush()
                # Some other problem
                testd = None
                # Create results
                results = {
                    "TestStatus": False,
                    "TestStatus_Init": False
                }
            # Run the test
            if testd is not None:
                # Run the driver to get results
                results = testd.run()
            # Get execution time
            ttot = results.get("TestRunTimeTotal", 0.0)
            # Determine status
            if results["TestStatus"]:
                # Success: show time
                msg = fmt2 % ttot
            else:
                # Failure: find reason
                tststr = results.get("TestStatus_Init", True)
                tstrc  = results.get("TestStatus_ReturnCode", [])
                tstt   = results.get("TestStatus_MaxTime",  [])
                tstout = results.get("TestStatus_STDOUT", [])
                tsterr = results.get("TestStatus_STDERR", [])
                # Find the first cause of failure, with preferred order
                if not tststr:
                    ifail = 0
                    reason = "JSON read"
                elif not all(tstrc):
                    ifail = tstrc.index(False)
                    reason = "return code"
                elif not all(tstt):
                    ifail = tstt.index(False)
                    reason = "max time"
                elif not all(tstout):
                    ifail = testout.index(False)
                    reason = "STDOUT"
                elif not all(tsterr):
                    ifail = testerr.index(False)
                    reason = "STDERR"
                else:
                    ifail = 0
                    reason = "no reason..."
                # Failure: show command and cause
                msg = fmt3 % (ifail+1, reason, ttot)
            # Final update
            sys.stdout.write(msg)
            sys.stdout.flush()
        # Get recursive folder list
        for fdir in self.crawldirs:
            # Enter the test folder
            os.chdir(self.RootDir)
            os.chdir(fdir)
            # Create a crawler
            crawler = self.__class__(**kw)
            # Run the crawler
            crawler.crawl()
        # Go back to original location
        os.chdir(fpwd)
# class TestCrawler
    
    
# Command-line interface
def cli(*a, **kw):
    """Test crawler command-line interface
    
    :Call:
        >>> cli(*a, **kw)
    :Inputs:
        *f*, *json*: {``"cape-test.json"``} | :class:`str`
            Name of JSON settings file for crawler
    :Versions:
        * 2019-07-03 ``@ddalle``: First version
    """
    # Get an instance of the crawler class
    crawler = TestCrawler(**kw)
    # Run the crawler
    crawler.crawl()


