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
        *testd*: :class:`cape.testutils.testd.TestDriver`
            Test driver controller
    :Versions:
        * 2019-07-03 ``@ddalle``: Started
    """
    
    # Standard attributes
    opts = {}
    fname = "cape-test.json"
    frst = None
    RootDir = None
    dirname = None
    # Results attributes
    TestStatus = True
    TestStatus_ReturnCode = True
    TestStatus_MaxTime = True
    TestStatus_STDOUT = True
    TestStatus_STDERR = True
    TestRunTimeTotal = 0.0
    TestRunTimeList = []
    TestReturnCodes = []
    TestCommandsNum = 0
    TestCommandsRun = 0
    
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
        # Save directory name
        self.dirname = os.path.split(self.RootDir)[1]
        # Process options
        self.opts = testopts.TestOpts(fname)
        # Get commands to run
        cmds = self.opts.get("Commands", [])
        # Ensure list
        cmds = testopts.enlist(cmds)
        # Save number of commands
        self.TestCommandsNum = len(cmds)
        
    # String method
    def __str__(self):
        """String method
        
        :Versions:
            * 2019-07-09 ``@ddalle``: <TestDriver('$dirname', n=$ncommands)>
        """
        return "<%s('%s', n=%i)>" % (
            self.__class__.__name__,
            self.dirname,
            len(testopts.enlist(self.opts.get("Commands", []))))

    # Representation method
    def __repr__(self):
        """Representation method
        
        :Versions:
            * 2019-07-09 ``@ddalle``: <TestDriver('$dirname', n=$ncommands)>
        """
        return "<%s('%s', n=%i)>" % (
            self.__class__.__name__,
            self.dirname,
            len(testopts.enlist(self.opts.get("Commands", []))))

    # Reset results for test
    def init_test_results(self):
        """(Re)initialize attributes that store results of test
        
        :Call:
            >>> testd.init_test_results()
        :Inputs:
            *testd*: :class:`cape.testutils.testd.TestDriver`
                Test driver controller
        :Versions:
            * 2019-07-07 ``@ddalle``: First version
        """
        # Reset test results attributes
        self.TestStatus = True
        # Reasons for failure
        self.TestStatus_ReturnCode = []
        self.TestStatus_MaxTime = []
        self.TestStatus_STDOUT = []
        self.TestStatus_STDERR = []
        # Statistics
        self.TestRunTimeTotal = 0.0
        self.TestRunTimeList = []
        self.TestReturnCodes = []
        self.TestCommandsNum = 0
        self.TestCommandsRun = 0
        
    # Run the main test
    def run(self):
        """Execute the test controlled by the driver
        
        :Call:
            >>> results = testd.run()
        :Inputs:
            *testd*: :class:`cape.testutils.testd.TestDriver`
                Test driver controller
        :Outputs:
            *results*: :class:`dict`
                Results from :func:`get_results_dict`
        :Versions:
            * 2019-07-05 ``@ddalle``: First version
        """
        # Go to home folder
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Prepare files (also enters working folder)
        self.prepare_files()
        # Begin documentation
        self.write_rst_intro()
        # Run any commands
        results = self.exec_commands()
        # Close file
        self.close_rst()
        # Return to original location
        os.chdir(fpwd)
        # Output
        return results

    # Prepare a test
    def prepare_files(self):
        """Prepare test folder for execution
        
        :Call:
            >>> testd.prepare_files()
        :Inputs:
            *testd*: :class:`cape.testutils.testd.TestDriver`
                Test driver controller
        :Attributes:
            *testd.frst*: ``None`` | :class:`file`
                File handle to new ReST file if applicable
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
        
    # Start log file
    def init_rst(self):
        """Initialize ReST file of test results
        
        :Call:
            >>> testd.init_rst()
        :Inputs:
            *testd*: :class:`cape.testutils.testd.TestDriver`
                Test driver controller
        :Attributes:
            *testd.frst*: ``None`` | :class:`file`
                Open or newly opened file handle if applicable
        :Versions:
            * 2019-07-09 ``@ddalle``: First version
        """
        # If *frst* is already a file, do nothing
        if isinstance(self.frst, file):
            return
        # Get option for root level
        nroot = self.opts.get("RootLevel")
        # Relative path to test documentation from "root"
        fdoc_rel = self.opts.get("DocFolder")
        # Check these options
        if nroot is None:
            # No root level
            self.close_rst()
            return
        elif fdoc_rel is None:
            # No documentation folder
            self.close_rst()
            return
        elif not isinstance(nroot, int):
            # Bad type for root level
            raise TypeError(
                "'RootLevel' option must be int (got '%s')"
                % froot.__class__.__name__)
        elif nroot > 0:
            # Bad root level
            raise ValueError(
                "'RootLevel' option must be <= 0 (got %i)" % nroot)
        elif not isinstance(fdoc_rel, (str, unicode)):
            # Bad type for doc folder
            raise TypeError(
                "'DocFolder' option must be str (got '%s')"
                % fdoc_rel.__class__.__name__)
        # Initialize root folder for documentation
        fdoc = self.RootDir
        # Go up *nroot* levels
        for i in range(-nroot):
            # Raise one level
            fdoc = os.path.dirname(fdoc)
        # Remember current location
        fpwd = os.getcwd()
        # Catch errors during folder generation
        try:
            # Change to documentation root folder
            os.chdir(fdoc)
            # List of subdirectories, last one copies test folder name
            docdirs = fdoc_rel.split("/") + [self.dirname]
            # Create folders as needed
            for fdir in docdirs:
                # Check if folder exists
                if not os.path.isdir(fdir):
                    # Otherwise, create it
                    os.mkdir(fdir)
                # Enter it
                os.chdir(fdir)
                # Join to doc folder
                fdoc = os.path.join(fdoc, fdir)
            # Return to original location
            os.chdir(fpwd)
        except Exception:
            # Return to original location
            os.chdir(fpwd)
            # Fail
            raise SystemError(
                "Failed to create folder '%s' in '%s'" % (fdir, fdoc))
        # Total path
        fname = os.path.join(fdoc, "index.rst")
        # Open the file
        self.frst = open(fname, "w")

    # Write header for ReST file
    def write_rst_intro(self):
        """Write intro section for ReST log file
        
        :Call:
            >>> testd.write_rst_intro()
        :Inputs:
            *testd*: :class:`cape.testutils.testd.TestDriver`
                Test driver controller
        :Attributes:
            *testd.frst*: ``None`` | :class:`file`
                File handle to which intro is written, if applicable
        :Versions:
            * 2019-07-09 ``@ddalle``: First version
        """
        # Open file
        self.init_rst()
        # Check if file is actually open
        if self.frst is None:
            return
        # Get title
        ttl = self.opts.get("DocTitle")
        # Default title
        if ttl is None:
            # Use the folder name
            ttl = "Test ``%s``" % self.dirname
        # Get the current time
        t = time.localtime()
        # Get timezone name using DST flag
        if t.tm_isdst:
            # Daylight savings timezone name
            tz = time.tzname[1]
        else:
            # Standard timezone name
            tz = time.tzname[0]
        # Write a header comment
        self.frst.write("\n")
        self.frst.write(".. This documentation written by TestDriver()\n")
        self.frst.write("   on ")
        self.frst.write("%04i-%02i-%02i " % (t.tm_year, t.tm_mon, t.tm_mday))
        self.frst.write("at %02i:%02i %s" % (t.tm_hour, t.tm_min, tz))
        self.frst.write("\n\n")
        # Write title
        self.frst.write(ttl + "\n")
        self.frst.write("=" * (len(ttl) + 2))
        self.frst.write("\n\n")
        # Check for intro written beforehand
        fintro = self.opts.get("DocIntroFile")
        # Check if it's a file name and exists
        if fintro is None:
            # Do nothing
            pass
        elif not isinstance(fintro, (str, unicode)):
            # Invalid type
            raise TypeError(
                "'DocIntroFile' must be a string (got '%s')"
                % fintro.__class__.__name__)
        else:
            # Absolute path
            if not os.path.isabs(fintro):
                # Relative to test folder, not working folder
                fintro = os.path.join(self.RootDir, fintro)
            # Check if file exists
            if not os.path.isfile(fintro):
                raise SystemError("DocIntroFile '%s' does not exist" % fintro)
            # Otherwise, copy the file
            self.fsrt.write(open(fintro).read())
            # Add a blank line for good measure
            self.frst.write("\n")
    
    # Close ReST file
    def close_rst(self):
        """Close ReST log file, if open
        
        :Call:
            >>> testd.close_rst()
        :Inputs:
            *testd*: :class:`cape.testutils.testd.TestDriver`
                Test driver controller
        :Attributes:
            *testd.frst*: ``None`` | :class:`file`
                Closed file handle if applicable
        :Versions:
            * 2019-07-08 ``@ddalle``: First version
        """
        # Check if *frst* is a file
        if isinstance(self.frst, file):
            # Close it
            self.frst.close()
            # Delete handle
            self.frst = None

    # Execute test
    def exec_commands(self):
        """Execute tests in the current folder
        
        :Call:
            >>> results = testd.exec_commands()
        :Inputs:
            *testd*: :class:`cape.testutils.testd.TestDriver`
                Test driver controller
        :Outputs:
            *results*: :class:`dict`
                Results from :func:`get_results_dict`
        :Versions:
            * 2019-07-05 ``@ddalle``: First version
        """
        # Get commands to run
        cmds = self.opts.get("Commands", [])
        # Ensure list
        cmds = testopts.enlist(cmds)
        # Initialize test results
        self.init_test_results()
        # Save number of commands
        self.TestCommandsNum = len(cmds)
        # Loop through commands
        for i, cmd in enumerate(cmds):
            # Break command into parts
            cmdi = shlex.split(cmd)
            # Get handles
            fnout, fout = self.opts.get_STDOUT(i)
            fnerr, ferr = self.opts.get_STDERR(i, fout)
            # Maximum allowed time
            tmax = self.opts.getel("MaxTime", i, vdef=None)
            tstp = self.opts.getel("MaxTimeCheckInterval", i, vdef=None)
            # Total time is cumulative
            if tmax:
                tmax = tmax - self.TestRunTimeTotal
            # Call the command
            t, ierr, out, err = testshell.comm(
                cmdi, maxtime=tmax, dt=tstp, stdout=fout, stderr=ferr)
            # Close files
            if isinstance(fout, file):
                fout.close()
            # (No concern about closing same file twice if STDERR==STDOUT)
            if isinstance(ferr, file):
                ferr.close()
            # Check the return code
            self.check_status_returncode(i, ierr)
            # Check for timeout and update timers
            self.check_status_maxtime(i, t)
            # Check STDOUT and STDERR against targets
            self.check_status_stdout(i, fnout)
            self.check_status_stderr(i, fnerr, fnout)
            # Update number of commands run
            self.TestCommandsRun = i + 1
            # Exit if a failure was detected
            if not self.TestStatus:
                break
        # Output
        return self.get_results_dict()

    # Ensure an attribute has at least *i* entries
    def _extend_attribute_list(self, k, i):
        """Ensure attribute *k* is a :class:`list` with *i* entries
        
        :Call:
            >>> testd._extend_attribute_list(k, i)
        :Inputs:
            *testd*: :class:`cape.testutils.testd.TestDriver`
                Test driver controller
            *k*: :class:`str`
                Name of attribute
            *i*: :class:`int`
                Command number
        :Versions:
            * 2019-07-08 ``@ddalle``: First version
        """
        # Get the attribute
        V = getattr(self, k)
        # Check it
        if not isinstance(V, list):
            raise AttributeError("No list-type driver attribute '%s'" % k)
        # Number of entries in attribute
        n = len(V)
        # Append if needed
        for j in range(n, i+1):
            V.append(None)

    # Check return code status
    def check_status_returncode(self, i, ierr):
        """Check the return code against target for command *i*
        
        :Call:
            >>> q = testd.check_status_returncode(i, ierr)
        :Inputs:
            *testd*: :class:`cape.testutils.testd.TestDriver`
                Test driver controller
            *i*: :class:`int`
                Command number
            *ierr*: :class:`int`
                Exit status from command *i*
        :Outputs:
            *q*: ``True`` | ``False``
                Whether or not *ierr* matches expected value
        :Attributes:
            *testd.TestReturnCode[i]*: *ierr*
                Return code save for future reference
            *testd.TestStatus_ReturnCode[i]*: *q*
                Whether or not *ierr* matches expected value
            *testd.TestStatus*: ``True`` | ``False``
                Set to ``False`` if above test fails
        :Versions:
            * 2019-07-08 ``@ddalle``: First version
        """
        # Extend attributes as necessary
        self._extend_attribute_list("TestStatus_ReturnCode", i)
        self._extend_attribute_list("TestReturnCodes", i)
        # Target return code
        rc_target = self.opts.getel("ReturnCode", i, vdef=0)
        # Save the return code
        self.TestReturnCodes[i] = ierr
        # Check result
        q = (ierr == rc_target)
        # Check for nonzero exit status
        if q:
            # Update status lists
            self.TestStatus_ReturnCode[i] = True
        else:
            # Update status lists
            self.TestStatus_ReturnCode[i] = False
            # Fail the test and abort
            self.TestStatus = False
        # Output
        return q
        
    # Check timer status
    def check_status_maxtime(self, i, t):
        """Check the maximum time for command *i*
        
        :Call:
            >>> q = testd.check_status_maxtime(i, t)
        :Inputs:
            *testd*: :class:`cape.testutils.testd.TestDriver`
                Test driver controller
            *i*: :class:`int`
                Command number
            *t*: :class:`float`
                Time used by command *i*
        :Outputs:
            *q*: ``True`` | ``False``
                Whether or not *t* exceeds proscribed *MaxTime*
        :Attributes:
            *testd.TestRunTimeList[i]*: *t*
                Execution time for command *i* saved
            *testd.TestRunTimeTotal*: :class:`float`
                Total execution time increased by *t*
            *testd.TestStatus_MaxTime[i]*: *q*
                Whether or not total time exceeds proscription
            *testd.TestStatus*: ``True`` | ``False``
                Set to ``False`` if above test fails
        :Versions:
            * 2019-07-08 ``@ddalle``: First version
        """
        # Maximum time
        tmax = self.opts.getel("MaxTime", i, vdef=None)
        # Number of entries in *TestStatus_MaxTime*
        ntest = len(self.TestStatus_MaxTime)
        # Check if already processed
        if i <= ntest - 1:
            return self.TestStatus_MaxTime[i]
        # Get preceding total time used
        ttot = self.TestRunTimeTotal
        # Update timers
        self.TestRunTimeList.append(t)
        self.TestRunTimeTotal += t
        # Process maximum time consideration
        if tmax:
            # Check for expiration
            q = (ttot + t < tmax)
        else:
            # No test to fail
            q = True
        # Save status
        self.TestStatus_MaxTime.append(q)
        # Update overall status
        self.TestStatus = self.TestStatus and q
        # Output
        return q
        
    # Check contents of STDOUT
    def check_status_stdout(self, i, fnout):
        """Compare STDOUT from command *i* to target
        
        :Call:
            >>> q = testd.check_status_stdout(i, fnout)
        :Inputs:
            *testd*: :class:`cape.testutils.testd.TestDriver`
                Test driver controller
            *i*: :class:`int`
                Command number
            *fnout*: :class:`str`
                Name of file that captured STDOUT from command
        :Outputs:
            *q*: ``True`` | ``False``
                Whether or not STDOUT matched target (``True`` if no
                target specified)
        :Attributes:
            *testd.TestStatus_STDOUT[i]*: *q*
                Whether or not STDOUT matched target
            *testd.TestStatus*: ``True`` | ``False``
                Set to ``False`` if above test fails
        :Versions:
            * 2019-07-08 ``@ddalle``: First version
        """
        # Get target STDOUT file
        fntout = self.opts.get_TargetSTDOUT(i)
        # Extend attributes as necessary
        self._extend_attribute_list("TestStatus_STDOUT", i)
        # Perform test on STDOUT
        if not fntout:
            # No target: PASS
            q = True
        elif not fnout:
            # No STDOUT file: unacceptable if target
            q = False
        elif not isinstance(fnout, (str, unicode)):
            # STDOUT not mapped to file: unacceptable if target
            q = False
        else:
            # Get options for file comparisons
            kw_comp = self.opts.get_FileComparisonOpts(i)
            # Target is in the parent folder
            if not os.path.isabs(fntout):
                # If relative, compare to parent
                fntout = os.path.join(os.path.realpath(".."), fntout)
            # Compare STDOUT files
            q = fileutils.compare_files(fnout, fntout, **kw_comp)
        # Save result
        self.TestStatus_STDOUT[i] = q
        # Output
        return q

    # Check contents of STDERR
    def check_status_stderr(self, i, fnerr, fnout):
        """Compare STDERR from command *i* to target
        
        :Call:
            >>> q = testd.check_status_stderr(i, fnerr, fnout)
        :Inputs:
            *testd*: :class:`cape.testutils.testd.TestDriver`
                Test driver controller
            *i*: :class:`int`
                Command number
            *fnerr*: :class:`str`
                Name of file that captured STDERR from command
            *fnout*: :class:`str`
                Name of file that captured STDOUT from command
        :Outputs:
            *q*: ``True`` | ``False``
                Whether or not STDERR matched target (``True`` if no
                target specified or if SDTERR mapped to STDOUT)
        :Attributes:
            *testd.TestStatus_STDERR[i]*: *q*
                Whether or not STDERR matched target
            *testd.TestStatus*: ``True`` | ``False``
                Set to ``False`` if above test fails
        :Versions:
            * 2019-07-08 ``@ddalle``: First version
        """
        # Get target STDOUT file
        fnterr = self.opts.get_TargetSTDERR(i)
        # Extend attributes as necessary
        self._extend_attribute_list("TestStatus_STDERR", i)
        # Perform test on STDERR
        if not fnterr:
            # No target: PASS
            q = True
        elif not fnerr:
            # No STDERR file: unacceptable if target
            q = False
        elif not isinstance(fnerr, (str, unicode)):
            # STDERR not mapped to file: unacceptable if target
            q = False
        elif (fnerr == fnout):
            # STDERR mapped to STDOUT file; unacceptable if target
            q = False
        else:
            # Get options for file comparisons
            kw_comp = self.opts.get_FileComparisonOpts(i)
            # Target is in the parent folder
            if not os.path.isabs(fnterr):
                # If relative, compare to parent
                fnterr = os.path.join(os.path.realpath(".."), fnterr)
            # Compare STDOUT files
            q = fileutils.compare_files(fnerr, fnterr, **kw_comp)
        # Save result
        self.TestStatus_STDERR[i] = q
        # Output
        return q

    # Get results dictionary
    def get_results_dict(self):
        """Create a dictionary of results from the test
        
        :Call:
            >>> results = testd.get_results_dict()
        :Inputs:
            *testd*: :class:`cape.testutils.testd.TestDriver`
                Test driver controller
        :Outputs:
            *results*: :class:`dict`
                Dictionary of results with the following keys
            *TestStatus*: ``True`` | ``False``
                Overall result of the test
            *TestCommandsNum*: :class:`int` >= 0
                Number of commands proscribed
            *TestCommandsRun*: :class:`int` >= 0
                Number of commands actually run
            *TestStatus_ReturnCode*: :class:`list`\ [:class:`bool`]
                Return code test results for each command
            *TestStatus_MaxTime*: :class:`list`\ [:class:`bool`]
                Timeout test results for each command
            *TestStatus_STDOUT*: :class:`list`\ [:class:`bool`]
                STDOUT comparison test results for each command
            *TestStatus_STDERR*: :class:`list`\ [:class:`bool`]
                STDERR comparison test results for each command
            *TestReturnCodes*: :class:`list`\ [:class:`int`]
                Return codes fro each command run
            *TestRunTimeTotal*: :class:`float`
                Total time taken by all commands run
            *TestRunTimeList*: :class:`list`\ [:class:`float`]
                Time taken by each command run
        :Versions:
            * 2019-07-09 ``@ddalle``: First version
        """
        # Create dictionary and return it
        return {
            "TestStatus":            self.TestStatus,
            "TestCommandsNum":       self.TestCommandsNum,
            "TestCommandsRun":       self.TestCommandsRun,
            "TestStatus_ReturnCode": self.TestStatus_ReturnCode,
            "TestStatus_MaxTime":    self.TestStatus_MaxTime,
            "TestStatus_STDOUT":     self.TestStatus_STDOUT,
            "TestStatus_STDERR":     self.TestStatus_STDERR,
            "TestReturnCodes":       self.TestReturnCodes,
            "TestRunTimeTotal":      self.TestRunTimeTotal,
            "TestRunTimeList":       self.TestRunTimeList,
        }
                
# class TestDriver
    
    
# Command-line interface
def cli(*a, **kw):
    """Test case command-line interface
    
    :Call:
        >>> results = cli(*a, **kw)
    :Inputs:
        *f*, *json*: {``"cape-test.json"``} | :class:`str`
            Name of JSON settings file for crawler
    :Outputs:
        *results*: :class:`dict`
            Dictionary of results with the following keys
        *TestStatus*: ``True`` | ``False``
            Overall result of the test
        *TestCommandsNum*: :class:`int` >= 0
            Number of commands proscribed
        *TestCommandsRun*: :class:`int` >= 0
            Number of commands actually run
        *TestStatus_ReturnCode*: :class:`list`\ [:class:`bool`]
            Return code test results for each command
        *TestStatus_MaxTime*: :class:`list`\ [:class:`bool`]
            Timeout test results for each command
        *TestStatus_STDOUT*: :class:`list`\ [:class:`bool`]
            STDOUT comparison test results for each command
        *TestStatus_STDERR*: :class:`list`\ [:class:`bool`]
            STDERR comparison test results for each command
        *TestReturnCodes*: :class:`list`\ [:class:`int`]
            Return codes fro each command run
        *TestRunTimeTotal*: :class:`float`
            Total time taken by all commands run
        *TestRunTimeList*: :class:`list`\ [:class:`float`]
            Time taken by each command run
    :Versions:
        * 2019-07-03 ``@ddalle``: First version
        * 2019-07-08 ``@ddalle``: Added output
    """
    # Get an instance of the crawler class
    driver = TestDriver(**kw)
    # Run the crawler
    return testd.run()

