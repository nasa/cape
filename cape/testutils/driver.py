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
    fdoc = None
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
        cmds = self.opts.get_commands()
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
            len(self.opts.get_commands()))

    # Representation method
    def __repr__(self):
        """Representation method
        
        :Versions:
            * 2019-07-09 ``@ddalle``: <TestDriver('$dirname', n=$ncommands)>
        """
        return "<%s('%s', n=%i)>" % (
            self.__class__.__name__,
            self.dirname,
            len(self.opts.get_commands()))

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
        # Save documentation folder
        self.fdoc = fdoc
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
        # Create an indent
        tab = "    "
        # Get handle to save some characters
        f = self.frst
        # Write a header comment
        f.write("\n")
        f.write(".. This documentation written by TestDriver()\n")
        f.write("   on ")
        f.write("%04i-%02i-%02i " % (t.tm_year, t.tm_mon, t.tm_mday))
        f.write("at %02i:%02i %s" % (t.tm_hour, t.tm_min, tz))
        f.write("\n\n")
        # Write title
        f.write(ttl + "\n")
        f.write("=" * (len(ttl) + 2))
        f.write("\n\n")
        # Indentation
        tab = "    "
        # Check for intro written beforehand
        fintro = self.opts.get("DocFileIntro")
        # Check if it's a file name and exists
        if fintro is None:
            # Do nothing
            pass
        elif not isinstance(fintro, (str, unicode)):
            # Invalid type
            raise TypeError(
                "'DocFileIntro' must be a string (got '%s')"
                % fintro.__class__.__name__)
        else:
            # Absolute path
            if not os.path.isabs(fintro):
                # Relative to test folder, not working folder
                fintro = os.path.join(self.RootDir, fintro)
            # Check if file exists
            if not os.path.isfile(fintro):
                raise SystemError("DocFileIntro '%s' does not exist" % fintro)
            # Otherwise, copy the file
            f.write(open(fintro).read())
            # Add a blank line for good measure
            f.write("\n")
        # Summary: location
        f.write("This test is run in the folder:\n\n")
        f.write("    ``%s%s``\n\n" % (self.RootDir, os.sep))
        # Summary: container
        fwork = self.opts.get("ContainerName", "work")
        f.write("and the working folder for the test is\n\n")
        f.write("    ``%s%s``\n\n" % (fwork, os.sep))
        # Summary: commands
        f.write("The commands executed by this test are\n\n")
        f.write(tab + ".. code-block:: console\n\n")
        # Loop through commands
        for cmd in self.opts.get_commands():
            f.write(2*tab + "$ " + cmd + "\n")
        # Blank line
        f.write("\n")
        # Check for files to print in folder
        fshow_list = self.opts.get("DocFilesShow", [])
        fshow_list = testopts.enlist(fshow_list)
        # Loop through files
        for (i, fshow) in enumerate(fshow_list):
            # Absolute path
            if not os.path.isabs(fshow):
                fshow = os.path.join(self.RootDir, fshow)
            # # Check if file exists
            if not os.path.isfile(fshow):
                raise SystemError("DocFilesShow '%s' does not exist" % fshow)
            # Get lexer
            lang = self.opts.getel("DocFilesLexer", i, vdef="none")
            # Split into parts
            fdir, fname = os.path.split(fshow)
            # Create a header
            f.write("**Included file:** ``%s``\n\n" % fname)
            # Start a code block
            f.write(tab + (".. code-block:: %s\n\n" % lang))
            # Insert contents
            for line in open(fshow).readlines():
                # Write it
                f.write(tab + tab + line)
            # Check for that stupid modern convention that the last
            # line doesn't end with a newline
            if not line.endswith("\n"):
                # End the damn line
                f.write("\n")
            # Blank line
            f.write("\n")
        # Check for files to link
        flink_list = self.opts.get("DocFilesLink", [])
        flink_list = testopts.enlist(flink_list)
        # Create header if appropriate
        if len(flink_list) > 0:
            # Start bullet list
            f.write(":Included Files:\n")
        # Loop through files
        for flink in flink_list:
            # Absolute path
            if not os.path.isabs(flink):
                flink = os.path.join(self.RootDir, flink)
            # # Check if file exists
            if not os.path.isfile(flink):
                raise SystemError("DocFilesLink '%s' does not exist" % flink)
            # Split into parts
            fdir, fname = os.path.split(flink)
            # Copy the file
            shutil.copy(flink, os.path.join(self.fdoc, fname))
            # Create link
            f.write("    * :download:`%s`\n" % fname)
        # Blank line after download list
        if len(flink_list) > 0:
            # Start bullet list
            f.write("\n")
            
            
        
    
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
        cmds = self.opts.get_commands()
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
            # Convert maximum time to seconds
            tsec = testshell._time2sec(tmax)
            # Total time is cumulative
            if tsec:
                # Subtract any existing time from previous commands
                tsec = tsec - self.TestRunTimeTotal
            # Call the command
            t, ierr, out, err = testshell.comm(
                cmdi, maxtime=tsec, dt=tstp, stdout=fout, stderr=ferr)
            # Close files
            if isinstance(fout, file):
                fout.close()
            # (No concern about closing same file twice if STDERR==STDOUT)
            if isinstance(ferr, file):
                ferr.close()
            # Start the log for this command
            self.process_results_summary(i, cmd)
            # Check the return code
            self.process_results_returncode(i, ierr)
            # Check for timeout and update timers
            self.process_results_maxtime(i, t)
            # Check STDOUT and STDERR against targets
            self.process_results_stdout(i, fnout)
            self.process_results_stderr(i, fnerr, fnout)
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

    # Start log output
    def process_results_summary(self, i, cmd):
        """Start the reST results summary for command *i*
        
        :Call:
            >>> testd.process_results_summary(i, cmd)
        :Inputs:
            *testd*: :class:`cape.testutils.testd.TestDriver`
                Test driver controller
            *i*: :class:`int`
                Command number
            *cmd*: :class:`str`
                The command that was run in step *i*
        :Versions:
            * 2019-07-09 ``@ddalle``: First version
        """
        # Get file handle
        f = self.frst
        # reST output
        if not isinstance(f, file) or f.closed:
            return
        # Get subtitle
        subt = self.opts.getel("CommandTitles", i, vdef=None)
        # Create an indent
        tab = "    "
        # Form the title for subsection
        if subt:
            # Include subtitle
            ttl = "Command %i: %s\n" % (i+1, subt)
        else:
            # No subtitle; just command number
            ttl = "Command %i\n" % (i+1)
        # Write title
        f.write(ttl)
        # Delimiter line
        f.write("-" * len(ttl))
        f.write("\n\n")
        # Show the command
        f.write(":Command:\n")
        f.write(tab)
        f.write(".. code-block:: console\n\n")
        f.write(tab + tab + "$ " + cmd)
        f.write("\n\n")
            
    # Check return code status
    def process_results_returncode(self, i, ierr):
        """Check the return code against target for command *i*
        
        :Call:
            >>> q = testd.process_results_returncode(i, ierr)
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
        # Get file handle
        f = self.frst
        # reST output
        if isinstance(f, file) and (not f.closed):
            # Return code section
            f.write(":Return Code:\n")
            # Write status of the test
            if q:
                f.write("    * **PASS**\n")
            else:
                f.write("    * **FAIL**\n")
            # Write targets
            f.write("    * Output: ``%i``\n" % ierr)
            f.write("    * Target: ``%i``\n" % rc_target)
        # Output
        return q
        
    # Check timer status
    def process_results_maxtime(self, i, t):
        """Check the maximum time for command *i*
        
        :Call:
            >>> q = testd.process_results_maxtime(i, t)
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
        # Convert to seconds
        tsec = testshell._time2sec(tmax)
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
        if tsec:
            # Check for expiration
            q = (ttot + t < tsec)
        else:
            # No test to fail
            q = True
        # Save status
        self.TestStatus_MaxTime.append(q)
        # Update overall status
        self.TestStatus = self.TestStatus and q
        # Get file handle
        f = self.frst
        # reST output
        if isinstance(f, file) and (not f.closed):
            # Return code section
            f.write(":Time Taken:\n")
            # Write status of the test
            if q:
                f.write("    * **PASS**\n")
            else:
                f.write("    * **FAIL**\n")
            # Write time taken
            f.write("    * Command took %4g seconds\n" % t)
            f.write("    * Cumulative time: %4g seconds\n" % (ttot + t))
            # Write constraint
            if tsec:
                f.write("    * Max allowed: %4g seconds (%s)\n" % (tsec, tmax))
        # Output
        return q
        
    # Check contents of STDOUT
    def process_results_stdout(self, i, fnout):
        """Compare STDOUT from command *i* to target
        
        :Call:
            >>> q = testd.process_results_stdout(i, fnout)
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
        # Individual tests on whether STDOUT and TargetSTDOUT exist
        qtarget = fntout and isinstance(fntout, (str, unicode))
        qactual = fnout  and isinstance(fnout,  (str, unicode))
        # Absolutize the path to *fntout*; usually in parent folder
        if qtarget and not os.path.isabs(fntout):
            # If relative, compare to parent
            fntout = os.path.join(os.path.realpath(".."), fntout)
        # Check if files exist
        qtarget = qtarget and os.path.isfile(fntout)
        qactual = qactual and os.path.isfile(fnout)
        # Perform test on STDOUT
        if not fntout:
            # No target: PASS
            q = True
        elif not fnout:
            # No STDOUT file: unacceptable if target
            q = False
        elif not qactual:
            # STDOUT not mapped to file: unacceptable if target
            q = False
        else:
            # Get options for file comparisons
            kw_comp = self.opts.get_FileComparisonOpts(i)
            # Compare STDOUT files
            q = fileutils.compare_files(fnout, fntout, **kw_comp)
        # Save result
        self.TestStatus_STDOUT[i] = q
        # Update overall status
        self.TestStatus = self.TestStatus and q
        # Get file handle
        f = self.frst
        # Check for file
        if not isinstance(f, file) or f.closed:
            # If no file, exit now
            return q
        # Indentation
        tab = "    "
        # reST settings
        show_out = self.opts.getel("ShowSTDOUT", i, vdef=None)
        link_out = self.opts.getel("LinkSTDOUT", i, vdef=False)
        show_trg = self.opts.getel("ShowTargetSTDOUT", i, vdef=True)
        link_trg = self.opts.getel("LinkTargetSTDOUT", i, vdef=False)
        # Get language for Lexer
        lang = self.opts.getel("LexerSTDOUT", i, vdef="none")
        # Return code section
        f.write(":STDOUT:\n")
        # Write status of the test
        if q:
            f.write("    * **PASS**\n")
        else:
            f.write("    * **FAIL**\n")
        # Show actual STDOUT
        if qactual and link_out:
            # Link file name
            flink = "STDOUT.%02i" % (i+1)
            # Copy the file
            shutil.copy(fnout, os.path.join(self.fdoc, flink))
            # Create the link
            f.write(tab)
            f.write("* Actual: :download:`%s`\n" % flink)
        elif qactual and (
                (show_out is None and not (q and show_trg)) or show_out):
            # Read it
            txt = open(fnout).read()
            # Check for content
            if len(txt) > 0:
                # Write header information
                f.write(tab + "* Actual:\n\n")
                # Use language
                f.write(tab + "  .. code-block:: %s\n\n" % lang)
                # Loop through lines
                for line in txt.split("\n"):
                    # Indent it 8 spaces
                    f.write(tab + tab + line + "\n")
                # Blank line
                f.write("\n")
            else:
                # Write empty actual
                f.write(tab + "* Actual: (empty)\n")
        # Show target STDOUT
        if qtarget and link_trg:
            # Link file name
            flink = "STDOUT-target.%02i" % (i+1)
            # Copy the file
            shutil.copy(fnout, os.path.join(self.fdoc, flink))
            # Create the link
            f.write(tab)
            f.write("* Target: :download:`%s`\n" % flink)
        elif qtarget and show_trg:
            # Read it
            txt = open(fntout).read()
            # Check for content
            if len(txt) > 0:
                # Write header information
                f.write(tab + "* Target:\n\n")
                # Use language
                f.write(tab + "  .. code-block:: %s\n\n" % lang)
                # Loop through lines
                for line in txt.split("\n"):
                    # Indent it 8 spaces
                    f.write(tab + tab + line + "\n")
                # Blank line
                f.write("\n")
            else:
                # Write empty actual
                f.write(tab + "* Target: (empty)\n")
        # Output
        return q

    # Check contents of STDERR
    def process_results_stderr(self, i, fnerr, fnout):
        """Compare STDERR from command *i* to target
        
        :Call:
            >>> q = testd.process_results_stderr(i, fnerr, fnout)
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
        # Individual tests on whether STDOUT and TargetSTDOUT exist
        qtarget = fnterr and isinstance(fnterr, (str, unicode))
        qactual = fnerr  and isinstance(fnerr,  (str, unicode))
        # Absolutize the path to *fntout*; usually in parent folder
        if qtarget and not os.path.isabs(fnterr):
            # If relative, assume it's in parent folder
            fnterr = os.path.join(os.path.realpath(".."), fnterr)
        # Check if files exist
        qtarget = qtarget and os.path.isfile(fnterr)
        qactual = qactual and os.path.isfile(fnerr)
        # Perform test on STDERR
        if not fnterr:
            # No target: check for actual
            if not fnerr:
                # Neither
                q = True
            elif not isinstance(fnerr, (str, unicode)):
                # STDERR not mapped to file: unknowable
                q = True
            elif (fnerr == fnout):
                # STDERR mapped to STDOUT file; test elsewhere
                q = True
            elif os.path.isfile(fnerr):
                # Actual reported; check if it's empty
                q = os.path.getsize(fnerr) == 0
        elif not fnerr:
            # No STDERR file: unacceptable if target
            q = False
        elif not qactual:
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
        # Update overall status
        self.TestStatus = self.TestStatus and q
        # Get file handle
        f = self.frst
        # Check for a log file
        if not isinstance(f, file) or f.closed:
            # Early output
            return q
        # Indentation
        tab = "    "
        # reST settings
        show_out = self.opts.getel("ShowSTDERR", i, vdef=None)
        link_out = self.opts.getel("LinkSTDERR", i, vdef=False)
        show_trg = self.opts.getel("ShowTargetSTDERR", i, vdef=True)
        link_trg = self.opts.getel("LinkTargetSTDERR", i, vdef=False)
        # Get language for Lexer
        lang = self.opts.getel("LexerSTDERR", i, vdef="none")
        # Return code section
        f.write(":STDERR:\n")
        # Write status of the test
        if q:
            f.write("    * **PASS**\n")
        else:
            f.write("    * **FAIL**\n")
        # Show actual STDERR
        if qactual and link_out:
            # Link file name
            flink = "STDERR.%02i" % (i+1)
            # Copy the file
            shutil.copy(fnout, os.path.join(self.fdoc, flink))
            # Create the link
            f.write(tab)
            f.write("* Actual: :download:`%s`\n" % flink)
        elif qactual and (
                (show_out is None and not (q and show_trg)) or show_out):
            # Read it
            txt = open(fnerr).read()
            # Check for content
            if len(txt) > 0:
                # Write header information
                f.write(tab + "* Actual:\n\n")
                # Use language
                f.write(tab + "  .. code-block:: %s\n\n" % lang)
                # Loop through lines
                for line in txt.split("\n"):
                    # Indent it 8 spaces
                    f.write(tab + tab + line + "\n")
                # Blank line
                f.write("\n")
            else:
                # Write empty actual
                f.write(tab + "* Actual: (empty)\n")
        # Show target STDERR
        if qtarget and link_trg:
            # Link file name
            flink = "STDERR-target.%02i" % (i+1)
            # Copy the file
            shutil.copy(fnout, os.path.join(self.fdoc, flink))
            # Create the link
            f.write(tab)
            f.write("* Target: :download:`%s`\n" % flink)
        elif qtarget and show_trg:
            # Read it
            txt = open(fnterr).read()
            # Check for content
            if len(txt) > 0:
                # Write header information
                f.write(tab + "* Target:\n\n")
                # Use language
                f.write(tab + "  .. code-block:: %s\n\n" % lang)
                # Loop through lines
                for line in txt.split("\n"):
                    # Indent it 8 spaces
                    f.write(tab + tab + line + "\n")
                # Blank line
                f.write("\n")
            else:
                # Write empty actual
                f.write(tab + "* Actual: (empty)\n")
        # End section
        f.write("\n")
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
    testd = TestDriver(**kw)
    # Run the crawler
    return testd.run()

