#!/usr/bin/python
"""
:mode:`cape.test`: CAPE Testing Module
========================================

This module provides some useful methods for testing pyCart, pyFun, and pyOver
capabilities.
"""

# System interface
import os, sys
import subprocess as sp
# Text tools
import re

# Paths
fcape = os.path.split(os.path.abspath(__file__))[0]
froot = os.path.split(fcape)[0]
ftest = os.path.join(froot, 'test')
# Module folders
fpycart = os.path.join(froot, 'pyCart')
fpyfun  = os.path.join(froot, 'pyFun')
fpyover = os.path.join(froot, 'pyOver')
# Module test folders
ftcape   = os.path.join(ftest, 'cape')
ftpycart = os.path.join(ftest, 'pycart')
ftpyfun  = os.path.join(ftest, 'pyfun')
ftpyover = os.path.join(ftest, 'pyover')

# Sphinx folders
fdoc   = os.path.join(froot, "doc")
fbuild = os.path.join(fdoc,  "_build")
# Doctest folder
fdoctest = os.path.join(fbuild, "doctest")


# Wrapper for :func:`os.system` to write error message and exit on fail
def callt(cmd, msg):
    """Call a message while checking the exit status
    
    :Call:
        >>> callt(cmd, msg)
    :Inputs:
        *cmd*: :class:`str`
            Command to run with :func:`os.system`
        *msg*: :class:`str`
            Error message to show on failure
    :Versions:
        * 2017-03-06 ``@ddalle``: First version
    """
    # Status update
    print("\n$ %s" % cmd)
    # Flush to update
    sys.stdout.flush()
    # Run the command
    ierr = os.system(cmd)
    # Check for failure
    if ierr:
        # Open a FAIL file to write error message
        f = open('FAIL', 'w')
        # Write the error message
        f.write(msg + '\n')
        # Get out of here
        f.close()
        sys.exit(ierr)
        
# Simple shell call with status
def shell(cmd, msg=None):
    """Call a command while checking the exit status
    
    The command writes ``$`` and the name of the command, *cmd* to the file
    ``test.out`` within the current working directory.  This is followed the
    contents of both STDOUT and STDERR.
    
    If the command exists with a status other than ``0``, this function raises
    an exception, writing a short message input by the user (if given) to
    ``test.out``.
    
    :Call:
        >>> shell(cmd, msg)
    :Inputs:
        *cmd*: :class:`str`
            Command to run with :func:`os.system`
        *msg*: {``None``} | :class:`str`
            Error message to show on failure
    :Versions:
        * 2018-03-17 ``@ddalle``: First version
    """
    # Open status file
    f = open("test.out", "a")
    # Status update
    f.write("\n$ %s\n" % cmd)
    # Call the command
    ierr = sp.call(cmd, shell=True, stdout=f, stderr=f)
    # Check for failure
    if ierr:
        # Check if message
        if msg is None:
            # No message
            f.write("FAILED\n")
        else:
            # Add a message
            f.write("FAILED: %s\n" % msg)
        # Close the file
        f.close()
        # Raise an exception
        raise ValueError("See status in 'test.out' file")
    else:
        # Close file
        f.close()
        
# Simple shell call with status and capturing output
def check_output(cmd, msg=None):
    """Call a command and capture the contents of STDOUT
    
    The command writes ``$`` and the name of the command, *cmd* to the file
    ``test.out`` within the current working directory.  This is followed the
    contents of both STDOUT and STDERR.
    
    If the command exists with a status other than ``0``, this function raises
    an exception, writing a short message input by the user (if given) to
    ``test.out``.
    
    :Call:
        >>> txt = check_output(cmd, msg)
    :Inputs:
        *cmd*: :class:`str`
            Command to run with :func:`os.system`
        *msg*: {``None``} | :class:`str`
            Error message to show on failure
    :Outputs:
        *txt*: :class:`str`
            Contents of STDOUT
    :Versions:
        * 2018-03-17 ``@ddalle``: First version
    """
    # Open status file
    f = open("test.out", "a")
    # Status update
    f.write("\n$ %s\n" % cmd)
    # Call the command
    txt = sp.check_output(cmd, shell=True, stderr=f)
    # Close file
    f.close()
    # Output
    return txt

# Wrapper for :func:`os.system` that raises exception on failure
def calle(cmd, msg):
    """Call a message while checking the exit status
    
    :Call:
        >>> callt(cmd, msg)
    :Inputs:
        *cmd*: :class:`str`
            Command to run with :func:`os.system`
        *msg*: :class:`str`
            Error message to show on failure
    :Versions:
        * 2017-03-06 ``@ddalle``: First version
    """
    # Status update
    print("$ %s" % cmd)
    # Run the command
    ierr = os.system(cmd)
    # Check for failure
    if ierr:
        # Raise exception
        raise ValueError(msg)
        
# Test a value
def test_val(val, targ, tol, msg=None):
    """Test a value with a target and a tolerance
    
    :Call:
        >>> test_val(val, targ, tol, msg)
    :Inputs:
        *val*: :class:`float`
            Actual value
        *targ*: :class:`float`
            Target value
        *tol*: :class:`float`
            Acceptable tolerance (using <= as test)
        *msg*: {``None``} | :class:`str`
            Error message to show on failure
    :Versions:
        * 2017-03-06 ``@ddalle``: First version
    """
    # Status update
    print("\n> abs(%s - %s) <= %s" % (val, targ, tol))
    os.sys.stdout.flush()
    # Test the value
    if abs(val - targ) > tol:
        # Open a FAIL file to write error message
        f = open('FAIL', 'w')
        # Write the error message
        if msg is None:
            # Default error message
            f.write("Data book value did not match expected result.\n")
        else:
            # Error message provided
            f.write(msg + '\n')
        # Get out of here
        f.close()
        sys.exit(1)
        
# Convert a results.txt file
def convert_results():
    """Convert the ``results.txt`` file created by ``make doctest`` to reST
    
    :Call:
        >>> convert_results()
    :Versions:
        * 2018-03-27 ``@ddalle``: First version
    """
    # Get name of file
    ftxt = os.path.join(fdoctest, "results.txt")
    # Check if file exists
    if not os.path.isfile(ftxt):
        print("  No doctest file 'results.txt' file to interpret!")
        return
    # Converted file
    frst = os.path.join(fdoc, "test", "results.rst")
    # Open input and create output
    fi = open(ftxt, 'r')
    fo = open(frst, 'o')
    # Start output file
    fo.write("\n.. _test-results:\n\n")
    fo.write("Results of doctest builder\n")
    fo.write("==========================\n\n")
    # Read the input file
    lines = fi.readlines()
    # Close input file
    fi.close()
    # Split to dates
    tdate, ttime = lines[0].strip().split()[-2:]
    # Write compiled date and time
    fo.write("This test was run on %s at time %s.\n\n" % (tdate, ttime))
    # Write the summary
    fo.write("**Summary**:\n")
    # Write the last four lines (not counting empty line)
    for i in range(-4,0):
        # Write with a bullet
        fo.write("    * %s" % lines[i])
        
    # Start looping through documents
    i = 3
    # Test following line
    while lines[i].startswith("Document"):
        # Process the line
        i = convert_test_document(lines, fo, i)
    
    # Close output file
    fo.close()

# Interpret one result
def convert_test_document(lines, fo, i):
    """Convert the results of one document test
    
    :Call:
        >>> j = convert_test_document(lines, fo, i)
    :Inputs:
        *lines*: :class:`list` (:class:`str`)
            Lines from input file
        *fo*: :class:`file`
            Output file of interpreted reST, open ``'w'``
        *i*: :class:`int`
            Index of starting line for this document
    :Outputs:
        *j*: :class:`int`
            Index of starting line for next content
    :Versions:
        * 2018-03-27 ``@ddalle``: First version
    """
    # Get the name of the doucment
    txt, fdoc = lines[i].strip().split()
    # Make sure this is a document test
    if not txt.startswith("Document:"): return
    # Create section header
    txt = "Document: :doc:`%s </%s>`" % (fdoc, fdoc)
    # Write it
    fo.write(txt + "\n")
    fo.write("-"*len(txt) + "\n")
    # Discard header underlines
    i += 2
    # Loop through failed tests
    while lines[i].startswith("******"):
        # Read the next line
        i += 1
        # Check for summary line (dealing with extra '*******' line)
        if re.match("[0-9]+ items", lines[i]):
            break
        # Write failure line summary (and extra blank line)
        fo.write("Failure in f%s\n" % lines[i][1:])
        # Go to next line
        i += 1
        # Loop until next '******' line found
        while not lines[i].startswith("******"):
            # Read key (strip colon at end)
            hdr = lines[i].strip()[:-1]
            # Write the header
            fo.write("**%s**:\n\n" % hdr)
            # Check type
            if hdr == "Failed example":
                # Write a Python code block
                fo.write("    .. code-block:: python\n\n")
            elif hdr == "Exception raised":
                # Write a Python Traceback code block
                fo.write("    .. code-block:: pytb\n\n")
            else:
                # Write raw text
                fo.write("    .. code-block:: none\n\n")
            # Loop until unindent
            while lines[i+1].startswith("    "):
                # Go to next line
                i += 1
                # Write the content
                fo.write("    %s" % lines[i])
            # Clean up
            fo.write("\n")
            # Go to next line
            i += 1
    # Process summary
    fo.write("Document summary:\n\n")
    # Set previous indent
    indent = 0
    # Loop through lines
    while lines[i].strip():
        # Check if line has a summary
        if lines[i].startswith("    "):
            # Indent
            indent = 4
            # Write line
            fo.write("    - %s" % lines[i].lstrip())
        else:
            # Check for previous indent
            if indent == 4:
                fo.write("\n")
            # Write line
            fo.write("  * %s" % lines[i])
        # Move to next line
        i += 1
    # End; skip blank line
    return i + 1
        
    
