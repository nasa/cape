#!/usr/bin/python
"""
:mode:`cape.test`: pyCart Testing Module
========================================

This module provides some useful methods for testing pyCart, pyFun, and pyOver
capabilities.
"""

# System interface
import os, sys

# Function to call :func:`os.system` and write error message
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
            # Error message provided
            f.write(msg + '\n')
        else:
            # Default error message
            f.write("Data book value did not match expected result.\n")
        # Get out of here
        f.close()
        sys.exit(1)
        
# Test a list


