#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`cape.testutils.testcall`: Shell utilities for CAPE testing system
=======================================================================

This module contains various methods to perform system calls using the
:mod:`subprocess` module.
"""

# Standard library modules
import time
import subprocess as sp


# Call a function and return error code
def call(cmd, stdout=None, stderr=None):
    """Call a subprocess and return the error code
    
    Neither *stdout* nor *stderr* may be ``sp.PIPE``.  Since they are
    not returned by this function, they cannot be captured.  They can
    be either printed as in normal operations or redirected to an open
    file.  Redirecting both to the same file is allowed.
    
    :Call:
        >>> ierr = call(cmd, stdout=None, stderr=None)
    :Inputs:
        *cmd*: :class:`list`\ [:class:`str`]
            List of strings to be passed to :func:`sp.call`
        *stdout*: {``None``} | :class:`file`
            Optional file to contain STDOUT
        *stderr*: {``None``} | :class:`file`
            Optional file to contain STDERR
    :Ouputs:
        *ierr*: :class:`int` >= 0
            Command exit status
    :Versions:
        * 2019-07-01 ``@ddalle``: First version
    """
    # Check for disallowed PIPE
    if stdout == sp.PIPE:
        raise ValueError("STDOUT cannot be captured using call()")
    # Check for disallowed PIPE
    if stderr == sp.PIPE:
        raise ValueError("STDERR cannot be captured using call()")
    # Create a Popen process
    proc = sp.Popen(cmd, stdout=stdout, stderr=stderr)
    # Run the command
    out, err = proc.communicate()
    # Update *returncode*
    proc.poll()
    # Output
    return proc.returncode
    

# Call a function and return status, STDOUT, and STDERR
def comm(cmd, stdout=sp.PIPE, stderr=sp.PIPE):
    """Call a subprocess and return status, STDOUT and STDERR
    
    :Call:
        >>> ierr, out, err = comm(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    :Inputs:
        *cmd*: :class:`list`\ [:class:`str`]
            List of strings to be passed to :func:`sp.call`
        *stdout*: {``sp.PIPE``} | ``None`` | :class:`file`
            Optional file to contain STDOUT
        *stderr*: {``sp.PIPE``} | ``None`` | :class:`file`
            Optional file to contain STDERR
    :Ouputs:
        *ierr*: :class:`int` >= 0
            Command exit status
        *out*: :class:`str`
            String containing any STDOUT from command
        *err*: :class:`str`
            String containing any STDERR from command
    :Versions:
        * 2019-07-01 ``@ddalle``: First version
    """
    # Create a Popen process
    proc = sp.Popen(cmd, stdout=stdout, stderr=stderr)
    # Run the command
    out, err = proc.communicate()
    # Update *returncode*
    proc.poll()
    # Output
    return proc.returncode, out, err
    

# Run a process with a timeout
def timeout_call(cmd, maxtime=None, stdout=None, stderr=None, dt=None):
    
    # Check for disallowed PIPE
    if stdout == sp.PIPE:
        raise ValueError("STDOUT cannot be captured using call()")
    # Check for disallowed PIPE
    if stderr == sp.PIPE:
        raise ValueError("STDERR cannot be captured using call()")
    # Default time step
    if maxtime and (dt is None):
        dt = max(0.001, 0.001*maxtime)
    # Start timer
    tic = time.time()
    # Create a Popen process (starts running immediately)
    proc = sp.Popen(cmd, stdout=stdout, stderr=stderr)
    # Loop through timer
    while maxtime:
        # Check time
        if time.time() - tic >= maxtime:
            # Kill the process
            proc.kill()
            # Exit loop
            break
        # Check if process has ended
        ierr = proc.poll()
        # If still running, this will be ``None``
        if ierr is not None:
            break
        # Pause *dt* seconds before trying again
        time.sleep(dt)
    # Run the command
    out, err = proc.communicate()
    # Output
    return proc.poll()
    

# Check status
def check_call(cmd, stdout=None, stderr=None):
    """Call a subprocess and return the error code
    
    Neither *stdout* nor *stderr* may be ``sp.PIPE``.  Since they are
    not returned by this function, they cannot be captured.  They can
    be either printed as in normal operations or redirected to an open
    file.  Redirecting both to the same file is allowed.
    
    :Call:
        >>> ierr = call(cmd, stdout=None, stderr=None)
    :Inputs:
        *cmd*: :class:`list`\ [:class:`str`]
            List of strings to be passed to :func:`sp.call`
        *stdout*: {``None``} | :class:`file`
            Optional file to contain STDOUT
        *stderr*: {``None``} | :class:`file`
            Optional file to contain STDERR
    :Ouputs:
        *ierr*: ``0``
            Command exit status
    :Versions:
        * 2019-07-02 ``@ddalle``: First version
    """
    # Check for disallowed PIPE
    if stdout == sp.PIPE:
        raise ValueError("STDOUT cannot be captured using call()")
    # Check for disallowed PIPE
    if stderr == sp.PIPE:
        raise ValueError("STDERR cannot be captured using call()")
    # Create a Popen process
    proc = sp.Popen(cmd, stdout=stdout, stderr=stderr)
    # Run the command
    out, err = proc.communicate()
    # Update *returncode*
    proc.poll()
    # Check for error code
    if proc.returncode:
        raise sp.CalledProcessError(
            "Command failed with status %i'" % proc.returncode)
    # Output
    return proc.returncode
    