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


# Local relative imports
from .verutils import ver


# Create :class:`unicode` class for Python 3+
if ver > 2:
    unicode = str


# Convert time string to seconds
def _time2sec(t):
    """Convert a time :class:`float` or :class:`str` to seconds
    
    :Call:
        >>> s = _time2sec(t)
        >>> s = _time2sec("%ss" % tsec)
        >>> s = _time2sec("%sm" % tmin)
        >>> s = _time2sec("%sh" % thr)
        >>> s = _time2sec("%sd" % tday)
    :Inputs:
        *t*: ``None`` | :class:`float`
            Input time in seconds
        *tsec*: :class:`int` | :class:`float`
            Number of seconds
        *tmin*: :class:`int` | :class:`float`
            Number of minutes
        *thr*: :class:`int` | :class:`float`
            Number of hours
        *tday*: :class:`int` | :class:`float`
            Number of days
    :Outputs:
        *s*: ``None`` | :class:`float`
            Output time converted to seconds
    :Versions:
        * 2019-07-02 ``@ddalle``: First version
    """
    # Check for false-like input
    if not t:
        return None
    # Check for number
    if isinstance(t, (float, int)):
        # Return seconds as float
        return float(t)
    elif isinstance(t, (str, unicode)):
        # Try to convert to seconds
        try:
            return float(t)
        except Exception:
            pass
        # Try to get last character
        unit = t[-1]
        # Preceding characters are the amount
        stime = t[:-1]
        # Check case
        if unit == "s":
            # Seconds
            scale = 1.0
        elif unit == "m":
            # Minutes
            scale = 60.0
        elif unit == "h":
            # Hours
            scale = 3600.0
        elif unit == "d":
            # Days
            scale = 86400.0
        else:
            # Bad input
            raise ValueError(
                "Unable to determine units in time string '%s'" % t)
        # If empty, return 0
        if len(t) == 1:
            return None
        # Try to convert
        try:
            return float(t[:-1]) * scale
        except Exception:
            raise ValueError(
                "Unable to evaluate numerical part of time string '%s'" % t)
    else:
        # Bad input type
        raise TypeError("Time input must be None, float, or str")


# Call a function and return status, STDOUT, and STDERR
def comm(cmd, maxtime=None, dt=None, stdout=sp.PIPE, stderr=sp.PIPE):
    """Call a subprocess and return status, STDOUT and STDERR
    
    :Call:
        >>> t, ierr, out, err = comm(cmd, maxtime=None, dt=None
            stdout=sp.PIPE, stderr=sp.PIPE)
    :Inputs:
        *cmd*: :class:`list`\ [:class:`str`]
            List of strings to be passed to :func:`sp.call`
        *maxtime*: {``None``} | :class:`float` | :class:`str`
            Optional maximum time allowed for process
        *dt*: {``None``}  | :class:`float` | :class:`str`
            Sleep time to wait between checking for completion if
            *maxtime* is used; default is 1/1000th of *maxtime*
        *stdout*: {``sp.PIPE``} | ``None`` | :class:`file`
            Optional file to contain STDOUT
        *stderr*: {``sp.PIPE``} | ``None`` | :class:`file`
            Optional file to contain STDERR
    :Ouputs:
        *t*: :class:`float`
            Time taken to run process in seconds
        *ierr*: :class:`int` >= 0
            Command exit status
        *out*: :class:`str`
            String containing any STDOUT from command
        *err*: :class:`str`
            String containing any STDERR from command
    :Versions:
        * 2019-07-01 ``@ddalle``: First version
        * 2019-07-02 ``@ddalle``: Added *maxtime*
    """
    # Process maximum time to seconds
    tmax = _time2sec(maxtime)
    tstp = _time2sec(dt)
    # Default time step
    if tmax and (not tstp):
        tstp = max(0.001, 0.001*tmax)
    # Start timer
    tic = time.time()
    # Create a Popen process (starts running immediately)
    proc = sp.Popen(cmd, stdout=stdout, stderr=stderr)
    # Loop through timer
    while tmax:
        # Check time
        if time.time() - tic >= tmax:
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
        time.sleep(tstp)
    # Run the command
    out, err = proc.communicate()
    # Update *returncode*
    ierr = proc.poll()
    # Get total time
    toc = time.time()
    # Output
    return toc - tic, ierr, out, err


# Call a function and return error code
def call(cmd, maxtime=None, dt=None, stdout=None, stderr=None):
    """Call a subprocess and return the error code
    
    Neither *stdout* nor *stderr* may be ``sp.PIPE``.  Since they are
    not returned by this function, they cannot be captured.  They can
    be either printed as in normal operations or redirected to an open
    file.  Redirecting both to the same file is allowed.
    
    :Call:
        >>> ierr = call(cmd, maxtime=None, dt=None,
            stdout=None, stderr=None)
    :Inputs:
        *cmd*: :class:`list`\ [:class:`str`]
            List of strings to be passed to :func:`sp.call`
        *maxtime*: {``None``} | :class:`float` | :class:`str`
            Optional maximum time allowed for process
        *dt*: {``None``}  | :class:`float` | :class:`str`
            Sleep time to wait between checking for completion if
            *maxtime* is used; default is 1/1000th of *maxtime*
        *stdout*: {``None``} | :class:`file`
            Optional file to contain STDOUT
        *stderr*: {``None``} | :class:`file`
            Optional file to contain STDERR
    :Ouputs:
        *ierr*: :class:`int` >= 0
            Command exit status
    :Versions:
        * 2019-07-01 ``@ddalle``: First version
        * 2019-07-02 ``@ddalle``: Dependency on :func:`comm`
    """
    # Check for disallowed PIPE
    if stdout == sp.PIPE:
        raise ValueError("STDOUT cannot be captured using call()")
    # Check for disallowed PIPE
    if stderr == sp.PIPE:
        raise ValueError("STDERR cannot be captured using call()")
    # Call the main function
    _, ierr, _, _ = comm(cmd, maxtime, dt, stdout, stderr)
    # Output
    return ierr
    

# Check status
def check_call(cmd, maxtime=None, dt=None, stdout=None, stderr=None):
    """Call a subprocess and raise an exception if has an error
    
    Neither *stdout* nor *stderr* may be ``sp.PIPE``.  Since they are
    not returned by this function, they cannot be captured.  They can
    be either printed as in normal operations or redirected to an open
    file.  Redirecting both to the same file is allowed.
    
    :Call:
        >>> ierr = check_call(cmd, maxtime=None, dt=None,
            stdout=None, stderr=None)
    :Inputs:
        *cmd*: :class:`list`\ [:class:`str`]
            List of strings to be passed to :func:`sp.call`
        *maxtime*: {``None``} | :class:`float` | :class:`str`
            Optional maximum time allowed for process
        *dt*: {``None``}  | :class:`float` | :class:`str`
            Sleep time to wait between checking for completion if
            *maxtime* is used; default is 1/1000th of *maxtime*
        *stdout*: {``None``} | :class:`file`
            Optional file to contain STDOUT
        *stderr*: {``None``} | :class:`file`
            Optional file to contain STDERR
    :Ouputs:
        *ierr*: ``0``
            Command exit status
    :Versions:
        * 2019-07-01 ``@ddalle``: First version
        * 2019-07-02 ``@ddalle``: Dependency on :func:`comm`
    """
    # Check for disallowed PIPE
    if stdout == sp.PIPE:
        raise ValueError("STDOUT cannot be captured using call()")
    # Check for disallowed PIPE
    if stderr == sp.PIPE:
        raise ValueError("STDERR cannot be captured using call()")
    # Call the main function
    _, ierr, _, _ = comm(cmd, maxtime, dt, stdout, stderr)
    # Check for error code
    if ierr:
        raise sp.CalledProcessError(ierr, " ".join(cmd))
    # Output
    return ierr
    

# Check status after generic call
def check_comm(cmd, maxtime=None, dt=None, stdout=sp.PIPE, stderr=sp.PIPE):
    """Call a subprocess and check for errors
    
    :Call:
        >>> t, ierr, out, err = check_comm(cmd, maxtime=None, dt=None,
            stdout=None, stderr=None)
    :Inputs:
        *cmd*: :class:`list`\ [:class:`str`]
            List of strings to be passed to :func:`sp.call`
        *maxtime*: {``None``} | :class:`float` | :class:`str`
            Optional maximum time allowed for process
        *dt*: {``None``}  | :class:`float` | :class:`str`
            Sleep time to wait between checking for completion if
            *maxtime* is used; default is 1/1000th of *maxtime*
        *stdout*: {``sp.PIPE``} | ``None`` | :class:`file`
            Optional file to contain STDOUT
        *stderr*: {``sp.PIPE``} | ``None`` | :class:`file`
            Optional file to contain STDERR
    :Ouputs:
        *t*: :class:`float`
            Time taken to run process in seconds
        *ierr*: ``0``
            Command exit status
        *out*: :class:`str`
            String containing any STDOUT from command
        *err*: :class:`str`
            String containing any STDERR from command
    :Versions:
        * 2019-07-02 ``@ddalle``: First version
    """
    # Call the main function
    t, ierr, out, err = comm(cmd, maxtime, dt, stdout, stderr)
    # Check for error code
    if ierr:
        raise sp.CalledProcessError(ierr, " ".join(cmd))
    # Output
    return t, ierr, out, err


# Capture output and check for errors
def check_output(cmd, maxtime=None, dt=None, stderr=sp.PIPE):
    """Call a subprocess, check for errors, and return STDOUT
    
    :Call:
        >>> out = check_output(cmd, maxtime=None, dt=None, stderr=None)
    :Inputs:
        *cmd*: :class:`list`\ [:class:`str`]
            List of strings to be passed to :func:`sp.call`
        *maxtime*: {``None``} | :class:`float` | :class:`str`
            Optional maximum time allowed for process
        *dt*: {``None``}  | :class:`float` | :class:`str`
            Sleep time to wait between checking for completion if
            *maxtime* is used; default is 1/1000th of *maxtime*
        *stderr*: {``sp.PIPE``} | ``None`` | :class:`file`
            Optional file to contain STDERR
    :Ouputs:
        *out*: :class:`str`
            String containing any STDOUT from command
    :Versions:
        * 2019-07-02 ``@ddalle``: First version
    """
    # Call the main function
    t, ierr, out, err = comm(cmd, maxtime, dt, sp.PIPE, stderr)
    # Check for error code
    if ierr:
        raise sp.CalledProcessError(ierr, " ".join(cmd))
    # Output
    return out
