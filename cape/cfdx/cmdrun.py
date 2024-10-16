r"""
``cape.cfdx.cmdrun``: Execute system calls for CAPE
======================================================

This template module provides an interface for simple command-line
tools. The general approach for CAPE is to create a function for each
command-line binary that is called. This module contains two methods,
:func:`calli` and :func:`callf`, that are wrappers for the built-in
:func:`subprocess.call`, and several useful command-line utilities.

See also:
    * :mod:`cape.cfdx.cmdgen`
"""

# File system and operating system management
import os
import sys
import subprocess as sp

# Import local command-generating module for complex commands
from . import cmdgen


# Imitate sp.check_output() for older versions
def check_output(cmdi):
    r"""Capture output from a system command

    :Call:
        >>> txt = check_output(cmdi)
    :Inputs:
        *cmdi*: :class:`list`\ [:class:`str`]
            List of strings as for :func:`subprocess.call`
    :Outputs:
        *txt*: :class:`str`
            Contents of STDOUT while executing *cmdi*
    :Versions:
        * 2017-03-13 ``@ddalle``: v1.0
    """
    # Call sp.Popen
    out = sp.Popen(cmdi, stdout=sp.PIPE).communicate()
    # Output
    return out[0]


# Function to call commands with a different STDOUT
def calli(cmdi, f=None, e=None, shell=None, v=True):
    r"""Call a command with alternate STDOUT by filename

    :Call:
        >>> ierr = calli(cmdi, f=None, e=None, shell=None, v=True)
    :Inputs:
        *cmdi*: :class:`list`\ [:class:`str`]
            List of strings as for :func:`subprocess.call`
        *f*: :class:`str`
            File name to which to store STDOUT
        *e*: {*f*} | :class:`str`
            Name of separate file to write STDERR to
        *shell*: :class:`bool`
            Whether or not a shell is needed
        *v*: {``True``} | :class:`False`
            Verbose option; display *PWD* and *STDOUT* values
    :Outputs:
        *ierr*: :class:`int`
            Return code, ``0`` for successful execution
    :Versions:
        * 2014-08-30 ``@ddalle``: v1.0
        * 2015-02-13 ``@ddalle``: v2.0; return code
        * 2017-03-12 ``@ddalle``: v2.1; Add *v* option
        * 2019-06-10 ``@ddalle``: v2.2; Add *e* option
        * 2024-01-17 ``@ddalle``: v2.3; flush() STDOUT manually
    """
    # Process the shell option
    shell = bool(shell)
    # String version of command
    if isinstance(cmdi, str):
        # Already str
        cmdstr = cmdi
    else:
        # Combine list
        cmdstr = " ".join(cmdi)
    # Print the current location.
    if v:
        # Print the command
        print(" > " + cmdstr)
        # Get abbreviated path
        cwd_parts = os.getcwd().split(os.sep)
        # Just use the last two folders
        cwd = os.sep.join(cwd_parts[-2:]) + os.sep
        # Print the abbreviated path
        print("     (PWD = '%s')" % cwd)
        sys.stdout.flush()
    # Check for an output
    if f:
        # Print the location of STDOUT
        if v:
            print("     (STDOUT = '%s')" % os.path.basename(f))
            sys.stdout.flush()
        # Print the location of STDERR
        if v and (e is not None):
            print("     (STDERR = '%s')" % os.path.basename(e))
            sys.stdout.flush()
        # Open the files for STDOUT and STDERR
        fid = open(f, 'w')
        # Check for separate STDERR file
        if e is None:
            # Use STDOUT file
            fe = fid
        else:
            # Open separate file
            fe = open(e, 'w')
        # Call the command
        try:
            ierr = sp.call(cmdi, stdout=fid, stderr=fe, shell=shell)
        except FileNotFoundError:
            # Process not found; give an error code but don't raise
            ierr = 2
        # Close STDOUT
        fid.close()
        # Close STDERR
        if fe is not fid:
            fe.close()
    else:
        # Call the command.
        ierr = sp.call(cmdi, shell=shell)
    # Output
    return ierr


# Function to call commands with a different STDOUT
def callf(cmdi, f=None, e=None, shell=None, v=True, check=True):
    r"""Call a command with alternate STDOUT by filename

    :Call:
        >>> callf(cmdi, f=None, e=None, shell=None, v=True, check=True)
    :Inputs:
        *cmdi*: :class:`list` (:class:`str`)
            List of strings as for :func:`subprocess.call`
        *f*: :class:`str`
            File name to which to store STDOUT
        *e*: {*f*} | :class:`str`
            Separate file name for STDERR
        *shell*: :class:`bool`
            Whether or not a shell is needed
        *v*: {``True``} | :class:`False`
            Verbose option; display *PWD* and *STDOUT* values
    :Versions:
        * 2014-08-30 ``@ddalle``: v1.0
        * 2015-02-13 ``@ddalle``: v2.0; rely on :func:`calli`
        * 2017-03-12 ``@ddalle``: v2.1; add *v* option
        * 2019-06-10 ``@ddalle``: v2.2; add *e* option
        * 2024-05-25 ``@ddalle``: v2.3; don't remove RUNNING
    """
    # Call the command with output status
    ierr = calli(cmdi, f, e, shell, v=v)
    # Check the status.
    if ierr and check:
        # Exit with error notifier
        raise SystemError("Command failed with status %i." % ierr)
    # Otherwise return it
    return ierr


# Call command with output (since sp.check_output is Python 2.7+)
def callo(cmdi, shell=False):
    r"""Call a command and get the output text

    This function is basically a substitute for
    :func:`subprocess.check_output`, which was not available in
    Python 2.6.

    :Call:
        >>> txt = callo(cmdi, shell=False)
    :Inputs:
        *cmdi*: :class:`list`
            List of strings to use as a command
        *shell*: :class:`bool`
            Whether or not a shell is needed
    :Outputs:
        *txt*: :class:`str`
            Output of running the command
    :Versions:
        * 2016-04-01 ``@ddalle``: v1.0
    """
    # Call command and get output
    txt = sp.Popen(cmdi, stdout=sp.PIPE, shell=shell).communicate()[0]
    # Convert to unicode text
    return txt.decode("utf-8")


# Grep lines from a file
def grep(regex, fname):
    r"""Search for a regular expression in a file

    :Call:
        >>> lines = grep(regex, fname)
    :Inputs:
        *regex*: :class:`str`
            Regular expression for which to search
        *fname*: :class:`str`
            Name of file or wild card to search
    :Outputs:
        *lines*: :class:`list` (:class:`str`)
            List of lines containing the sought regular expression
    :Versions:
        * 2015-12-28 ``@ddalle``: v1.0
    """
    # Safely call
    try:
        # Call egrep so that regular expressions are expanded fully
        txt = callo(['egrep', str(regex), fname])
        # Split into list of lines
        return txt.split('\n')
    except Exception:
        return []


# Function to get the first line of a file.
def head(fname, n=1):
    r"""Extract the first *n* lines of a file

    :Call:
        >>> txt = head(fname, n=1)
    :Inputs:
        *fname*: :class:`str`
            Name of file to tail
        *n*: :class:`int`
            Number of lines to process
    :Outputs:
        *txt*: :class:`str`
            Output of built-in `tail` function
    :Versions:
        * 2015-01-12 ``@ddalle``: v1.0
    """
    # Create the command.
    cmdi = ['head', '-%i' % n, fname]
    # Use Popen because check_output is 2.7+
    return callo(cmdi)


# Function to get the last line of a file.
def tail(fname, n=1):
    r"""Tail the last *n* lines of a file

    :Call:
        >>> txt = tail(fname, n=1)
    :Inputs:
        *fname*: :class:`str`
            Name of file to tail
        *n*: :class:`int`
            Number of lines to process
    :Outputs:
        *txt*: :class:`str`
            Output of built-in `tail` function
    :Versions:
        * 2015-01-12 ``@ddalle``: v1.0
    """
    # Create the command.
    cmdi = ['tail', '-%i' % n, fname]
    # Use Popen because check_output is 2.7+
    return callo(cmdi)


# Simple function to make sure a file is present
def _assertfile(fname):
    r"""Assert that a given file exists or raise an exception

    :Call:
        >>> _assertfile(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file to test
    :Versions:
        * 2014-06-30 ``@ddalle``: v1.0
    """
    # Check for the file.
    if not os.path.isfile(fname):
        raise IOError("No input file '%s' found." % fname)


# Function to automate minor changes to docstrings to make them pyCart.Cntl
def _upgradeDocString(doccmd):
    r"""Upgrade docstrings from the :mod:`cape.pycart.bin` class

    :Call:
        >>> docbin = _upgradDocString(doccmd)
    :Inputs:
        *doccmd*: :class:`str`
            Docstring from method from :mod:`cape.pycart.cmd`
    :Outputs:
        *docbin*: :class:`str`
            Docstring for :mod:`cape.pycart.bin`
    :Versions:
        * 2014-09-10 ``@ddalle``: v1.0
    """
    # Delete output in the commands
    txt = doccmd.replace("cmd = ", "")
    # Initialize output
    docbin = ""
    kflag = True
    # Loop through lines
    for line in txt.split("\n"):
        # Check the line
        if line.lstrip().startswith(":Output"):
            kflag = False
        elif line.lstrip().startswith(":Versions:"):
            kflag = True
        # Keep the line if the flag is set
        if kflag:
            docbin += (line + "\n")
    # Output
    return docbin


# Function to call Tecplot on a macro
def tecmcr(mcr="export-lay.mcr", **kwargs):
    # Get command.
    cmdi = cmdgen.tecmcr(mcr, **kwargs)
    # Run the command.
    callf(cmdi, f="tecmcr.out", v=kwargs.get("v", True))
    # Remove the log file; it's useless
    os.remove("tecmcr.out")


# Docstring
tecmcr.__doc__ = _upgradeDocString(cmdgen.tecmcr.__doc__)


# Stand-alone function to run a Paraview script
def pvpython(lay, *args, **kw):
    r"""Stand-alone function to execute a Paraview Python script

    :Call:
        >>> pvpython(lay, *args, cmd="pvpython")
        >>> pvpython(lay, a1, a2, ...)
    :Inputs:
        *lay*: :class:`str`
            Name of script to run
        *cmd*: {``"pvpython"``} | :class:`str`
            Name of executable to use, may be full path
        *a1*: :class:`str`
            Command-line input to the script
    :Versions:
        * 2015-11-22 ``@ddalle``: v1.0
    """
    # Get name of executable
    fbin = kw.get('cmd', 'pvpython')
    # Command to run
    cmdi = [fbin] + [str(a) for a in args] + [lay]
    # Call the script
    callf(cmdi, f='pvpython.out')


# Stand-alone aflr3 binary
def aflr3(opts=None, j=0, **kw):
    r"""Run AFLR3 with the appropriate options

    :Call:
        >>> aflr3(opts=None, j=0, **kw)
    :Inputs:
        *opts*: :class:`cape.options.Options`
            Options interface with access to AFLR3 options
        *j*: :class:`int`
            Phase number
        *kw*: :class:`dict`
            Raw dictionary of command-line arguments
    :See also:
        * :func:`cape.cfdx.cmdgen.aflr3`
    :Versions:
        * 2016-04-04 ``@ddalle``: v1.0
    """
    # Get the command
    cmdi = cmdgen.aflr3(opts=opts, j=j, **kw)
    # Call the script
    callf(cmdi, f='aflr3.out')


# Function to call verify
def verify(opts=None, **kw):
    r"""Run Cart3D binary ``verify`` to test a triangulation

    :Call:
        >>> verify(opts=None, **kw)
    :Inputs:
        *opts*: :class:`cape.options.Options`
            Options interface with access to ``verify`` options
        *kw*: :class:`dict`
            Raw dictionary of command-line arguments
    :See also:
        * :func:`cape.cfdx.cmdgen.verify`
    :Versions:
        * 2016-04-05 ``@ddalle``: v1.0
    """
    # If there is currently a 'tecfile.bad' file, move it.
    if os.path.isfile('tecfile.bad'):
        os.rename('tecfile.bad', 'tecfile.old.bad')
    # Get command
    cmdi = cmdgen.verify(opts=opts, **kw)
    # Required file
    _assertfile(cmdi[1])
    # Run the command.
    ierr = calli(cmdi, f='verify.out')
    # Check status.
    if ierr or os.path.isfile('tecfile.bad'):
        # Create a failure file.
        f = open('FAIL', 'a+')
        # Write the reason
        f.write('verify\n')
        f.close()
        # Exit.
        raise SystemError('Triangulation contains errors!')


# Function to call intersect
def intersect(opts=None, **kw):
    r"""Run Cart3D ``intersect`` to combine overlapping triangulations

    :Call:
        >>> intersect(opts=None, **kw)
    :Inputs:
        *opts*: :class:`cape.options.Options`
            Options interface with access to ``verify`` options
        *kw*: :class:`dict`
            Raw dictionary of command-line arguments
    :See also:
        * :func:`cape.cfdx.cmdgen.intersect`
    :Versions:
        * 2016-04-05 ``@ddalle``: v1.0
    """
    # Get command.
    cmdi = cmdgen.intersect(opts=opts, **kw)
    # Required file
    _assertfile(cmdi[2])
    # Run the command.
    ierr = calli(cmdi, f='intersect.out')
    # Check status.
    if ierr or not os.path.isfile(cmdi[4]):
        # Create a failure file.
        f = open('FAIL', 'a+')
        # Write the reason
        f.write('intersect\n')
        f.close()
        # Exit.
        raise SystemError('Intersection failed!')

