"""
:mod:`cape.bin`: Cape binary interface module
=============================================

This template module provides an interface for simple command-line tools.  The
general approach for Cape is to create a function for each command-line binary
that is called.  This module contains two methods, :func:`cape.bin.calli` and
:func:`cape.bin.callf`, that are wrappers for the built-in
:func:`subprocess.call`, and several useful command-line utilities.

Furthermore, the module is programmed so that it is compatible with Python 2.6,
in which it is slightly more challenging to get the output of a system call.
The function :func:`cape.bin.callo` is provided to be a substitute for
:func:`subprocess.check_output` (which is only available in Python 2.7+).
Several useful system utilities are also provided that utilize this
output-gathering capability.

See also the :mod:`cape.cmd` module
"""

# File system and operating system management
import os
import subprocess as sp

# Import local command-generating module for complex commands
from . import cmd

# Imitate sp.check_output() for older versions
def check_output(cmdi):
    """Imitate the behavior of :func:`sp.check_output` using :func:`sp.Popen`
    
    :Call:
        >>> txt = check_output(cmdi)
    :Inputs:
        *cmdi*: :class:`list` (:class:`str`)
            List of strings as for :func:`subprocess.call`
    :Outputs:
        *txt*: :class:`str`
            Contents of STDOUT while executing *cmdi*
    :Versions:
        * 2017-03-13 ``@ddalle``: First version
    """
    # Call sp.Popen
    out = sp.Popen(cmdi, stdout=sp.PIPE).communicate()
    # Output
    return out[0]

# Function to call commands with a different STDOUT
def calli(cmdi, f=None, e=None, shell=None, v=True):
    """Call a command with alternate STDOUT by filename
    
    :Call:
        >>> ierr = calli(cmdi, f=None, e=None, shell=None, v=True)
    :Inputs:
        *cmdi*: :class:`list` (:class:`str`)
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
        * 2014-08-30 ``@ddalle``: First version
        * 2015-02-13 ``@ddalle``: Split into part with return code
        * 2017-03-12 ``@ddalle``: Added *v* option
        * 2019-06-10 ``@ddalle``: Added *e* option
    """
    # Process the shell option
    shell = bool(shell)
    # Print the command.
    print(" > " + " ".join(cmdi))
    # Print the current location.
    if v:
        print("     (PWD = '%s')" % os.getcwd())
    # Check for an output
    if f:
        # Print the location of STDOUT
        if v:
            print("     (STDOUT = '%s')" % os.path.split(f)[-1])
        # Print the location of STDERR
        if v and (e is not None):
            print("     (STDERR = '%s')" % os.path.split(e)[-1])
        # Open the files for STDOUT and STDERR
        fid = open(f, 'w')
        # Check for separate STDERR file
        if e is None:
            # Use STDOUT file
            fe = fid
        else:
            # Open separate file
            fe = open(e, 'w')
        # Call the command.
        ierr = sp.call(cmdi, stdout=fid, stderr=fe, shell=shell)
        # Close the file.
        fid.close()
    else:
        # Call the command.
        ierr = sp.call(cmdi, shell=shell)
    # Output
    return ierr
        
# Function to call commands with a different STDOUT
def callf(cmdi, f=None, e=None, shell=None, v=True, check=True):
    """Call a command with alternate STDOUT by filename
    
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
        * 2014-08-30 ``@ddalle``: First version
        * 2015-02-13 ``@ddalle``: Moved much to :func:`cape.bin.calli`
        * 2017-03-12 ``@ddalle``: Added *v* option
        * 2019-06-10 ``@ddalle``: Added *e* option
    """
    # Call the command with output status
    ierr = calli(cmdi, f, e, shell, v=v)
    # Check the status.
    if ierr and check:
        # Remove RUNNING file.
        if os.path.isfile('RUNNING'):
            # Delete it.
            os.remove('RUNNING')
        # Exit with error notifier.
        raise SystemError("Command failed with status %i." % ierr)
        
# Call command with output (since sp.check_output is Python 2.7+)
def callo(cmdi, shell=False):
    """Call a command and get the output text
    
    This function is basically a substitute for :func:`subprocess.check_output`,
    which is not available in Python 2.6.
    
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
        * 2016-04-01 ``@ddalle``: First version
    """
    # Call command and get output
    txt = sp.Popen(cmdi, stdout=sp.PIPE, shell=shell).communicate()[0]
    # Convert to unicode text
    return txt.decode("utf-8")

# Grep lines from a file
def grep(regex, fname):
    """Search for a regular expression in a file

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
        * 2015-12-28 ``@ddalle``: First version
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
    """Extract the first *n* lines of a file 
    
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
        * 2015-01-12 ``@ddalle``: First version
    """
    # Create the command.
    cmdi = ['head', '-%i'%n, fname]
    # Use Popen because check_output is 2.7+
    return callo(cmdi)
        
# Function to get the last line of a file.
def tail(fname, n=1):
    """Tail the last *n* lines of a file 
    
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
        * 2015-01-12 ``@ddalle``: First version
    """
    # Create the command.
    cmdi = ['tail', '-%i'%n, fname]
    # Use Popen because check_output is 2.7+
    return callo(cmdi)

# Simple function to make sure a file is present
def _assertfile(fname):
    """
    Assert that a given file exists and raise an exception otherwise
    
    :Call:
        >>> _assertfile(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file to test
    :Versions:
        * 2014-06-30 ``@ddalle``: First version
    """
    # Check for the file.
    if not os.path.isfile(fname):
        raise IOError("No input file '%s' found." % fname)
        
# Function to automate minor changes to docstrings to make them pyCart.Cntl
def _upgradeDocString(doccmd):
    """Upgrade docstrings from the :mod:`cape.pycart.bin` class
    
    :Call:
        >>> docbin = _upgradDocString(doccmd)
    :Inputs:
        *doccmd*: :class:`str`
            Docstring from method from :mod:`cape.pycart.cmd`
    :Outputs:
        *docbin*: :class:`str`
            Docstring for :mod:`cape.pycart.bin`
    :Versions:
        * 2014-09-10 ``@ddalle``: First version
    """
    # Replace module reference.
    txt = doccmd.replace(".cmd.", ".bin.")
    # Delete output in the commands.
    txt = txt.replace("cmd = ", "")
    # Initialize output.
    docbin = ""
    kflag = True
    # Loop through lines.
    for line in txt.split("\n"):
        # Check the line.
        if line.lstrip().startswith(":Output"):
            kflag = False
        elif line.lstrip().startswith(":Versions:"):
            kflag = True
        # Keep the line if the flag is set.
        if kflag: docbin += (line + "\n")
    # Output
    return docbin

# Function to call Tecplot on a macro
def tecmcr(mcr="export-lay.mcr", **kwargs):
    # Get command.
    cmdi = cmd.tecmcr(mcr, **kwargs)
    # Run the command.
    callf(cmdi, f="tecmcr.out")
    # Remove the log file; it's useless
    os.remove("tecmcr.out")
# Docstring
tecmcr.__doc__ = _upgradeDocString(cmd.tecmcr.__doc__)
    
# Stand-alone function to run a Paraview script
def pvpython(lay, *args, **kw):
    """Stand-alone function to execute a Paraview Python script
    
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
        * 2015-11-22 ``@ddalle``: First version
    """
    # Get name of executable
    fbin = kw.get('cmd', 'pvpython')
    # Command to run
    cmdi = [fbin, lay] + [str(a) for a in args]
    # Call the script
    callf(cmdi, f='pvpython.out')
    
# Stand-alone aflr3 binary
def aflr3(opts=None, j=0, **kw):
    """Run AFLR3 with the appropriate options
    
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
        * :func:`cape.cmd.aflr3`
    :Versions:
        * 2016-04-04 ``@ddalle``: First version
    """
    # Get the command
    cmdi = cmd.aflr3(opts=opts, j=j, **kw)
    # Call the script
    callf(cmdi, f='aflr3.out')

# Function to call verify
def verify(opts=None, **kw):
    """Run Cart3D binary ``verify`` to test a triangulation
    
    :Call:
        >>> verify(opts=None, **kw)
    :Inputs:
        *opts*: :class:`cape.options.Options`
            Options interface with access to ``verify`` options
        *kw*: :class:`dict`
            Raw dictionary of command-line arguments
    :See also:
        * :func:`cape.cmd.verify`
    :Versions:
        * 2016-04-05 ``@ddalle``: First version
    """
    # If there is currently a 'tecplot.bad' file, move it.
    if os.path.isfile('tecplot.bad'):
        os.rename('tecplot.bad', 'tecplot.old.bad')
    # Get command
    cmdi = cmd.verify(opts=opts, **kw)
    # Required file
    _assertfile(cmdi[1])
    # Run the command.
    ierr = calli(cmdi, f='verify.out')
    # Check status.
    if ierr or os.path.isfile('tecplot.bad'):
        # Create a failure file.
        f = open('FAIL', 'a+')
        # Write the reason
        f.write('verify\n')
        f.close()
        # Exit.
        raise SystemError('Triangulation contains errors!')

# Function to call intersect
def intersect(opts=None, **kw):
    """Run Cart3D binary ``intersect`` to combine overlapping triangulations
    
    :Call:
        >>> intersect(opts=None, **kw)
    :Inputs:
        *opts*: :class:`cape.options.Options`
            Options interface with access to ``verify`` options
        *kw*: :class:`dict`
            Raw dictionary of command-line arguments
    :See also:
        * :func:`cape.cmd.intersect`
    :Versions:
        * 2016-04-05 ``@ddalle``: First version
    """
    # Get command.
    cmdi = cmd.intersect(opts=opts, **kw)
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
# def intersect

