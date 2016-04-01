"""
Cape binary interface module: :mod:`cape.bin`
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
"""

# File system and operating system management
import os
import subprocess as sp

# Function to call commands with a different STDOUT
def calli(cmdi, f=None, shell=None):
    """Call a command with alternate STDOUT by filename
    
    :Call:
        >>> calli(cmdi, f=None, shell=None)
    :Inputs:
        *cmdi*: :class:`list` (:class:`str`)
            List of strings as for :func:`subprocess.call`
        *f*: :class:`str`
            File name to which to store STDOUT
        *shell*: :class:`bool`
            Whether or not a shell is needed
    :Outputs:
        *ierr*: :class:`int`
            Return code, ``0`` for successful execution
    :Versions:
        * 2014-08-30 ``@ddalle``: First version
        * 2015-02-13 ``@ddalle``: Split into part with return code
    """
    # Process the shell option
    shell = bool(shell)
    # Print the command.
    print(" > " + " ".join(cmdi))
    # Print the current location.
    print("     (PWD = '%s')" % os.getcwd())
    # Check for an output
    if f:
        # Print the location of STDOUT
        print("     (STDOUT = '%s')" % str(f))
        # Open the file.
        fid = open(f, 'w')
        # Call the command.
        ierr = sp.call(cmdi, stdout=fid, stderr=fid, shell=shell)
        # Close the file.
        fid.close()
    else:
        # Call the command.
        ierr = sp.call(cmdi, shell=shell)
    # Output
    return ierr
        
# Function to call commands with a different STDOUT
def callf(cmdi, f=None, shell=None, t=None):
    """Call a command with alternate STDOUT by filename
    
    :Call:
        >>> callf(cmdi, f=None, shell=None)
    :Inputs:
        *cmdi*: :class:`list` (:class:`str`)
            List of strings as for :func:`subprocess.call`
        *f*: :class:`str`
            File name to which to store STDOUT
        *shell*: :class:`bool`
            Whether or not a shell is needed
    :Versions:
        * 2014-08-30 ``@ddalle``: First version
        * 2015-02-13 ``@ddalle``: Split most of code to :func:`cape.bin.calli`
    """
    # Call the command with output status
    ierr = calli(cmdi, f, shell)
    # Check the status.
    if ierr:
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
    return sp.Popen(cmdi, stdout=sp.PIPE, shell=shell).communicate()[0]

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
        
# Function to automate minor changes to docstrings to make them pyCart.Cart3d
def _upgradeDocString(doccmd):
    """Upgrade docstrings from the :mod:`pyCart.bin` class
    
    :Call:
        >>> docbin = _upgradDocString(doccmd)
    :Inputs:
        *doccmd*: :class:`str`
            Docstring from method from :mod:`pyCart.cmd`
    :Outputs:
        *docbin*: :class:`str`
            Docstring for :mod:`pyCart.bin`
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
    
# Stand-alone function to run a Tecplot layout file
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
    
    
