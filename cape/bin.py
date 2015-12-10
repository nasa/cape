"""
CAPE binary interface module: :mod:`cape.bin`
=============================================

This template module provides an interface for simple command-line tools
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
        # 2015-02-13 ``@ddalle``: Split most of code to :func:`calli`
    """
    # Open time file
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
    return sp.Popen(cmdi, stdout=sp.PIPE).communicate()[0]
        
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
    cmd = ['head', '-%i'%n, fname]
    # Use Popen because check_output is 2.7+
    return sp.Popen(cmd, stdout=sp.PIPE).communicate()[0]

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
def pvpython(lay, *args):
    """Stand-alone function to execute a Paraview Python script
    
    :Call:
        >>> pvpython(lay, *args)
        >>> pvpython(lay, a1, a2, ...)
    :Inputs:
        *lay*: :class:`str`
            Name of script to run
        *a1*: :class:`str`
            Command-line input to the script
    :Versions:
        * 2015-11-22 ``@ddalle``: First version
    """
    # Command to run
    cmdi = ['pvpython', lay] + [str(a) for a in args]
    # Call the script
    callf(cmdi, stdout='pvpython.out')
    
    
