"""
Cart3D binary interface module: :mod:`pyCart.bin`
=================================================

"""

# File system and operating system management
import os
import subprocess as sp

# Command option processing
import cmd

# Function to call commands with a different STDOUT
def calli(cmdi, f=None, shell=None):
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
def callf(cmdi, f=None, shell=None):
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
    cmd = ['tail', '-%i'%n, fname]
    # Use Popen because check_output is 2.7+
    return sp.Popen(cmd, stdout=sp.PIPE).communicate()[0]
        
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
    

# Function to call cubes.
def cubes(cart3d=None, **kwargs):
    # Required file
    _assertfile('input.c3d')
    # Get command
    cmdi = cmd.cubes(cart3d=cart3d, **kwargs)
    # Run the command.
    callf(cmdi, f='cubes.out')
# Docstring
cubes.__doc__ = _upgradeDocString(cmd.cubes.__doc__)

# Function to call verify
def verify(ftri='Components.i.tri'):
    # Required file
    _assertfile(ftri)
    # If there is currently a 'tecplot.bad' file, move it.
    if os.path.isfile('tecplot.bad'):
        os.rename('tecplot.bad', 'tecplot.old.bad')
    # Get command
    cmdi = cmd.verify(ftri)
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
# Docstring
verify.__doc__ = _upgradeDocString(cmd.verify.__doc__)

# Function to call intersect
def intersect(fin='Components.tri', fout='Components.i.tri'):
    # Required file
    _assertfile(fin)
    # Get command.
    cmdi = cmd.intersect(fin=fin, fout=fout)
    # Run the command.
    ierr = calli(cmdi, f='intersect.out')
    # Check status.
    if ierr or not os.path.isfile(fout):
        # Create a failure file.
        f = open('FAIL', 'a+')
        # Write the reason
        f.write('intersect\n')
        f.close()
        # Exit.
        raise SystemError('Intersection failed!')
    
# Function to call mgPrep
def mgPrep(cart3d=None, **kwargs):
    # Required file
    _assertfile('Mesh.R.c3d')
    # Get the command.
    cmdi = cmd.mgPrep(cart3d=cart3d, **kwargs)
    # Run the command.
    callf(cmdi, f='mgPrep.out')
# Docstring
mgPrep.__doc__ = _upgradeDocString(cmd.mgPrep.__doc__)
    
# Function to call mgPrep
def autoInputs(cart3d=None, **kwargs):
    # Get command.
    cmdi = cmd.autoInputs(cart3d, **kwargs)
    # Run the command.
    callf(cmdi, f='autoInputs.out')
# Docstring
autoInputs.__doc__ = _upgradeDocString(cmd.autoInputs.__doc__)
    
    
    
# Function to call flowCart
def flowCart(cart3d=None, i=0, **kwargs):
    # Check for cart3d input
    if cart3d is not None:
        # Get values from internal settings.
        nProc   = cart3d.opts.get_OMP_NUM_THREADS(i)
    else:
        # Get values from keyword arguments
        nProc   = kwargs.get('nProc', 4)
    # Set environment variable.
    if nProc:
        os.environ['OMP_NUM_THREADS'] = str(nProc)
    # Get command.
    cmdi = cmd.flowCart(cart3d=cart3d, i=i, **kwargs)
    # Run the command
    callf(cmdi, f='flowCart.out')
# Docstring
flowCart.__doc__ = _upgradeDocString(cmd.flowCart.__doc__)

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

