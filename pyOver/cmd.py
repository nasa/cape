"""
Create command strings for FUN3D binaries: :mod:`pyFun.cmd`
===========================================================

This module contains direct interfaces to Cart3D binaries so that they can be
called from Python.  Settings for the binaries that are usually specified as
command-line options are either specified as keyword arguments or inherited from
a :class:`pyFun.fun3d.Fun3d` instance or inherited from a
:class:`pyFun.options.Options` or :class:`pyFun.options.runControl.RunControl`
instance.
"""

# File checking.
import os.path


# Function to create ``nodet`` or ``nodet_mpi`` command
def overrun(opts=None, i=0, **kw):
    """Interface to FUN3D binary ``nodet`` or ``nodet_mpi``
    
    :Call:
        >>> cmdi = cmd.overrun(opts, i=0)
        >>> cmdi = cmd.overrun(**kw)
    :Inputs:
        *opts*: :class:`pyFun.options.Options`
            Global pyFun options interface or "RunControl" interface
        *i*: :class:`int`
            Phase number
        *args*: :class:`str`
            Extra arguments to *cmd*
        *aux*: :class:`str`
            Auxiliary flag
        *cmd*: :class:`str`
            Name of OVERFLOW binary to use
    :Outputs:
        *cmdi*: :class:`list` (:class:`str`)
            Command split into a list of strings
    :Versions:
        * 2016-02-02 ``@ddalle``: First version
    """
    # Check for options input
    if opts is not None:
        # Get values for run configuration
        n_mpi  = opts.get_MPI(i)
        nProc  = opts.get_nProc(i)
        # OVERFLOW flags
        ofcmd  = opts.get_overrun_cmd(i)
        aux    = opts.get_overrun_aux(i)
        args   = opts.get_overrun_args(i)
        # Base name
        pre = opts.get_Prefix(i)
    else:
        # Get values from keyword arguments
        n_mpi  = kw.get("MPI", False)
        nProc  = kw.get("nProc", 1)
        mpicmd = kw.get("mpicmd", "mpiexec")
        # OVERFLOW flags
        ofcmd  = kw.get("cmd", "overrunmpi")
        aux    = kw.get("aux", '\"-v pcachem -- dplace -s1\"')
        args   = kw.get("args", "")
        # Prefix
        pre    = kw.get("Prefix", "run")
    # Split command
    ofcmd = ofcmd.split()
    # Form string for initial part of command
    if ofcmd[0] == "overrunmpi":
        # Use the ``overrunmpi`` script
        cmdi = ofcmd + ['-np', str(nProc), pre, '%02i'%(i+1)]
    elif ofcmd[0] == "overflowmpi":
        # Use the ``overflowmpi`` command
        cmdi = mpicmd + ['-np', str(nProc), ofcmd]
    elif ofcmd[0] == "overrun":
        # Use the ``overrun`` script
        cmdi = ofcmd + [pre, '%02i'%(i+1)]
    elif n_mpi:
        # Default to "overflowmpi"
        cmdi = [mpicmd, '-np', str(nProc)] + ofcmd
    else:
        # Use the serial 
        cmdi = ofcmd
    # Append ``-aux`` flag
    if aux: cmdi = cmdi + ['-aux', aux]
    # Append extra arguments
    if args: cmdi.append(args)
    # Output
    return cmdi
        
