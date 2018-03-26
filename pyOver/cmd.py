"""
:mod:`pyOver.cmd`: Create commands for OVERFLOW executables 
=============================================================

This module creates system commands as lists of strings for executable binaries
or scripts for OVERFLOW.  While OVERFLOW has only one main executable, there
are several versions, which are all interpreted by the primary function of this
module, :func:`overrun`.

The basic format of the OVERFLOW command is determined using the value returned
from :func:`pyOver.options.runControl.get_overrun_cmd` (or the keyword argument
*cmd* to :func:`overrun`).  In some cases, the command will also depend on the
OVERFLOW file prefix (usually ``"run"``), which is stored in a variable *pre*,
the number of processors (technically cores) in *nProc*, and the phase number
*i* (0-based; OVERFLOW uses 1-based phase indexing).  Let 

    .. code-block:: python
    
        j = "%02i" % i
        
for convenience.  Finally, *mpicmd* is the system's call to run MPI
executables, which is often ``"mpiexec"``.

The format of the command that is generated can
take any of the following versions based on the input options.

    ====================  =====================================================
    Value for *cmd*       System command
    ====================  =====================================================
    ``"overrunmpi"``      [``"overrunmpi"``, ``"-np"``, *nProc*, *pre*, *j*]
    ``"overflowmpi"``     [*mpicmd*, ``"-np"``, *nProc*, ``"overflowmpi"``]
    ``"overrun"``         [``"overrun"``, *pre*, *j*]
    ``"overflow"``        [``"overflow"``]
    ====================  =====================================================

Commands are created in the form of a list of strings.  This is the format used
in the built-in module :mod:`subprocess` and also with :func:`cape.bin.calli`. 
As a very simple example, the system command ``"ls -lh"`` becomes the list
``["ls", "-lh"]``.

:See also:
    * :mod:`cape.cmd`
    * :mod:`cape.bin`
    * :mod:`pyOver.bin`
    * :mod:`pyOver.options.runControl`

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
        # Other args
        ofkw = opts.get_overrun_kw(i)
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
        # Additional args
        ofkw = {}
        for k in kw:
            # Check for recognized argument
            if k in ["MPI", "nProc", "mpicmd", "cmd", "aux", "args", "Prefix"]:
                continue
            # Otherwise, add the argument
            ofkw[k] = kw[k]
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
    # Loop through dictionary of other arguments
    for k in ofkw:
        # Get the value
        v = ofkw[k]
        # Check the type
        if v in [None, False]:
            continue
        elif v == True:
            # Just add a tag
            cmdi.append('-%s' % k)
        else:
            # Append option and value
            cmdi.append('-%s' % k)
            cmdi.append('%s' % v)
    # Output
    return cmdi
        
