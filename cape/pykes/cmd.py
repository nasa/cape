"""
:mod:`cape.pykes.cmd`: Create commands for Kestrel executables 
=================================================================

This module creates system commands as lists of strings for executable
binaries or scripts for Kestrel. The command-line interface to Kestrel
is quite simple, so commands just take the form

    .. code-block:: console

        $ csi -p NUMCORES -i XMLFILE

"""

# Standard library
import os.path


# Function to create ``nodet`` or ``nodet_mpi`` command
def csi(opts=None, i=0, **kw):
    r"""Create commands for main Kestrel executable
    
    :Call:
        >>> cmdi = csi(opts, i=0)
        >>> cmdi = csi(**kw)
    :Inputs:
        *opts*: :class:`Options` | :class:`RunControl`
            Global pykes options interface or "RunControl" interface
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
        n_proc  = opts.get_nProc(i)
        # Job prefix
        fxml = opts.get_Prefix(i)
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
        
