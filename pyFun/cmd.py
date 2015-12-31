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
# Get command to run tecplot
from .util import GetTecplotCommand


# Function to create ``nodet`` or ``nodet_mpi`` command
def nodet(opts=None, i=0, **kw):
    """Interface to FUN3D binary ``nodet`` or ``nodet_mpi``
    
    :Call:
        >>> cmdi = cmd.nodet(opts, i=0)
        >>> cmdi = cmd.nodet(**kw)
    :Inputs:
        *opts*: :class:`pyFun.options.Options`
            Global pyFun options interface or "RunControl" interface
        *i*: :class:`int`
            Run sequence index
        *animation_freq*: :class:`int`
            Output frequency
    :Outputs:
        *cmdi*: :class:`list` (:class:`str`)
            Command split into a list of strings
    :Versions:
        * 2015-11-24 ``@ddalle``: First version
    """
    # Check for options input
    if opts is not None:
        # Get values for run configuration
        n_mpi  = opts.get_MPI(i)
        nProc  = opts.get_nProc(i)
        mpicmd = opts.get_mpicmd(i)
        # Get dictionary of command-line inputs
        if "nodet" in opts:
            # pyFun.options.runControl.RunControl instance
            cli_nodet = opts["nodet"]
        elif "RunControl" in opts and "nodet" in opts["RunControl"]:
            # pyFun.options.Options instance
            cli_nodet = opts["RunControl"]["nodet"]
        else:
            # No command-line arguments
            cli_nodet = {}
    else:
        # Get values from keyword arguments
        n_mpi  = kw.get("MPI", False)
        nProc  = kw.get("nProc", 1)
        mpicmd = kw.get("mpicmd", "mpiexec")
        # Form other command-line argument dictionary
        cli_nodet = kw
        # Remove above options
        if "MPI"    in cli_nodet: del cli_nodet["MPI"]
        if "nProc"  in cli_nodet: del cli_nodet["nProc"]
        if "mpicmd" in cli_nodet: del cli_nodet["mpicmd"]
    # Form the initial command.
    if n_mpi:
        # Use the ``nodet_mpi`` command
        cmdi = [mpicmd, '-np', str(nProc), 'nodet_mpi']
    else:
        # Use the serial ``nodet`` command
        cmdi = ['nodet']
    # Check for "--adapt" flag
    if kw.get("adapt"):
        # That should be the only command-line argument
        cmdi.append("--adapt")
        return cmdi
    # Loop through command-line inputs
    for k in cli_nodet:
        # Get the value.
        v = cli_nodet[k]
        # Check the type
        if v == True:
            # Just an option with no value
            cmdi.append('--'+k)
        elif v == False or v is None:
            # Do not use.
            pass
        else:
            # Append the option and value 
            cmdi.append('--'+k)
            cmdi.append(str(v))
    # Output
    return cmdi
        
