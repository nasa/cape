r"""
:mod:`cape.pyover.cmd`: Create commands for OVERFLOW executables
=================================================================

This module creates system commands as lists of strings for executable
binaries or scripts for OVERFLOW. While OVERFLOW has only one main
executable, there are several versions, which are all interpreted by the
primary function of this module, :func:`overrun`.

The basic format of the OVERFLOW command is determined using the value
returned from :func:`pyOver.options.runControl.get_overrun_cmd` (or the
keyword argument *cmd* to :func:`overrun`). In some cases, the command
will also depend on the OVERFLOW file prefix (usually ``"run"``), which
is stored in a variable *pre*, the number of processors (technically
cores) in *nProc*, and the phase number *i* (0-based; OVERFLOW uses
1-based phase indexing).  Let

    .. code-block:: python

        j = "%02i" % i

for convenience.  Finally, *mpicmd* is the system's call to run MPI
executables, which is often ``"mpiexec"``.

The format of the command that is generated can
take any of the following versions based on the input options.

    ================= ==================================================
    Value for *cmd*   System command
    ================= ==================================================
    ``"overrunmpi"``  [``"overrunmpi"``, ``"-np"``, *nProc*, *pre*, *j*]
    ``"overflowmpi"`` [*mpicmd*, ``"-np"``, *nProc*, ``"overflowmpi"``]
    ``"overrun"``     [``"overrun"``, *pre*, *j*]
    ``"overflow"``    [``"overflow"``]
    ================= ==================================================

Commands are created in the form of a list of strings. This is the
format used in the built-in module :mod:`subprocess` and also with
:func:`cape.bin.calli`. As a very simple example, the system command
``"ls -lh"`` becomes the list ``["ls", "-lh"]``.

:See also:
    * :mod:`cape.cmdgen`
    * :mod:`cape.cmdrun`
    * :mod:`cape.pyover.cmdrun`
    * :mod:`cape.pyover.options.runctlopts`
"""

# Standard library

# Local imports
from .options import Options
from ..cfdx.cmdgen import isolate_subsection, append_cmd_if


# Function to create ``nodet`` or ``nodet_mpi`` command
def overrun(opts=None, j=0, **kw):
    r"""Interface to OVERFLOW binary

    :Call:
        >>> cmdi = overrun(opts, i=0)
        >>> cmdi = overrun(**kw)
    :Inputs:
        *opts*: :class:`pyFun.options.Options`
            Global pyFun options interface or "RunControl" interface
        *j*: {``0``} | :class:`int`
            Phase number
        *args*: :class:`str`
            Extra arguments to *cmd*
        *aux*: :class:`str`
            Auxiliary flag
        *cmd*: :class:`str`
            Name of OVERFLOW binary to use
    :Outputs:
        *cmdi*: :class:`list`\ [:class:`str`]
            Command split into a list of strings
    :Versions:
        * 2016-02-02 ``@ddalle``: v1.0
        * 2023-06-21 ``@ddalle``: v1.1; use isolate_subsection()
    """
    # Isolate options
    opts = isolate_subsection(opts, Options, ("RunControl",))
    # Get values for run configuration
    n_mpi = opts.get_MPI(j)
    nProc = opts.get_nProc(j)
    mpicmd = opts.get_opt("mpicmd", j=j)
    # OVERFLOW flags
    ofcmd = opts.get_overrun_cmd(j)
    args = opts.get_overrun_args(j)
    aux = opts.get_overrun_aux(j)
    # Other args
    ofkw = opts.get_overrun_kw(j)
    # Base name
    pre = opts.get_Prefix(j)
    # Split command
    ofcmd = ofcmd.split()
    # Form string for initial part of command
    if ofcmd[0] == "overrunmpi":
        # Use the ``overrunmpi`` script
        cmdi = ofcmd + ['-np', str(nProc), pre, '%02i' % (j+1)]
    elif ofcmd[0] == "overflowmpi":
        # Use the ``overflowmpi`` command
        cmdi = mpicmd + ['-np', str(nProc), ofcmd]
    elif ofcmd[0] == "overrun":
        # Use the ``overrun`` script
        cmdi = ofcmd + [pre, '%02i' % (j+1)]
    elif n_mpi:
        # Default to "overflowmpi"
        cmdi = [mpicmd, '-np', str(nProc)] + ofcmd
    else:
        # Use the serial
        cmdi = ofcmd
    # Append ``-aux`` flag
    append_cmd_if(cmdi, aux, ['-aux', aux])
    # Append extra arguments
    append_cmd_if(cmdi, args, args)
    # Loop through dictionary of other arguments
    for k, v in ofkw.items():
        # Create command
        cmdk = [f'-{k}']
        # Process v=True as just '-k', else '-k v'
        if v is not True:
            cmdk.append(str(v))
        # Append command if *v* is not False-like
        append_cmd_if(cmdi, v, cmdk)
    # Output
    return cmdi

