r"""
:mod:`cape.pyfun.cmdgen`: Create commands for FUN3D executables
=================================================================

This module creates system commands as lists of strings for executable
binaries or scripts for FUN3D.  The main FUN3D executables are ``nodet``
or ``nodet_mpi``, for which command are created using :func:`nodet`, and
``dual`` or ``dual_mpi``, whose commands are constructed using
:func:`dual`.

Commands are created in the form of a list of strings.  This is the
format used in the built-in module :mod:`subprocess` and also with
:func:`cape.bin.calli`. As a very simple example, the system command
``"ls -lh"`` becomes the list ``["ls", "-lh"]``.

These commands also include prefixes such as ``mpiexec`` if necessary.
The decision to use ``nodet`` or ``nodet_mpi`` is made based on the
options input of keyword input ``"MPI"``.  For example, two versions of
the command returned by :func:`nodet` could be

    .. code-block:: python

        ["mpiexec", "-np", "240", "nodet_mpi", "--plt_tecplot_output"]
        ["nodet", "--plt_tecplot_output"]

:See also:
    * :mod:`cape.cfdx.cmd`
    * :mod:`cape.cfdx.bin`
    * :mod:`cape.pyfun.bin`
    * :mod:`cape.pyfun.options.runctlopts`

"""

# Local imports
from .options import runctlopts, Options
from .options.util import getel
from ..cfdx.cmdgen import isolate_subsection


# Available Refine commands
_REFINE_COMMANDS = [
    "adapt",
    "bootstrap",
    "collar",
    "distance",
    "examine",
    "interpolate",
    "loop",
    "multiscale",
    "surface",
    "translate",
    "visualize"
]


# Function to create ``nodet`` or ``nodet_mpi`` command
def nodet(opts=None, j=0, **kw):
    r"""Interface to FUN3D binary ``nodet`` or ``nodet_mpi``

    :Call:
        >>> cmdi = nodet(opts, i=0)
        >>> cmdi = nodet(**kw)
    :Inputs:
        *opts*: :class:`pyFun.options.Options`
            Global pyFun options interface or "RunControl" interface
        *i*: :class:`int`
            Phase number
        *animation_freq*: :class:`int`
            Output frequency
    :Outputs:
        *cmdi*: :class:`list`\ [:class:`str`]
            Command split into a list of strings
    :Versions:
        * 2015-11-24 ``@ddalle``: v1.0
        * 2023-08-18 ``@ddalle``: v1.0; use isolate_subsection()
    """
    # Isolate opts for "RunControl" section
    opts = isolate_subsection(opts, Options, ("RunControl",))
    # Get nodet options
    nodet_opts = opts["nodet"]
    nodet_opts = nodet_opts.__class__(nodet_opts)
    # Apply other options
    nodet_opts.set_opts(kw)
    # Get values for run configuration
    n_mpi  = opts.get_MPI(j)
    nProc  = opts.get_nProc(j)
    mpicmd = opts.get_mpicmd(j)
    # Form the initial command.
    if n_mpi:
        # Use the ``nodet_mpi`` command
        if isinstance(nProc, int) and nProc > 0:
            # Request specific number of processes
            cmdi = [mpicmd, '-np', str(nProc), 'nodet_mpi']
        else:
            # Determine process count automatically
            cmdi = [mpicmd, 'nodet_mpi']
    else:
        # Use the serial ``nodet`` command
        cmdi = ['nodet']
    # Loop through command-line inputs
    for k in nodet_opts:
        # Get the value
        v = nodet_opts.get_opt(k, j=j)
        # Check the type
        if k == "run":
            # Skip pyfun setting not passed to nodet
            continue
        elif v is True:
            # Just an option with no value
            cmdi.append(f'--{k}')
        elif v is False or v is None:
            # Do not use.
            pass
        else:
            # Append the option and value
            cmdi.append(f'--{k}')
            cmdi.append(str(v))
    # Output
    return cmdi


# Function to create ``dual`` or ``dual_mpi`` command
def dual(opts=None, i=0, **kw):
    r"""Interface to FUN3D binary ``dual`` or ``dual_mpi``

    :Call:
        >>> cmdi = dual(opts, i=0)
        >>> cmdi = dual(**kw)
    :Inputs:
        *opts*: :class;`pyFun.options.Options`
            Global pyFun options interface or "RunControl" interface
        *i*: :class:`int`
            Phase number
        *outer_loop_krylov*: {``True``} | ``False``
            Whether or not to use ``--outer_loop_krylov`` option
        *rad*: {``True``} | ``False``
            Whether or not to use residual adjoint dot product
        *adapt*: {``True``} | ``False``
            Whether or not to adapt at the end of the cycle
    :Outputs:
        *cmdi*: :class:`list`\ [:class:`str`]
            Command split into a list of strings
    :Versions:
        * 2016-04-28 ``@ddalle``: v1.0
    """
    # Check for options input
    if opts is not None:
        # Downselect to "RunControl" section if necessary
        if "RunControl" in opts:
            opts = opts["RunControl"]
        # Downselect to "dual" section if necessary
        if "dual" in opts:
            # Get values for run configuration
            n_mpi  = opts.get_MPI(i)
            nProc  = opts.get_nProc(i)
            mpicmd = opts.get_mpicmd(i)
            # Downselect
            opts = opts["dual"]
        else:
            # Use defaults
            n_mpi  = runctlopts.rc0('MPI')
            nProc  = runctlopts.rc0('nProc')
            mpicmd = runctlopts.rc0('mpicmd')
    else:
        # Use defaults
        n_mpi  = runctlopts.rc0('MPI')
        nProc  = runctlopts.rc0('nProc')
        mpicmd = runctlopts.rc0('mpicmd')
        # Default
        opts = runctlopts.dual()
    # Process keyword overrides
    n_mpi  = kw.get('MPI',    n_mpi)
    nProc  = kw.get('nProc',  nProc)
    mpicmd = kw.get('mpicmd', mpicmd)
    # Process default options
    olk   = opts.get_dual_outer_loop_krylov(i)
    rad   = opts.get_dual_rad(i)
    adapt = opts.get_dual_adapt(i)
    # Process keyword overrides
    olk   = kw.get('outer_loop_krylov', olk)
    rad   = kw.get('rad',               rad)
    adapt = kw.get('adapt',             adapt)
    # Process other options
    cli_dual = dict(**opts)
    # Known flags
    flags = ['MPI', 'nProc', 'mpicmd', 'outer_loop_krylov', 'rad', 'adapt']
    # Remove known flags
    for k in cli_dual.keys():
        # Remove it
        if k in flags:
            del cli_dual[k]
    # Append *kw* flags
    for k in kw:
        # Check if recognized
        if k not in flags:
            cli_dual[k] = kw[k]
    # Form the initial command
    if n_mpi:
        # Use the ``dual_mpi`` command
        cmdi = [mpicmd, '-np', str(nProc), 'dual_mpi']
    else:
        # Use the serial ``dual`` command
        cmdi = ['dual']
    # Process known options
    if adapt:
        cmdi.append('--adapt')
    if rad:
        cmdi.append('--rad')
    if olk:
        cmdi.append('--outer_loop_krylov')
    # Loop through command-line inputs
    for k in cli_dual:
        # Get the value.
        v = cli_dual[k]
        # Check the type
        if v is True:
            # Just an option with no value
            cmdi.append('--'+k)
        elif v is False or v is None:
            # Do not use.
            pass
        else:
            # Select phase option
            vi = getel(v, i)
            # Append the option and value
            cmdi.append('--'+k)
            cmdi.append(str(vi))
    # Output
    return cmdi


# Function to create ``ref`` or ``refmpi`` command
def refine(opts=None, i=0, **kw):
    r"""Interface to Refine adaptation binary ``ref`` or ``refmpi``

    :Call:
        >>> cmdi = refine(opts, i=0)
        >>> cmdi = refine(**kw)
    :Inputs:
        *opts*: :class:`pyFun.options.Options`
            Global pyFun options interface or "RunControl" interface
        *i*: :class:`int`
            Phase number
    :Outputs:
        *cmdi*: :class:`list`\ [:class:`str`]
            Command split into a list of strings
    :Versions:
        * 2023-06-30 ``@jmeeroff``: v1.0
    """
    # Check for options input
    if opts is not None:
        # Get values for run configuration; MPI will be set by global var
        n_mpi  = opts.get_MPI(i)
        nProc  = opts.get_nProc(i)
        mpicmd = opts.get_mpicmd(i)
        # Get dictionary of command-line inputs
        if "refine" in opts:
            # pyFun.options.runctlopts.RunControl instance
            cli_refine = opts["refine"]
        elif "RunControl" in opts and "refine" in opts["RunControl"]:
            # pyFun.options.Options instance
            cli_refine = opts["RunControl"]["refine"]
        else:
            # No command-line arguments
            cli_refine = {}
    else:
        # Get values from keyword arguments
        n_mpi  = kw.pop("MPI", False)
        nProc  = kw.pop("nProc", 1)
        mpicmd = kw.pop("mpicmd", "mpiexec")
        # Form other command-line argument dictionary
        cli_refine = kw
    # Form the initial command.
    if n_mpi:
        # Use the ``refmpi`` command
        cmdi = [mpicmd, '-np', str(nProc), 'refmpi']
    else:
        # Use the serial ``ref`` command
        cmdi = ['ref']
        # Loop through command-line inputs
    for k in cli_refine:
        # Get the value
        v = cli_refine[k]
        # Check the type
        if v in _REFINE_COMMANDS:
            # This is a refine function so just append it
            cmdi.append(v)
        # Check keys for input/ouput
        elif k in ['input', 'output']:
            # Append it
            cmdi.append(v)
    # Output
    return cmdi


