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
from ..cfdx.cmdgen import (
    append_cmd_if,
    isolate_subsection,
    mpiexec,
    mpiexec_nogpu)


# Available Refine commands
_REFINE_COMMANDS = [
    "loop",
    "translate",
]

_REFINE_CMD_ARGS = [
    "sweeps",
]

_REFINE_CMD_KWARGS = [
    "aspect-ratio",
    "interpolant",
    "gradation",
    "mapbc",
    "mixed",
    "norm-power",
    "uniform",
]

# Available Refine commands
_REFINE_CMD_MAP = {
    "mapbc": "fun3d-mapbc",
    "sweeps": "s",
    "aspect_ratio": "aspect-ratio"
}


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
        * 2024-08-08 ``@ddalle``: v1.1; use mpiexec() for MPI directives
    """
    # Isolate opts for "RunControl" section
    opts = isolate_subsection(opts, Options, ("RunControl",))
    # Get nodet options
    nodet_opts = opts["nodet"]
    nodet_opts = nodet_opts.__class__(nodet_opts)
    # Apply other options
    nodet_opts.set_opts(kw)
    # MPI on/off
    q_mpi = opts.get_MPI(j)
    # MPI launch command, if appropriate
    cmdi = mpiexec(opts, j=j)
    # Name of executable
    f3dexec = "nodet_mpi" if q_mpi else "nodet"
    # Add to command
    cmdi.append(f3dexec)
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
            # Do not use
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
def refine(opts=None, j=0, **kw):
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
        * 2025-04-04 ``@ddalle``: v1.1; use ``mpiexec()``
        * 2025-06-13 ``@ddalle``: v1.2; more general, ramp_complexity
    """
    # Isolate opts for "RunControl" section
    opts = isolate_subsection(opts, Options, ("RunControl",))
    # Check for refine function
    func = kw.pop("function", "loop")
    # Generic refine call
    rfunc = "refine"
    # Append additional function if provided
    if func != "loop":
        rfunc += f"_{func}"
    # Get nodet options
    refine_opts = opts[rfunc]
    refine_opts = refine_opts.__class__(refine_opts)
    # Apply other options
    refine_opts.set_opts(kw)
    # MPI on/off
    q_mpi = opts.get_MPI(j)
    # MPI launch command, if appropriate
    cmdi = mpiexec_nogpu(opts, j=j)
    # Refine command
    refexec = "refmpi" if q_mpi else "ref"
    cmdi.append(refexec)
    # Add function
    cmdi.append(func)
    # Arg commands
    cmda = []
    # Keyword commands
    cmdk = []
    # Get complexity ramp values
    c0 = refine_opts.get_opt("initial_complexity", j=0)
    c1 = refine_opts.get_opt("ramp_complexity", j=0)
    # Get adaptation number
    n1 = opts.get_AdaptationNumber(j)
    # Apply ramp
    cr = None if (c0 is None or c1 is None) else c0 + n1*c1
    c = refine_opts.get_opt("complexity", j=j, vdef=cr)
    # Uset ramp if specified
    complexity = str(c if cr is None else c)
    # Add input file and output file
    ifile = refine_opts.get_opt("input", j=j)
    ofile = refine_opts.get_opt("output", j=j)
    igrid = refine_opts.get_opt("input_grid", j=j)
    ogrid = refine_opts.get_opt("output_grid", j=j)
    append_cmd_if(cmdi, ifile, [ifile])
    append_cmd_if(cmdi, ofile, [ofile])
    append_cmd_if(cmdi, igrid, [igrid])
    append_cmd_if(cmdi, ogrid, [ogrid])
    # Add complexity (required) to command
    if func == "loop" and complexity is not None:
        cmdi.append(complexity)
    # Loop through command-line inputs
    for k, v in refine_opts.items():
        # Check for special args
        if k in ("input", "output", "complexity", "run"):
            # Already processed
            continue
        elif "_" in k:
            # Different kind of option
            continue
        # Append args
        elif k in ['dist_solb']:
            cmda.append(str(v))
        # Use -s for "sweeps"
        k = 's' if k == "sweeps" else k
        # Check for single-char option
        prefix = '-' if len(k) == 1 else '--'
        cmdk.append(f"{prefix}{k}")
        # Index if list
        vj = getel(v, j)
        # Check for boolean
        if vj is True:
            continue
        # Append value
        cmdk.append(str(vj))
    # Ensure order i/o, args, kws
    cmdi += cmda + cmdk
    # Output
    return cmdi
