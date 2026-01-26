r"""
:mod:`cape.cfdx.cmd`: Creating system commands
==============================================

This module creates system commands as lists of strings for binaries or
scripts that require multiple command-line options. It is closely tied
to :mod:`cape.cfdx.bin`.

Commands are created in the form of a list of strings.  This is the
format used in :mod:`subprocess` commands (with *shell*\ =``False``\ )
and also :func:`cape.bin.calli`, etc.  As a very simple example, the
system command ``"ls -lh"`` becomes the list ``["ls", "-lh"]``.

Inputs to the functions in this module take one of two forms:

    * A :class:`cape.options.Options` object or subset thereof
    * Keyword arguments

The first method allows any appropriate :class:`Options` interface and
then extracts the appropriate portion. For instance :func:`verify` can
be given either a top-level options object, a ``"RunControl"`` options
interface (e.g. from :mod:`cape.options.runControl`), or a :class:`dict`
of options specific to ``verify``.  It does this using the following
Python commands:

    .. code-block:: python

        opts = opts.get("RunControl", opts)
        opts = opts.get("verify", opts)

Other functions, such as :func:`aflr3`, rely on the built-in methods of
the :class:`cape.cfdx.options.Options` class.  For example,
``opts.get_aflr3_i(j)`` returns the ``aflr3`` input file name for phase
*j* if *opts* is in any of the following classes:

    ============================================  ======================
    Class                                         Description
    ============================================  ======================
    :class:`cape.cfdx.options.Options`            All Cape settings
    :class:`cape.cfdx.options.RunControlOpts`     All run settings
    :mod:`cape.cfdx.options.aflr3opts`            AFLR3 settings
    ============================================  ======================

"""

# Standard library
import os
from typing import Optional, Union

# Local imports
from ..cfdx.options import Options
from ..optdict import OptionsDict
from ..util import GetTecplotCommand


# Generic function to isloate a subsection
def isolate_subsection(opts, cls: type, subsecs: tuple):
    r"""Get an options class instance for a section or subsection

    For example this class can be used to get the ``"RunControl"``
    section whether *opts* is a :class:`dict`, overall :class:`Options`
    instance, or already is an instance of :class:`RunControlOpts`. It
    allows the user to get the correct section automatically by either
    using an instance of that section's class directly or by giving
    any parent thereof.

    :Call:
        >>> myopts = isolate_subsection(opts, cls, subsecs)
    :Inputs:
        *opts*: ``None`` | :class:`dict` | :class:`OptionsDict`
            Options interface of either target section or parent thereof
        *cls*: :class:`type`
            Subclass of :class:`OptionsDict` for top-level options
        *subsecs*: :class:`tuple`\ [:class:`str`]
            Tuple of subsections of *cls* to isolate
    :Versions:
        * 2023-08-18 ``@ddalle``: v1.0
    """
    # Check class
    if not isinstance(cls, type):
        # Not a type
        raise TypeError(
            "Target class must be of type 'type'; " +
            f"got '{cls.__class__.__name__}'")
    elif not issubclass(cls, OptionsDict):
        # Not a subclass of OptionsDict
        raise TypeError(
            f"Target class '{cls.__name__}' is not a subclass of OptionsDict")
    # Check inputs
    if not isinstance(subsecs, tuple):
        raise TypeError(
            "List of sections must be 'tuple'; " +
            f"got '{subsecs.__class__.__name__}'")
    # Check types
    for j, secname in enumerate(subsecs):
        # Ensure name of section is a string
        if not isinstance(secname, str):
            raise TypeError(
                f"Subsection {j} name must be 'str'; " +
                f"got '{secname.__class__.__name__}'")
    # Check for case with no subsections
    if len(subsecs) == 0:
        # Check input type
        if opts is None:
            # Create empty instance of appropriate class
            return cls()
        elif isinstance(opts, cls):
            # Already got target class
            return opts
        elif isinstance(opts, dict):
            # Convert raw dict to target class
            return cls(opts)
        else:
            # Invalid
            raise TypeError(
                f"Options must be None, 'dict', or '{cls.__name__}'; " +
                f"got '{opts.__class__.__name__}'")
    # Get name of first-level subsection
    secname = subsecs[0]
    # Get class for first subsection
    seccls = cls.getx_cls_key("_sec_cls", secname)
    # Check for recognized section
    if seccls is None:
        raise ValueError(f"Class '{cls.__name__}' has no section '{secname}'")
    # Isolate section if needed
    if isinstance(opts, cls):
        # Get appropriate sec (guaranteed present b/c of OptionsDict)
        opts = opts[secname]
    # Recurse
    return isolate_subsection(opts, seccls, subsecs[1:])


# Append to a command if not None
def append_cmd_if(cmdi: list, opt, app: list, exc=None):
    r"""Append to command list if value is not False-like

    :Call:
        >>> append_cmd_if(cmdi, opt, app, exc=None)
    :Inputs:
        *cmdi*: :class:`list`\ [:class:`str`]
            List of strings of existing command, appended
        *opt*: :class:`object`
            Value to to test whether to append
        *app*: :class:`list`\ [:class:`str`]
            Commands to append to *cmdi* if *opt*
        *exc*: {``None``} | :class:`BaseException`
            Exception to raise if *opt* is False-like
    :Versions:
        * 2023-08-19 ``@ddalle``: v1.0
    """
    # Check validity
    if opt:
        # Otherwise append command
        cmdi.extend(app)
        return
    # Check for exception
    if exc:
        raise exc


# Append to a command if not None
def append_cmd_if_not_none(cmdi: list, opt, app: list, exc=None):
    r"""Append to command list if value is not ``None``

    :Call:
        >>> append_cmd_if_not_none(cmdi, opt, app, exc=None)
    :Inputs:
        *cmdi*: :class:`list`\ [:class:`str`]
            List of strings of existing command, appended
        *opt*: :class:`object`
            Value to to test whether to append
        *app*: :class:`list`\ [:class:`str`]
            Commands to append to *cmdi* if *opt* is not ``None``
        *exc*: {``None``} | :class:`BaseException`
            Exception to raise if *opt* is ``None``
    :Versions:
        * 2023-08-19 ``@ddalle``: v1.0
    """
    # Check validity
    if opt is not None:
        # Otherwise append command
        cmdi.extend(app)
        return
    # Check for exception
    if exc:
        raise exc


# Function to run tecplot
def tecmcr(mcr="export-lay.mcr", **kw):
    r"""Run a Tecplot macro

    :Call:
        >>> cmd = tecmcr(mcr="export-lay.mcr")
    :Inputs:
        *mcr*: :class:`str`
            File name of Tecplot macro
    :Outputs:
        *cmd*: :class:`list` (:class:`str`)
            Command split into a list of strings
    :Versions:
        * 2015-03-10 ``@ddalle``: v1.0
    """
    # Get tecplot command
    t360 = GetTecplotCommand()
    # Form the command
    cmd = [t360, '-b', '-p', mcr, '-mesa']
    # Output
    return cmd


# Function to create MPI command (usually a prefix)
def mpiexec(opts: Optional[OptionsDict] = None, j: int = 0, **kw) -> list:
    r"""Create command [prefix] to run MPI, e.g. ``mpiexec -np 10``

    :Call:
        >>> cmdi = mpiexec(opts=None, j=0, **kw)
    :Inputs:
        *opts*: {``None``} | :class:`OptionsDict`
            Options interface, either global or "RunControl" section
        *j*: {``0``} | :class:`int`
            Phase number
    :Outputs:
        *cmdi*: :class:`list`\ [:class:`str`]
            System command created as list of strings
    :Versions:
        * 2024-08-08 ``@ddalle``: v1.0
        * 2024-10-10 ``@ddalle``: v1.1; add *args* and *threads*
    """
    # Isolate opts for "RunControl" section
    rc = isolate_subsection(opts, Options, ("RunControl",))
    # Get mpi options section
    mpi_opts = rc["mpi"]
    mpi_opts = mpi_opts.__class__(mpi_opts)
    # Apply other options
    mpi_opts.set_opts(kw)
    # Extra options
    flags = mpi_opts.get_mpi_flags()
    flags = {} if flags is None else flags
    # Additional raw args
    rawargs = mpi_opts.get_mpi_args()
    args = [] if rawargs is None else [str(arg) for arg in rawargs]
    # Check if MPI is called for this command
    q_mpi = rc.get_MPI(j)
    # Name of MPI executable
    mpipre = rc.get_mpi_prefix(j=j)
    mpicmd = rc.get_mpicmd(j)
    mpicmd = rc.get_mpi_executable(j=j) or mpicmd
    # Exit if either criterion not met
    if (not q_mpi) or (not mpicmd):
        return []
    # Get number of MPI procs
    nproc = get_nproc(rc, j)
    perhost = rc.get_mpi_perhost(j)
    # Initialize command with prefix(es)
    mpipre = [] if mpipre is None else mpipre
    cmdi = mpipre if isinstance(mpipre, list) else [mpipre]
    # Add executable
    cmdi.append(mpicmd)
    # Check for hostfile
    host = rc.get_mpi_hostfile(j)
    if host:
        # If environment variable, get it explicitly
        if host.startswith("$"):
            host = os.environ.get(host.strip("$"))
    # Add hostfile if given
    append_cmd_if(cmdi, host, ['--hostfile', host])
    # If using mpt executable use special perhost arg
    if ("mpt" in mpicmd) or (mpicmd == "mpirun"):
        append_cmd_if(cmdi, perhost, ['-perhost', str(perhost)])
    else:
        append_cmd_if(cmdi, perhost, ['-npernode', str(perhost)])
    # Check for gpu number per host, number of MPI ranks, threads
    append_cmd_if(cmdi, nproc and (not perhost), ['-np', str(nproc)])
    # Add any generic options
    for k, v in flags.items():
        # Check type
        if isinstance(v, bool):
            # Check Boolean value
            if v:
                # Add '-$k' to command
                cmdi += ['-%s' % k]
        elif v is not None:
            # Convert to string, '-$k $v'
            cmdi += ['-%s' % k, '%s' % v]
    # Additional args (raw)
    append_cmd_if(cmdi, len(args), args)
    # Output
    return cmdi


def mpiexec_nogpu(
        opts: Optional[OptionsDict] = None, j: int = 0, **kw) -> list:
    r"""Create cmd to run MPI, ignoring "perhost" GPU settings

    :Call:
        >>> cmdi = mpiexec_nogpu(opts=None, j=0, **kw)
    :Inputs:
        *opts*: {``None``} | :class:`OptionsDict`
            Options interface, either global or "RunControl" section
        *j*: {``0``} | :class:`int`
            Phase number
    :Outputs:
        *cmdi*: :class:`list`\ [:class:`str`]
            System command created as list of strings
    :Versions:
        * 2025-04-04 ``@ddalle``: v1.0 (fork ``mpiexec()``)
    """
    # Isolate opts for "RunControl" section
    rc = isolate_subsection(opts, Options, ("RunControl",))
    # Get mpi options section
    mpi_opts = rc["mpi"]
    mpi_opts = mpi_opts.__class__(mpi_opts)
    # Apply other options
    mpi_opts.set_opts(kw)
    # Extra options
    flags = mpi_opts.get_mpi_flags()
    flags = {} if flags is None else flags
    # Additional raw args
    rawargs = mpi_opts.get_mpi_args()
    args = [] if rawargs is None else [str(arg) for arg in rawargs]
    # Check if MPI is called for this command
    q_mpi = rc.get_MPI(j)
    # Name of MPI executable
    mpipre = rc.get_mpi_prefix(j=j)
    mpicmd = rc.get_mpicmd(j)
    mpicmd = rc.get_mpi_executable(j=j) or mpicmd
    # Exit if either criterion not met
    if (not q_mpi) or (not mpicmd):
        return []
    # Get number of MPI procs
    nproc = get_nproc(rc, j)
    # Initialize command with prefix(es)
    mpipre = [] if mpipre is None else mpipre
    cmdi = mpipre if isinstance(mpipre, list) else [mpipre]
    # Add executable
    cmdi.append(mpicmd)
    # Check for hostfile
    host = rc.get_mpi_hostfile(j)
    if host:
        # If environment variable, get it explicitly
        if host.startswith("$"):
            host = os.environ.get(host.strip("$"))
    # Add hostfile if given
    append_cmd_if(cmdi, host, ["--hostfile", host])
    # Check for gpu number per host, number of MPI ranks, threads
    append_cmd_if(cmdi, nproc, ['-np', str(nproc)])
    # Add any generic options
    for k, v in flags.items():
        # Check type
        if isinstance(v, bool):
            # Check Boolean value
            if v:
                # Add '-$k' to command
                cmdi += ['-%s' % k]
        elif v is not None:
            # Convert to string, '-$k $v'
            cmdi += ['-%s' % k, '%s' % v]
    # Additional args (raw)
    append_cmd_if(cmdi, len(args), args)
    # Output
    return cmdi


# Function get aflr3 commands
def aflr3(opts=None, j=0, **kw):
    r"""Create AFLR3 system command as a list of strings

    :Call:
        >>> cmdi = aflr3(opts=None, j=0, **kw)
    :Inputs:
        *opts*: :class:`cape.options.Options`
            Options interface, either global, "RunControl", or "aflr3"
        *j*: :class:`int` | ``None``
            Phase number
        *blc*: :class:`bool`
            Whether or not to generate prism layers
        *blr*: :class:`float`
            Boundary layer stretching option
        *blds*: :class:`float`
            Initial surface stretching
        *cdfr*: :class:`float`
            Maximum geometric stretching
        *cdfs*: {``None``} | 0 <= :class:`float` <=10
            Distribution function exclusion zone
        *angblisimx*: :class:`float`
            Max BL intersection angle
    :Outputs:
        *cmdi*: :class:`list`\ [:class:`str`]
            System command created as list of strings
    :Versions:
        * 2016-04-04 ``@ddalle``: v1.0
        * 2023-08-18 ``@ddalle``: v1.1; use isolate_subsection()
    """
    # Isolate the inputs
    opts = isolate_subsection(opts, Options, ("RunControl", "aflr3"))
    # Apply extra options
    opts.set_opts(kw)
    # Get values
    fi = opts.get_aflr3_i(j)
    fo = opts.get_aflr3_o(j)
    blc = opts.get_aflr3_blc(j)
    blr = opts.get_aflr3_blr(j)
    bli = opts.get_aflr3_bli(j)
    blds = opts.get_aflr3_blds(j)
    grow = opts.get_aflr3_grow(j)
    cdfr = opts.get_aflr3_cdfr(j)
    cdfs = opts.get_aflr3_cdfs(j)
    nqual = opts.get_aflr3_nqual(j)
    mdsblf = opts.get_aflr3_mdsblf(j)
    angqbf = opts.get_aflr3_angqbf(j)
    angblisimx = opts.get_aflr3_angblisimx(j)
    # Extra options
    flags = opts.get_aflr3_flags()
    keys = opts.get_aflr3_keys()
    # Initialize command
    cmdi = ['aflr3']
    # Start with input and output files
    append_cmd_if_not_none(
        cmdi, fi, ['-i', fi],
        ValueError("Input file to aflr3 required"))
    # Check for output file
    append_cmd_if_not_none(
        cmdi, fo, ['-o', fo],
        ValueError("Output file for aflr3 required"))
    # Process boolean settings
    append_cmd_if(cmdi, blc, ['-blc'])
    # Process flag/value options
    append_cmd_if(cmdi, bli, ['-bli',  str(bli)])
    append_cmd_if(cmdi, blr, ['-blr',  str(blr)])
    append_cmd_if(cmdi, blds, ['-blds', str(blds)])
    append_cmd_if(cmdi, grow, ['-grow', str(grow)])
    # Process options that come with an equal sign
    append_cmd_if_not_none(cmdi, cdfs, ['cdfs=%s' % cdfs])
    append_cmd_if_not_none(cmdi, cdfr, ['cdfr=%s' % cdfr])
    append_cmd_if_not_none(cmdi, angqbf, ['angqbf=%s' % angqbf])
    append_cmd_if_not_none(cmdi, angblisimx, ['angblisimx=%s' % angblisimx])
    # Dash options that can be None
    append_cmd_if_not_none(cmdi, mdsblf, ['-mdsblf', str(mdsblf)])
    append_cmd_if_not_none(cmdi, nqual, [f'nqual={nqual}'])
    # Loop through flags
    for k, v in flags.items():
        # Check type
        if isinstance(v, bool):
            # Check Boolean value
            if v:
                # Add '-$k' to command
                cmdi += ['-%s' % k]
        elif v is not None:
            # Convert to string, '-$k $v'
            cmdi += ['-%s' % k, '%s' % v]
    # Loop though keys
    for k, v in keys.items():
        # Append to command
        append_cmd_if_not_none(cmdi, v, ['%s=%s' % (k, v)])
    # Output
    return cmdi


# Function to call verify
def intersect(opts=None, j=0, **kw):
    r"""Interface to Cart3D binary ``intersect``

    :Call:
        >>> cmd = cape.cmd.intesect(opts=None, **kw)
    :Inputs:
        *opts*: :class:`cape.options.Options`
            Options interface
        *i*: :class:`str`
            Name of input ``tri`` file
        *o*: :class:`str`
            Name of output ``tri`` file
    :Outputs:
        *cmd*: :class:`list` (:class:`str`)
            Command split into a list of strings
    :Versions:
        * 2015-02-13 ``@ddalle``: v1.0
        * 2023-08-18 ``@ddalle``: v1.1; use isolate_subsection()
    """
    # Isolate options
    opts = isolate_subsection(opts, Options, ("RunControl", "intersect"))
    # Apply other options
    opts.set_opts(kw)
    # Get settings
    fin = opts.get_opt('i', j)
    fout = opts.get_opt('o', j)
    cutout = opts.get_opt('cutout', j)
    ascii = opts.get_opt('ascii', j)
    v = opts.get_opt('v', j)
    T = opts.get_opt('T', j)
    iout = opts.get_opt('intersect', j)
    # Build the command.
    cmdi = ['intersect', '-i', fin, '-o', fout]
    # Type option
    append_cmd_if(cmdi, cutout, ['-cutout', str(cutout)])
    append_cmd_if(cmdi, ascii, ['-ascii'])
    append_cmd_if(cmdi, v, ['-v'])
    append_cmd_if(cmdi, T, ['-T'])
    append_cmd_if(cmdi, iout, ['-intersections'])
    # Output
    return cmdi


# Function to call comp2tri
def comp2tri(opts=None, j=0, **kw):
    r"""Interface to Cart3D binary ``intersect``

    :Call:
        >>> cmd = cape.cmd.intesect(opts=None, **kw)
    :Inputs:
        *opts*: :class:`cape.options.Options`
            Options interface
        *i*: :class:`str`
            Name of input ``tri`` file
        *o*: :class:`str`
            Name of output ``tri`` file
    :Outputs:
        *cmd*: :class:`list` (:class:`str`)
            Command split into a list of strings
    :Versions:
        * 2015-02-13 ``@ddalle``: v1.0
        * 2023-08-18 ``@ddalle``: v1.1; use isolate_subsection()
    """
    # Isolate options
    opts = isolate_subsection(opts, Options, ("RunControl", "comp2tri"))
    # Apply other options
    opts.set_opts(kw)
    # Get file names, in and out
    ifiles = opts.get_opt('i', j)
    ofile = opts.get_opt('o', j)
    # Get other options
    ascii = opts.get_opt("ascii", j)
    inflate = opts.get_opt("inflate", j)
    trix = opts.get_opt("trix", j)
    keepcomps = opts.get_opt("keepComps", j)
    maketags = opts.get_opt("makeGMPtags", j)
    tagoffset = opts.get_opt("gmpTagOffset", j)
    gmp2comp = opts.get_opt("gmp2comp", j)
    config = opts.get_opt("config", j)
    dp = opts.get_opt("dp", j)
    v = opts.get_opt('v', j)
    # Build the command.
    cmdi = ['comp2tri', '-o', ofile]
    # Type options
    append_cmd_if(cmdi, ascii, ['-ascii'])
    append_cmd_if(cmdi, trix, ['-trix'])
    append_cmd_if(cmdi, v, ['-v'])
    append_cmd_if(cmdi, inflate, ['-inflate'])
    append_cmd_if(cmdi, keepcomps, ['-keepComps'])
    append_cmd_if(cmdi, maketags, ['-makeGMPtags'])
    append_cmd_if(cmdi, tagoffset, ['-gmpTagOffset', str(tagoffset)])
    append_cmd_if(cmdi, gmp2comp, ['-gmp2comp'])
    append_cmd_if(cmdi, config, ['-config'])
    append_cmd_if(cmdi, dp, ['-dp'])
    # Add list of input files
    cmdi.extend(ifiles)
    # Output
    return cmdi


# Function to call verify
def verify(opts=None, **kw):
    r"""Generate command for Cart3D executable ``verify``

    :Call:
        >>> cmdi = cape.cmd.verify(opts=None, **kw)
    :Inputs:
        *opts*: :class:`cape.options.Options`
            Options interface
        *i*: {``"Components.i.tri"`` | :class:`str`
            Name of *tri* file to test
        *ascii*: {``True``} | ``False``
            Flag to consider input file ASCII
        *binary*: ``True`` | {``False``}
            Flag to consider input file binary
    :Outputs:
        *cmdi*: :class:`list` (:class:`str`)
            Command split into a list of strings
    :Versions:
        * 2015-02-13 ``@ddalle``: v1.0
        * 2023-08-18 ``@ddalle``: v1.1; use isolate_subsection()
    """
    # Isolate options
    opts = isolate_subsection(opts, Options, ("RunControl", "verify"))
    # Apply keyword options
    opts.set_opts(kw)
    # Get settings
    ftri = opts.get_opt('i')
    binry = opts.get_opt('binary')
    ascii = opts.get_opt('ascii')
    # Build the command.
    cmdi = ['verify', ftri]
    # Options
    append_cmd_if(cmdi, binry, ['-binary'])
    append_cmd_if(cmdi, ascii, ['-ascii'])
    # Output
    return cmdi


# Infix a phase number into a file name
def infix_phase(fname: str, j: int = 0) -> str:
    r"""Add a two-digit phase number to file name before final extension

    :Call:
        >>> fnamej = infix_phase(fname, j)
    :Inputs:
        *fname*: :class:`str`
            Oritinal name of file
        *j*: {``0``} | :class:`int`
            Phase number
    :Outputs:
        *fnamej*: :class:`str`
            Infixed file name
    :Examples:
        >>> infix_phase("fun3d.nml", 2)
        'fun3d.02.nml'
    :Versions:
        * 2024-10-10 ``@ddalle``: v1.0
    """
    # Split file name into parts
    parts = fname.rsplit('.', 1)
    # Reconstruct with infix
    return f"{parts[0]}.{j:02d}.{parts[-1]}"


def get_nproc(rc: Options, j: int = 0) -> int:
    r"""Get the number of processes available or specified

    :Call:
        >>> nproc = get_nproc(rc, j=0)
    :Inputs:
        *rc*: :class:`Options`
            Options interface, or *RunControl* section
        *j*: {``0``} | :class:`int`
            Phase number
    :Outputs:
        *nproc*: :class:`int`
            Number of CPUs or GPUs
    :Versions:
        * 2024-08-12 ``@ddalle``: v1.0
    """
    # Get number of MPI procs
    nprocdef = get_mpi_procs()
    # Check for explicit number of processes
    nproc = rc.get_nProc(j)
    nproc = rc.get_mpi_np(j, vdef=nproc)
    # Check type of user-specified *nProc*
    if isinstance(nproc, float):
        # User specifies a fraction of procs to use
        nproc = int(nproc * nprocdef)
    elif isinstance(nproc, int) and nproc < 0:
        # User specified number to *omit*
        nproc = nprocdef + nproc
    # Override any explicit "null" (None) values
    nproc = nprocdef if nproc is None else nproc
    # Output with overrides
    return nproc


# Get default number of CPUs
def get_mpi_procs() -> int:
    r"""Estimate number of MPI processes available

    This works by reading PBS or Slurm environment variables

    :Call:
        >>> nproc = get_mpi_procs()
    :Outputs:
        *nproc*: :class:`int`
            Number of CPUs indicated by environment variables
    :Versions:
        * 2024-08-08 ``@ddalle``: v1.0
    """
    # Check environment variables
    if os.environ.get("SLURM_JOB_ID"):
        # Use Slurm environment variables
        return _get_mpi_procs_slurm()
    else:
        # Use PBS (defaults to OMP_NUM_THREADS)
        return _get_mpi_procs_pbs()


# Get default number of cores for MPI
def _get_mpi_procs_pbs() -> int:
    # Check node file
    nodefile = os.environ.get("PBS_NODEFILE", "")
    # Check if file exists
    if os.path.isfile(nodefile):
        # Read it and count lines
        return len(open(nodefile).readlines())
    else:
        # Use OMP_NUM_THREADS as backup
        return int(os.environ.get("OMP_NUM_THREADS", "1"))


# Get default number of of cores for Slurm
def _get_mpi_procs_slurm() -> Union[int, str]:
    # Get number of MPI tasks
    ntask_raw = os.environ.get("SLURM_NTASKS", "1")
    # Convert to integer
    try:
        ntask = int(ntask_raw)
    except ValueError:
        ntask = ntask_raw
    # Output
    return ntask

