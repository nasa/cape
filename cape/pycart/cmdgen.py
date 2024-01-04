"""
:mod:`cape.pycart.cmdgen`: Create commands for Cart3D executables
==================================================================

This module creates system commands as lists of strings for executables
or scripts for Cart3D.  It is closely tied to :mod:`cape.pycart.bin`,
which actually runs the executables.

Commands are created in the form of a list of strings.  This is the
format used in the built-in module :mod:`subprocess` and also with
:func:`cape.bin.calli`. As a very simple example, the system command
``"ls -lh"`` becomes the list ``["ls", "-lh"]``.

Calls to the main Cart3D flow solver via ``flowCart`` or
``mpix_flowCart`` include a great deal of command-line options. This
module simplifies the creation of this command by determining those
inputs from a :class:`cape.pycart.options.Options` or
:class:`cape.pycart.options.runctlopts.RunControlOpts` instance.

:See also:
    * :mod:`cape.cmd`
    * :mod:`cape.bin`
    * :mod:`cape.pycart.bin`

"""

# Standard library modules
import os.path

# Local imports
from .options import Options
from ..cfdx.cmdgen import isolate_subsection, append_cmd_if


# Function to call cubes.
def cubes(opts=None, j=0, **kw):
    r"""Interface to Cart3D script ``cubes``

    :Call:
        >>> cmd = cubes(opts=opts, j=0, **kw)
    :Inputs:
        *j*: {``0``} | :class:`int`
            Phase number
        *opts*: :class:`Options` | :class:`dict`
            Options interface, run control options instance, raw dict
        *maxR*: :class:`int`
            Number of refinements to make
        *reorder*: :class:`bool`
            Whether or not to reorder mesh
        *a*: :class:`int`
            `cubes` angle criterion
        *b*: :class:`int`
            Number of buffer layers
        *sf*: :class:`int`
            Number of extra refinements around sharp edges
    :Outputs:
        *cmd*: :class:`list`\ [:class:`str`]
            Command split into a list of strings
    :Versions:
        * 2014-06-30 ``@ddalle``: v1.0
        * 2014-09-02 ``@ddalle``: v2.0; new options paradigm
        * 2014-09-10 ``@ddalle``: v3.0; two modules: ``cmd``, ``bin``
        * 2014-12-02 ``@ddalle``: v3.1; add kwargs, add *sf*
        * 2023-08-21 ``@ddalle``: v4.0; elim *cntl* arg, use isolate_()
    """
    # Isolate options
    opts = isolate_subsection(opts, Options, ("RunControl",))
    # Apply kwargs
    opts.set_opts(kw)
    # Get values
    maxR = opts.get_maxR(j)
    reorder = opts.get_reorder(j)
    cubes_a = opts.get_cubes_a(j)
    cubes_b = opts.get_cubes_b(j)
    sf = opts.get_sf(j)
    # Initialize command
    cmd = ['cubes', '-pre', 'preSpec.c3d.cntl']
    # Add optional args
    append_cmd_if(cmd, maxR, ['-maxR', str(maxR)])
    append_cmd_if(cmd, reorder, ['-reorder'])
    append_cmd_if(cmd, cubes_a, ['-a', str(cubes_a)])
    append_cmd_if(cmd, cubes_b, ['-b', str(cubes_b)])
    append_cmd_if(cmd, sf, ['-sf', str(sf)])
    # Return the command.
    return cmd


# Function to call mgPrep
def mgPrep(opts=None, j=0, **kw):
    r"""Interface to Cart3D executable ``mgPrep``

    :Call:
        >>> cmd = mgPrep(opts=None, j=0, **kw)
    :Inputs:
        *opts*: :class:`Options` | :class:`dict`
            Options interface
        *j*: {``0``} | :class:`int`
            Phase number
        *mg_fc*: :class:`int`
            Number of multigrid levels to prepare
    :Outputs:
        *cmd*: :class:`list`\ [:class:`str`]
            Command split into a list of strings
    :Versions:
        * 2014-09-02 ``@ddalle``: v1.0
        * 2023-08-21 ``@ddalle``: v1.1; use isolate_subsection()
    """
    # Isolate options
    opts = isolate_subsection(opts, Options, ("RunControl",))
    # Apply kwargs
    opts.set_opts(kw)
    # Get option values
    mg_fc = opts.get_mg_fc(j)
    pmg = opts.get_pmg(j)
    # Initialize command.
    cmd = ['mgPrep']
    # Add optional inputs
    append_cmd_if(cmd, mg_fc, ['-n', str(mg_fc)])
    append_cmd_if(cmd, pmg, ['-pmg'])
    # Return the command
    return cmd


# Function to call mgPrep
def autoInputs(opts=None, j=0, **kw):
    r"""Interface to Cart3D executable ``autoInputs``

    :Call:
        >>> cmd = autoInputs(opts=opts, j=0, **kw)
    :Inputs:
        *opts*: :class:`Options` | :class:`dict`
            Options interface
        *j*: {``0``} | :class:`int`
            Phase number
        *r*: :class:`int`
            Mesh radius
        *ftri*: :class:`str`
            Name of surface triangulation file
        *maxR*: :class:`int`
            Number of refinements to make
    :Outputs:
        *cmd*: :class:`list`\ [:class:`str`]
            Command split into a list of strings
    :Versions:
        * 2014-09-02 ``@ddalle``: v1.0
        * 2023-08-21 ``@ddalle``: v1.1; use isolate_subsection()
    """
    # Isolate options
    opts = isolate_subsection(opts, Options, ("RunControl",))
    # Apply kwargs
    opts.set_opts(kw)
    # Get option values
    r = opts.get_autoInputs_r(j)
    maxR = opts.get_autoInputs_maxR(j)
    nDiv = opts.get_autoInputs_nDiv(j)
    # Check for the appropriate tri file type
    if os.path.isfile('Components.i.tri'):
        # Intersected surface
        ftri = 'Components.i.tri'
    else:
        # Surface will be intersected later
        ftri = 'Components.tri'
    # Initialize command
    cmd = ['autoInputs']
    # Add optional args
    append_cmd_if(cmd, r, ['-r', str(r)])
    append_cmd_if(cmd, ftri, ['-t', ftri])
    append_cmd_if(cmd, maxR, ['-maxR', str(maxR)])
    append_cmd_if(cmd, nDiv, ['-nDiv', str(nDiv)])
    # Return the command
    return cmd


# Function to create flowCart command
def flowCart(opts=None, j=0, **kw):
    r"""Interface to Cart3D executable ``flowCart``

    :Call:
        >>> cmdi = flowCart(opts=None, i=0, **kw)
    :Inputs:
        *opts*: :class:`Options` | :class:`dict`
            Options interface
        *j*: {``0``} | :class:`int`
            Phase number
        *n*: :class:`int`
            Iteration number from which to start (affects ``-N`` setting)
        *clic*: :class:`bool`
            Whether or not to create ``Components.i.triq`` file
        *mpi_fc*: :class:`bool`
            Whether or not to use MPI version
        *it_fc*: :class:`int`
            Number of iterations to run ``flowCart``
        *mg_fc*: :class:`int`
            Number of multigrid levels to use
        *it_avg*: :class:`int`
            Iterations between averaging break; overrides *it_fc*
        *it_start*: :class:`int`
            Startup iterations before starting averaging
        *fmg*: :class:`bool`
            Whether to use full multigrid (adds ``-no_fmg`` flag if ``False``)
        *pmg*: :class:`bool`
            Whether or not to use poly multigrid
        *cfl*: :class:`float`
            Nominal CFL number
        *cflmin*: :class:`float`
            Minimum CFL number allowed for problem cells or iterations
        *tm*: :class:`bool`
            Whether or not to use second-order cut cells
        *buffLim*: :class:`bool`
            Whether or not to use buffer limiting to smear shocks
        *binaryIO*: :class:`bool`
            Whether or not to use binary format for input/output files
        *limiter*: :class:`int`
            Limiter index
        *y_is_spanwise*: :class:`bool`
            Whether or not to consider angle of attack in *z* direction
        *nProc*: :class:`int`
            Number of threads to use for ``flowCart`` operation
        *buffLim*: :class:`bool`
            Whether or not to use buffer limits for strong off-body shocks
    :Outputs:
        *cmdi*: :class:`list`\ [:class:`str`]
            Command split into a list of strings
    :Versions:
        * 2014-09-07 ``@ddalle``: v1.0
        * 2023-08-21 ``@ddalle``: v1.1; use isolate_subsection()
    """
    # Isolate options
    opts = isolate_subsection(opts, Options, ("RunControl",))
    # Offset iterations
    n = kw.pop('n', 0)
    # Apply kwargs
    opts.set_opts(kw)
    # Get option values
    mpi_fc = opts.get_MPI(j)
    it_fc = opts.get_it_fc(j)
    it_avg = opts.get_it_avg(j)
    it_sub = opts.get_it_sub(j)
    limiter = opts.get_limiter(j)
    y_span = opts.get_y_is_spanwise(j)
    binIO = opts.get_binaryIO(j)
    tecO = opts.get_tecO(j)
    cfl = opts.get_cfl(j)
    mg_fc = opts.get_mg_fc(j)
    fmg = opts.get_fmg(j)
    tm = opts.get_tm(j)
    nProc = opts.get_nProc(j)
    mpicmd = opts.get_mpicmd(j)
    buffLim = opts.get_buffLim(j)
    td_fc = opts.get_unsteady(j)
    dt = opts.get_dt(j)
    chkptTD = opts.get_checkptTD(j)
    vizTD = opts.get_vizTD(j)
    clean = opts.get_fc_clean(j)
    nstats = opts.get_fc_stats(j)
    clic = opts.get_clic(j)
    # Check for override *it_fc* iteration count.
    if it_avg:
        it_fc = it_avg
    # Initialize command.
    if mpi_fc:
        # Use mpi_flowCart but not unsteady
        cmd = [mpicmd, '-np', str(nProc), 'mpix_flowCart', '-his']
    else:
        # Use single-node flowCart.
        cmd = ['flowCart', '-his']
    # Check 'clic' option (creates ``Components.i.triq``)
    append_cmd_if(cmd, clic, ['-clic'])
    # Check for restart
    if (j > 0) or (n > 0):
        cmd += ['-restart']
    # Number of steps
    if td_fc:
        # Increase iteration count by the number of previously computed iters.
        it_fc += n
        # Number of steps, subiterations (mg cycles), and time step
        append_cmd_if(cmd, it_fc, ['-nSteps', str(it_fc)])
        append_cmd_if(cmd, it_sub, ['-N', str(it_sub)])
        append_cmd_if(cmd, dt, ['-dt', str(dt)])
        # Other unsteady options
        append_cmd_if(cmd, chkptTD, ['-checkptTD', str(chkptTD)])
        append_cmd_if(cmd, vizTD, ['-vizTD', str(vizTD)])
        append_cmd_if(cmd, clean, ['-clean'])
        append_cmd_if(cmd, nstats, ['-stats', str(nstats)])
    else:
        # Increase the iteration count
        it_fc += n
        # Multigrid cycles
        append_cmd_if(cmd, it_fc, ['-N', str(it_fc)])
    # Add options
    append_cmd_if(cmd, y_span, ['-y_is_spanwise'])
    append_cmd_if(cmd, limiter, ['-limiter', str(limiter)])
    append_cmd_if(cmd, tecO, ['-T'])
    append_cmd_if(cmd, cfl, ['-cfl', str(cfl)])
    append_cmd_if(cmd, mg_fc, ['-mg', str(mg_fc)])
    append_cmd_if(cmd, fmg, ['-no_fmg'])
    append_cmd_if(cmd, buffLim, ['-buffLim'])
    append_cmd_if(cmd, binIO, ['-binaryIO'])
    # Process and translate the cut-cell gradient variable
    if (tm is not None) and tm:
        # Second-order cut cells
        cmd += ['-tm', '1']
    elif (tm is not None) and (not tm):
        # First-order cut cells
        cmd += ['-tm', '0']
    # Output
    return cmd

