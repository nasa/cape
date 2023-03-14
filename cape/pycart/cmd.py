"""
This module creates system commands as lists of strings for executable binaries
or scripts for Cart3D.  It is closely tied to :mod:`cape.pycart.bin`, which actually
runs the executables.

Commands are created in the form of a list of strings.  This is the format used
in the built-in module :mod:`subprocess` and also with :func:`cape.bin.calli`. 
As a very simple example, the system command ``"ls -lh"`` becomes the list
``["ls", "-lh"]``.

Calls to the main Cart3D flow solver via ``flowCart`` or ``mpix_flowCart``
include a great deal of command-line options.  This module simplifies the
creation of this command by determining those inputs from a
:class:`pyCart.options.Options` or
:class:`pyCart.options.runControl.RunControl` object.

:See also:
    * :mod:`cape.cmd`
    * :mod:`cape.bin`
    * :mod:`cape.pycart.bin`

"""

# Standard library modules
import os.path

# CAPE modules
from cape.cfdx.cmd import *

# Local modules
from .util import GetTecplotCommand

# Function to call cubes.
def cubes(cart3d=None, opts=None, j=0, **kw):
    """
    Interface to Cart3D script `cubes`
    
    :Call:
        >>> cmd = pyCart.cmd.cubes(cart3d, j=0)
        >>> cmd = pyCart.cmd.cubes(opts=opts, j=0)
        >>> cmd = pyCart.cmd.cubes(opts=rc, j=0)
        >>> cmd = pyCart.cmd.cubes(maxR=11, reorder=True, **kwargs)
    :Inputs:
        *cart3d*: :class:`cape.pycart.cntl.Cntl`
            Global pyCart settings instance
        *j*: {``0``} | :class:`int`
            Phase number
        *opts*: :class:`pyCart.options.Options`
            Options interface
        *rc*: :class:`pyCart.options.runControl.RunControl`
            Options interface
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
        * 2014-06-30 ``@ddalle``: First version
        * 2014-09-02 ``@ddalle``: Rewritten with new options paradigm
        * 2014-09-10 ``@ddalle``: Split into cmd and bin
        * 2014-12-02 ``@ddalle``: Moved to keyword arguments and added *sf*
    """
    # Check cart3d input
    if cart3d is not None:
        # Extract options
        opts = cntl.opts
    # Check input method
    if opts is not None:
        # Apply values
        maxR    = opts.get_maxR(j)
        reorder = opts.get_reorder(j)
        cubes_a = opts.get_cubes_a(j)
        cubes_b = opts.get_cubes_b(j)
        sf      = opts.get_sf(j)
    else:
        # Get keyword arguments
        maxR    = kw.get('maxR', 11)
        reorder = kw.get('reorder', True)
        cubes_a = kw.get('a', 10)
        cubes_b = kw.get('b', 2)
        sf      = kw.get('sf', 0)
    # Initialize command
    cmd = ['cubes', '-pre', 'preSpec.c3d.cntl']
    # Add options.
    if maxR:    cmd += ['-maxR', str(maxR)]
    if reorder: cmd += ['-reorder']
    if cubes_a: cmd += ['-a', str(cubes_a)]
    if cubes_b: cmd += ['-b', str(cubes_b)]
    if sf:      cmd += ['-sf', str(sf)]
    # Return the command.
    return cmd
    
# Function to call mgPrep
def mgPrep(cart3d=None, opts=None, j=0, **kw):
    """
    Interface to Cart3D script `mgPrep`
    
    :Call:
        >>> cmd = pyCart.cmd.mgPrep(cart3d, j=0)
        >>> cmd = pyCart.cmd.mgPrep(opts=opts, j=0)
        >>> cmd = pyCart.cmd.mgPrep(opts=rc, j=0)
        >>> cmd = pyCart.cmd.mgPrep(**kw)
    :Inputs:
        *cart3d*: :class:`cape.pycart.cntl.Cntl`
            Global pyCart settings instance
        *j*: {``0``} | :class:`int`
            Phase number
        *opts*: :class:`pyCart.options.Options`
            Options interface
        *rc*: :class:`pyCart.options.runControl.RunControl`
            Options interface
        *mg_fc*: :class:`int`
            Number of multigrid levels to prepare
    :Outputs:
        *cmd*: :class:`list`\ [:class:`str`]
            Command split into a list of strings
    :Versions:
        * 2014-09-02 ``@ddalle``: First version
    """
    # Check cart3d input.
    if cart3d is not None:
        # Extract the options
        opts = cntl.opts
    # Check input type
    if opts is not None:
        # Apply values
        mg_fc = opts.get_mg_fc(j)
        pmg   = opts.get_pmg(j)
    else:
        # Get values
        mg_fc = kw.get('mg_fc', 3)
        pmg   = kw.get('pmg', 0)
    # Initialize command.
    cmd = ['mgPrep']
    # Add options.
    if mg_fc: cmd += ['-n', str(mg_fc)]
    if pmg:   cmd += ['-pmg']
    # Return the command
    return cmd
    
# Function to call mgPrep
def autoInputs(cart3d=None, opts=None, ftri='Components.i.tri', j=0, **kw):
    """Interface to Cart3D script ``autoInputs``
    
    :Call:
        >>> cmd = pyCart.cmd.autoInputs(cart3d, j=0)
        >>> cmd = pyCart.cmd.autoInputs(opts=opts, j=0)
        >>> cmd = pyCart.cmd.autoInputs(opts=rc, j=0)
        >>> cmd = pyCart.cmd.autoInputs(ftri='Components.i.tri', **kw)
    :Inputs:
        *cart3d*: :class:`cape.pycart.cntl.Cntl`
            Global pyCart settings instance
        *j*: {``0``} | :class:`int`
            Phase number
        *opts*: :class:`pyCart.options.Options`
            Options interface
        *rc*: :class:`pyCart.options.runControl.RunControl`
            Options interface
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
        * 2014-09-02 ``@ddalle``: First version
    """
    # Check cart3d input.
    if cart3d is not None:
        # Extract options
        opts = cntl.opts
    # Check input type
    if opts is not None:
        # Apply values
        r     = opts.get_r(j)
        maxR  = opts.get_maxR(j)
        nDiv  = opts.get_nDiv(j)
        # Check for the appropriate tri file type.
        if os.path.isfile('Components.i.tri'):
            # Intersected surface
            ftri = 'Components.i.tri'
        else:
            # Surface will be intersected later
            ftri = 'Components.tri'
    else:
        # Get values from keyword arguments
        r     = kwagrs.get('r',    8)
        maxR  = kwargs.get('maxR', 10)
        nDiv  = kwargs.get('nDiv', 4)
    # Initialize command.
    cmd = ['autoInputs']
    # Add options.
    if r:    cmd += ['-r', str(r)]
    if ftri: cmd += ['-t', ftri]
    if maxR: cmd += ['-maxR', str(maxR)]
    if nDiv: cmd += ['-nDiv', str(nDiv)]
    # Return the command.
    return cmd
    

# Function to create flowCart command
def flowCart(cart3d=None, fc=None, i=0, **kwargs):
    """Interface to Cart3D binary ``flowCart``
    
    :Call:
        >>> cmdi = pyCart.cmd.flowCart(cart3d, i=0)
        >>> cmdi = pyCart.cmd.flowCart(fc=None, i=0)
        >>> cmdi = pyCart.cmd.flowCart(**kwargs)
    :Inputs:
        *cart3d*: :class:`cape.pycart.cntl.Cntl`
            Global pyCart settings instance
        *fc*: :class:`pyCart.options.flowCart.flowCart`
            Direct reference to ``cart3d.opts['flowCart']``
        *i*: :class:`int`
            Run index (restart if *i* is greater than *0*)
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
        * 2014-09-07 ``@ddalle``: First version
    """
    # Check for cart3d input
    if cart3d is not None:
        # Get values from internal settings.
        mpi_fc  = cntl.opts.get_MPI(i)
        it_fc   = cntl.opts.get_it_fc(i)
        it_avg  = cntl.opts.get_it_avg(i)
        it_sub  = cntl.opts.get_it_sub(i)
        limiter = cntl.opts.get_limiter(i)
        y_span  = cntl.opts.get_y_is_spanwise(i)
        binIO   = cntl.opts.get_binaryIO(i)
        tecO    = cntl.opts.get_tecO(i)
        cfl     = cntl.opts.get_cfl(i)
        cflmin  = cntl.opts.get_cflmin(i)
        mg_fc   = cntl.opts.get_mg_fc(i)
        fmg     = cntl.opts.get_fmg(i)
        tm      = cntl.opts.get_tm(i)
        nProc   = cntl.opts.get_nProc(i)
        mpicmd  = cntl.opts.get_mpicmd(i)
        buffLim = cntl.opts.get_buffLim(i)
        td_fc   = cntl.opts.get_unsteady(i)
        dt      = cntl.opts.get_dt(i)
        chkptTD = cntl.opts.get_checkptTD(i)
        vizTD   = cntl.opts.get_vizTD(i)
        clean   = cntl.opts.get_fc_clean(i)
        nstats  = cntl.opts.get_fc_stats(i)
        clic    = cntl.opts.get_clic(i)
    elif fc is not None:
        # Get values from direct settings.
        mpi_fc  = fc.get_MPI(i)
        it_fc   = fc.get_it_fc(i)
        it_avg  = fc.get_it_avg(i)
        it_sub  = fc.get_it_sub(i)
        limiter = fc.get_limiter(i)
        y_span  = fc.get_y_is_spanwise(i)
        binIO   = fc.get_binaryIO(i)
        tecO    = fc.get_tecO(i)
        cfl     = fc.get_cfl(i)
        cflmin  = fc.get_cflmin(i)
        mg_fc   = fc.get_mg_fc(i)
        fmg     = fc.get_fmg(i)
        tm      = fc.get_tm(i)
        nProc   = fc.get_nProc(i)
        mpicmd  = fc.get_mpicmd(i)
        buffLim = fc.get_buffLim(i)
        td_fc   = fc.get_unsteady(i)
        dt      = fc.get_dt(i)
        chkptTD = fc.get_checkptTD(i)
        vizTD   = fc.get_vizTD(i)
        clean   = fc.get_fc_clean(i)
        nstats  = fc.get_fc_stats(i)
        clic    = fc.get_clic(i)
    else:
        # Get values from keyword arguments
        mpi_fc  = kwargs.get('MPI', False)
        it_fc   = kwargs.get('it_fc', 200)
        it_avg  = kwargs.get('it_avg', 0)
        it_sub  = kwargs.get('it_sub', 10)
        limiter = kwargs.get('limiter', 2)
        y_span  = kwargs.get('y_is_spanwise', True)
        binIO   = kwargs.get('binaryIO', False)
        tecO    = kwargs.get('tecO', False)
        cfl     = kwargs.get('cfl', 1.1)
        cflmin  = kwargs.get('cflmin', 0.8)
        mg_fc   = kwargs.get('mg_fc', 3)
        fmg     = kwargs.get('fmg', True)
        tm      = kwargs.get('tm', None)
        nProc   = kwargs.get('nProc', 8)
        mpicmd  = kwargs.get('mpicmd', 'mpiexec')
        buffLim = kwargs.get('buffLim', False)
        td_fc   = kwargs.get('unsteady', False)
        dt      = kwargs.get('dt', 0.1)
        chkptTD = kwargs.get('checkptTD', None)
        vizTD   = kwargs.get('vizTD', None)
        clean   = kwargs.get('clean', False)
        nstats  = kwargs.get('stats', 0)
        clic    = kwargs.get('clic', True)
    # Check for override *it_fc* iteration count.
    if it_avg: it_fc = it_avg
    # Initialize command.
    if mpi_fc:
        # Use mpi_flowCart but not unsteady
        cmd = [mpicmd, '-np', str(nProc), 'mpix_flowCart', '-his']
    else:
        # Use single-node flowCart.
        cmd = ['flowCart', '-his']
    # Check 'clic' option (creates ``Components.i.triq``)
    if clic: cmd += ['-clic']
    # Offset iterations
    n = kwargs.get('n', 0)
    # Check for restart
    if (i>0) or (n>0): cmd += ['-restart']
    # Number of steps
    if td_fc:
        # Increase iteration count by the number of previously computed iters.
        it_fc += n
        # Number of steps, subiterations (mg cycles), and time step
        if it_fc:  cmd += ['-nSteps', str(it_fc)]
        if it_sub: cmd += ['-N', str(it_sub)]
        if dt:     cmd += ['-dt', str(dt)]
        # Other unsteady options.
        if chkptTD: cmd += ['-checkptTD', str(chkptTD)]
        if vizTD:   cmd += ['-vizTD', str(vizTD)]
        if clean:   cmd += ['-clean']
        if nstats:  cmd += ['-stats', str(nstats)]
    else:
        # Increase the iteration count.
        it_fc += n
        # Multigrid cycles
        if it_fc:   cmd += ['-N', str(it_fc)]
    # Add options
    if y_span:  cmd += ['-y_is_spanwise']
    if limiter: cmd += ['-limiter', str(limiter)]
    if tecO:    cmd += ['-T']
    if cfl:     cmd += ['-cfl', str(cfl)]
    if mg_fc:   cmd += ['-mg', str(mg_fc)]
    if not fmg: cmd += ['-no_fmg']
    if buffLim: cmd += ['-buffLim']
    if binIO:   cmd += ['-binaryIO']
    # Process and translate the cut-cell gradient variable.
    if (tm is not None) and tm:
        # Second-order cut cells
        cmd += ['-tm', '1']
    elif (tm is not None) and (not tm):
        # First-order cut cells
        cmd += ['-tm', '0']
    # Output
    return cmd
# def flowCart

