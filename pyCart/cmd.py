"""
Create command strings for Cart3D binaries: :mod:`pyCart.cmd`
=============================================================

This module contains direct interfaces to Cart3D binaries so that they can be
called from Python.  Settings for the binaries that are usually specified as
command-line options are either specified as keyword arguments or inherited from
a :class:`pyCart.cart3d.Cart3d` instance.
"""

# File checking.
import os.path

# Function to call cubes.
def cubes(cart3d=None, **kw):
    """
    Interface to Cart3D script `cubes`
    
    :Call:
        >>> cmd = pyCart.cmd.cubes(cart3d)
        >>> cmd = pyCart.cmd.cubes(maxR=11, reorder=True, **kwargs)
    :Inputs:
        *cart3d*: :class:`pyCart.cart3d.Cart3d`
            Global pyCart settings instance
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
        *cmd*: :class:`list` (:class:`str`)
            Command split into a list of strings
    :Versions:
        * 2014-06-30 ``@ddalle``: First version
        * 2014-09-02 ``@ddalle``: Rewritten with new options paradigm
        * 2014-09-10 ``@ddalle``: Split into cmd and bin
        * 2014-12-02 ``@ddalle``: Moved to keyword arguments and added *sf*
    """
    # Check cart3d input.
    if cart3d is not None:
        # Apply values
        maxR    = cart3d.opts.get_maxR(0)
        reorder = cart3d.opts.get_reorder(0)
        cubes_a = cart3d.opts.get_cubes_a(0)
        cubes_b = cart3d.opts.get_cubes_b(0)
        sf      = cart3d.opts.get_sf(0)
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
def mgPrep(cart3d=None, **kw):
    """
    Interface to Cart3D script `mgPrep`
    
    :Call:
        >>> cmd = pyCart.cmd.mgPrep(cart3d)
        >>> cmd = pyCart.cmd.mgPrep(**kw)
    :Inputs:
        *cart3d*: :class:`pyCart.cart3d.Cart3d`
            Global pyCart settings instance
        *mg_fc*: :class:`int`
            Number of multigrid levels to prepare
    :Outputs:
        *cmd*: :class:`list` (:class:`str`)
            Command split into a list of strings
    :Versions:
        * 2014-09-02 ``@ddalle``: First version
    """
    # Check cart3d input.
    if cart3d is not None:
        # Apply values
        mg_fc = cart3d.opts.get_mg_fc(0)
        pmg   = cart3d.opts.get_pmg(0)
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
def autoInputs(cart3d=None, r=8, ftri='Components.i.tri', **kw):
    """
    Interface to Cart3D script `mgPrep`
    
    :Call:
        >>> cmd = pyCart.cmd.autoInputs(cart3d)
        >>> cmd = pyCart.cmd.autoInputs(r=8, ftri='components.i.tri', **kw)
    :Inputs:
        *cart3d*: :class:`pyCart.cart3d.Cart3d`
            Global pyCart settings instance
        *r*: :class:`int`
            Mesh radius
        *ftri*: :class:`str`
            Name of surface triangulation file
        *maxR*: :class:`int`
            Number of refinements to make
    :Outputs:
        *cmd*: :class:`list` (:class:`str`)
            Command split into a list of strings
    :Versions:
        * 2014-09-02 ``@ddalle``: First version
    """
    # Check cart3d input.
    if cart3d is not None:
        # Apply values
        r     = cart3d.opts.get_r(0)
        maxR  = cart3d.opts.get_maxR(0)
        nDiv  = cart3d.opts.get_nDiv(0)
        # Check for the appropriate tri file type.
        if os.path.isfile('Components.i.tri'):
            # Intersected surface
            ftri = 'Components.i.tri'
        else:
            # Surface will be intersected later
            ftri = 'Components.tri'
    else:
        # Get values from keyword arguments
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
    
# Function to call verify
def verify(ftri='Components.i.tri'):
    """Interface to Cart3D binary `verify`
    
    :Call:
        >>> cmd = pyCart.cmd.verify(ftri='Components.i.tri')
    :Inputs:
        *ftri*: :class:`str`
            Name of *tri* file to test
    :Outputs:
        *cmd*: :class:`list` (:class:`str`)
            Command split into a list of strings
    :Versions:
        * 2015-02-13 ``@ddalle``: First version
    """
    # Build the command.
    cmd = ['verify', ftri]
    # Output
    return cmd
    
# Function to call intersect
def intersect(fin='Components.tri', fout='Components.i.tri'):
    """Interface to Cart3D binary `intersect`
    
    :Call:
        >>> cmd = cmd.intersect(fin='Components.tri', fout='Components.i.tri')
    :Inputs:
        *fin*: :class:`str`
            Name of input *tri* file
        *fout*: :class:`str`
            Name of output *tri* file (intersected)
    :Outputs:
        *cmd*: :class:`list` (:class:`str`)
            Command split into a list of strings
    :Versions:
        * 2015-02-13 ``@ddalle``: First version
    """
    # Build the command
    cmd = ['intersect', '-i', fin, '-o', fout]
    # Output
    return cmd

# Function to create flowCart command
def flowCart(cart3d=None, fc=None, i=0, **kwargs):
    """Interface to Cart3D binary ``flowCart``
    
    :Call:
        >>> cmd = pyCart.cmd.flowCart(cart3d, i=0)
        >>> cmd = pyCart.cmd.flowCart(fc=None, i=0)
        >>> cmd = pyCart.cmd.flowCart(**kwargs)
    :Inputs:
        *cart3d*: :class:`pyCart.cart3d.Cart3d`
            Global pyCart settings instance
        *fc*: :class:`pyCart.options.flowCart.flowCart`
            Direct reference to ``cart3d.opts['flowCart']``
        *i*: :class:`int`
            Run index (restart if *i* is greater than *0*)
        *n*: :class:`int`
            Iteration number from which to start (affects ``-N`` setting)
        *mpi_fc*: :class:`bool`
            Whether or not to use MPI version
        *it_fc*: :class:`int`
            Number of iterations to run ``flowCart``
        *mg_fc*: :class:`int`
            Number of multigrid levels to use
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
        *cmd*: :class:`list` (:class:`str`)
            Command split into a list of strings
    :Versions:
        * 2014-09-07 ``@ddalle``: First version
    """
    # Check for cart3d input
    if cart3d is not None:
        # Get values from internal settings.
        mpi_fc  = cart3d.opts.get_mpi_fc(i)
        it_fc   = cart3d.opts.get_it_fc(i)
        limiter = cart3d.opts.get_limiter(i)
        y_span  = cart3d.opts.get_y_is_spanwise(i)
        binIO   = cart3d.opts.get_binaryIO(i)
        tecO    = cart3d.opts.get_tecO(i)
        cfl     = cart3d.opts.get_cfl(i)
        cflmin  = cart3d.opts.get_cflmin(i)
        mg_fc   = cart3d.opts.get_mg_fc(i)
        fmg     = cart3d.opts.get_fmg(i)
        tm      = cart3d.opts.get_tm(i)
        nProc   = cart3d.opts.get_nProc(i)
        mpicmd  = cart3d.opts.get_mpicmd(i)
        buffLim = cart3d.opts.get_buffLim(i)
        td_fc   = cart3d.opts.get_unsteady(i)
        dt      = cart3d.opts.get_dt(i)
        nSteps  = cart3d.opts.get_nSteps(i)
        chkptTD = cart3d.opts.get_checkptTD(i)
        vizTD   = cart3d.opts.get_vizTD(i)
        clean   = cart3d.opts.get_fc_clean(i)
        nstats  = cart3d.opts.get_fc_stats(i)
    elif fc is not None:
        # Get values from direct settings.
        mpi_fc  = fc.get_mpi_fc(i)
        it_fc   = fc.get_it_fc(i)
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
        nSteps  = fc.get_nSteps(i)
        chkptTD = fc.get_checkptTD(i)
        vizTD   = fc.get_vizTD(i)
        clean   = fc.get_fc_clean(i)
        nstats  = fc.get_fc_stats(i)
    else:
        # Get values from keyword arguments
        mpi_fc  = kwargs.get('mpi_fc', False)
        it_fc   = kwargs.get('it_fc', 200)
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
        nSteps  = kwargs.get('nSteps', 100)
        chkptTD = kwargs.get('checkptTD', None)
        vizTD   = kwargs.get('vizTD', None)
        clean   = kwargs.get('clean', False)
        nstats  = kwargs.get('stats', 0)
    # Initialize command.
    if td_fc and mpi_fc:
        # Unsteady MPI flowCart
        cmd = [mpicmd, '-np', str(nProc), 'td_mpix_flowCart', 
            '-his', '-clic', '-unsteady']
    elif td_fc:
        # Unsteady but not MPI
        cmd = ['td_flowCart', '-his', '-clic', '-unsteady']
    elif mpi_fc:
        # Use mpi_flowCart but not unsteady
        cmd = [mpicmd, '-np', str(nProc), 'td_mpix_flowCart', '-his', '-clic']
    else:
        # Use single-node flowCart.
        cmd = ['flowCart', '-his', '-clic']
    # Check for restart
    if (i>0): cmd += ['-restart']
    # Number of steps
    if td_fc:
        # Increase iteration ocunt.
        nSteps += kwargs.get('n', 0)
        # Number of steps and time step
        if nSteps: cmd += ['-nSteps', str(nSteps)]
        if dt:     cmd += ['-dt', str(dt)]
        # Other unsteady options.
        if chkptTD: cmd += ['-checkptTD', str(chkptTD)]
        if vizTD:   cmd += ['-vizTD', str(vizTD)]
        if clean:   cmd += ['-clean']
        if nstats:  cmd += ['-stats', str(nstats)]
    else:
        # Increase the iteration count.
        it_fc += kwargs.get('n', 0)
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


