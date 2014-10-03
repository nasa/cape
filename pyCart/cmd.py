"""
Create command strings for Cart3D binaries: :mod:`pyCart.cmd`
=============================================================

This module contains direct interfaces to Cart3D binaries so that they can be
called from Python.  Settings for the binaries that are usually specified as
command-line options are either specified as keyword arguments or inherited from
a :class:`pyCart.cart3d.Cart3d` instance.
"""


# Function to call cubes.
def cubes(cart3d=None, maxR=10, reorder=True, pre='preSpec.c3d.cntl',
    cubes_a=None, cubes_b=None):
    """
    Interface to Cart3D script `cubes`
    
    :Call:
        >>> cmd = pyCart.cmd.cubes(cart3d)
        >>> cmd = pyCart.cmd.cubes(maxR=10, reorder=True, pre='preSpec.c3d.cntl')
    :Inputs:
        *cart3d*: :class:`pyCart.cart3d.Cart3d`
            Global pyCart settings instance
        *maxR*: :class:`int`
            Number of refinements to make
        *reorder*: :class:`bool`
            Whether or not to reorder mesh
        *pre*: :class:`str`
            Name of prespecified bounding box file (or ``None``)
        *cubes_a*: :class:`int`
            `cubes` "a" parameter
        *cubes_b*: :class:`int`
            `cubes` "b" parameter
    :Outputs:
        *cmd*: :class:`list` (:class:`str`)
            Command split into a list of strings
    :Versions:
        * 2014.06.30 ``@ddalle``: First version
        * 2014.09.02 ``@ddalle``: Rewritten with new options paradigm
        * 2014.09.10 ``@ddalle``: Split into cmd and bin
    """
    # Check cart3d input.
    if cart3d is not None:
        # Apply values
        maxR    = cart3d.opts.get_maxR()
        pre     = cart3d.opts.get_pre()
        reorder = cart3d.opts.get_reorder()
        cubes_a = cart3d.opts.get_cubes_a()
        cubes_b = cart3d.opts.get_cubes_b()
    # Initialize command
    cmd = ['cubes']
    # Add options.
    if maxR:    cmd += ['-maxR', str(maxR)]
    if reorder: cmd += ['-reorder']
    if pre:     cmd += ['-pre', pre]
    if cubes_a: cmd += ['-a', str(cubes_a)]
    if cubes_b: cmd += ['-b', str(cubes_b)]
    # Return the command.
    return cmd
    
# Function to call mgPrep
def mgPrep(cart3d=None, mg_fc=3):
    """
    Interface to Cart3D script `mgPrep`
    
    :Call:
        >>> cmd = pyCart.cmd.mgPrep(cart3d)
        >>> cmd = pyCart.cmd.mgPrep(mg_fc=3)
    :Inputs:
        *cart3d*: :class:`pyCart.cart3d.Cart3d`
            Global pyCart settings instance
        *mg_fc*: :class:`int`
            Number of multigrid levels to prepare
    :Outputs:
        *cmd*: :class:`list` (:class:`str`)
            Command split into a list of strings
    :Versions:
        * 2014.09.02 ``@ddalle``: First version
    """
    # Check cart3d input.
    if cart3d is not None:
        # Apply values
        mg_fc = cart3d.opts.get_mg_fc()
    # Initialize command.
    cmd = ['mgPrep']
    # Add options.
    if mg_fc: cmd += ['-n', str(mg_fc)]
    # Return the command
    return cmd
    
# Function to call mgPrep
def autoInputs(cart3d=None, r=8, ftri='Components.i.tri'):
    """
    Interface to Cart3D script `mgPrep`
    
    :Call:
        >>> cmd = pyCart.cmd.autoInputs(cart3d)
        >>> cmd = pyCart.cmd.autoInputs(r=8, ftri='components.i.tri')
    :Inputs:
        *cart3d*: :class:`pyCart.cart3d.Cart3d`
            Global pyCart settings instance
        *r*: :class:`int`
            Mesh radius
        *ftri*: :class:`str`
            Name of surface triangulation file
    :Outputs:
        *cmd*: :class:`list` (:class:`str`)
            Command split into a list of strings
    :Versions:
        * 2014.09.02 ``@ddalle``: First version
    """
    # Check cart3d input.
    if cart3d is not None:
        # Apply values
        mg_fc = cart3d.opts.get_r()
        ftri  = 'Components.i.tri'
    # Initialize command.
    cmd = ['autoInputs']
    # Add options.
    if r:    cmd += ['-r', str(mg_fc)]
    if ftri: cmd += ['-t', ftri]
    # Return the command.
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
        *cfl*: :class:`float`
            Nominal CFL number
        *cflmin*: :class:`float`
            Minimum CFL number allowed for problem cells or iterations
        *tm*: :class:`bool`
            Whether or not to use second-order cut cells
        *binaryIO*: :class:`bool`
            Whether or not to use binary format for input/output files
        *limiter*: :class:`int`
            Limiter index
        *y_is_spanwise*: :class:`bool`
            Whether or not to consider angle of attack in *z* direction
        *nProc*: :class:`int`
            Number of threads to use for ``flowCart`` operation
    :Outputs:
        *cmd*: :class:`list` (:class:`str`)
            Command split into a list of strings
    :Versions:
        * 2014.09.07 ``@ddalle``: First version
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
        tm      = cart3d.opts.get_tm(i)
        nProc   = cart3d.opts.get_nProc(i)
        mpicmd  = cart3d.opts.get_mpicmd(i)
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
        tm      = fc.get_tm(i)
        nProc   = fc.get_nProc(i)
        mpicmd  = fc.get_mpicmd(i)
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
        tm      = kwargs.get('tm', None)
        nProc   = kwargs.get('nProc', 8)
        mpicmd  = kwargs.get('mpicmd', 'mpiexec')
    # Increase the iteration count.
    it_fc += kwargs.get('n', 0)
    # Initialize command.
    if mpi_fc:
        # Use mpi_flowCart
        cmd = [mpicmd, '-np', nProc, 'mpi_flowCart', '-his', '-v']
    else:
        # Use single-node flowCart.
        cmd = ['flowCart', '-his', '-v']
    # Check for restart
    if (i>0): cmd += ['-restart']
    # Add options
    if it_fc:   cmd += ['-N', str(it_fc)]
    if y_span:  cmd += ['-y_is_spanwise']
    if limiter: cmd += ['-limiter', str(limiter)]
    if binIO:   cmd += ['-binaryIO']
    if tecO:    cmd += ['-T']
    if cfl:     cmd += ['-cfl', str(cfl)]
    if mg_fc:   cmd += ['-mg', str(mg_fc)]
    # Process and translate the cut-cell gradient variable.
    if (tm is not None) and tm:
        # Second-order cut cells
        cmd += ['-tm', '1']
    elif (tm is not None) and (not tm):
        # First-order cut cells
        cmd += ['-tm', '0']
    # Output
    return cmd


