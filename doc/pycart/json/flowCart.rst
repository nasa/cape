
.. _pycart-json-flowCart:

----------------------------
Run Options for `flowCart`
----------------------------

The ``"flowCart"`` section of :file:`pyCart.json` controls settings for
executing the flow solver. That includes settings specific to ``flowCart`` and
``mpix_flowCart``. These settings are described in a dictionary that is a
subset of the ``"RunControl"`` section, which is written to each folder as
``case.json``.

Most important, there are options that specify command-line flags to
`flowCart` that specify some of the settings (or override values specified in
input files), options that determine how many iterations to run, options which
change settings in the input files, and options that specify what mode to
run the code in.

These settings are described in a dictionary within the ``"RunControl"``
:ref:`section <pycart-json-RunControl>`.

    .. code-block:: javascript
    
        "flowCart": {
            // Iteration control
            "it_fc": 200,
            "it_avg": 10,
            "it_start": 100,
            "it_sub": 10,
            // Command-line options
            "cfl": 1.1,
            "cflmin": 0.8,
            "mg_fc": 3,
            "dt": 0.1,
            "unsteady": true,
            "limiter": 2,
            "y_is_spanwise": true,
            "clic": True,
            "nSteps": 10,
            "checkptTD": false,
            "vizTD": false,
            "fc_clean": false,
            "fc_stats": 0,
            "binaryIO": true,
            "tecO": true,
            "fmg": true,
            "pmg": false,
            "tm": false,
            "buffLim": false,
            // Input file options
            "first_order": false,
            "robust_mode": false,
            "RKScheme": "",
            "nOrders": 12,
            // Run method options
            "nProc": 8,
            "unsteady": false,
            "mpi_fc": false,
            "mpicmd": "mpiexec",
            "qsub": true,
            "resub": false,
            "use_aero_csh": false,
            "jumpstart": false
        }


Iteration Options
====================

Three options in the *flowCart* section determine how many iterations to run
and which input set to use.  These options and their possible values are listed
below.
        
    *it_fc*: {``200``} | :class:`int` | :class:`list` (:class:`int`)
        Number of iterations to use in the next run of `flowCart`.  If this
        is a list, the number of iterations may vary from run to run, whereas a
        single integer value means that all runs will use the same number of
        iterations.
        
    *it_avg*: {``10``} | :class:`int` | :class:`list` (:class:`int`)
        Number of iterations between writing an output triq file for averaging.
        This is used to manually write surface solutions for averaging
        operations that may be needed for recording surface pressures.  Cleanup
        of intermediate files is automatically kept, and averaging is performed
        cumulatively
        
    *it_start*: {``100``} | :class:`int` | :class:`list` (:class:`int`)
        Do not perform cumulative averaging until this iteration is reached
        
    *it_sub*: {``10``} | :class:`int` | :class:`list` (:class:`int`)
        For unsteady runs, the number of subiterations for each time step
    
Command-Line Options
====================

The following dictionary of options get translated into command-line flags
specified to `flowCart`.  For example, ``flowCart -N 200 -cfl 1.1``.

    *cfl*: {``1.1``} | :class:`float` | :class:`list` (:class:`float`)
        Nominal CFL number to use for `flowCart`
        
    *cflmin*: {``0.8``} | :class:`float` | :class:`list` (:class:`float`)
        Fallback CFL number if nominal input fails
        
    *mg_fc*: {``3``} | :class:`int` | :class:`list` (:class:`int`)
        Number of multigrid cycles to use while running `flowCart`
        
    *dt*: {``0.1``} | :class:`float` | :class:`list` (:class:`float`)
        Physical time step if running in unsteady mode (otherwise ignored)
        
    *limiter*: {``2``} | ``5`` | :class:`list` (:class:`int`)
        Number of Cart3D limiter option to use.
        
    *y_is_spanwise*: {``true``} | ``false`` | :class:`list` (:class:`bool`)
        Whether or not the y-axis is the spanwise axis (otherwise the z-axis is
        the spanwise axis)
        
    *checkptTD*: {``false``} | :class:`int` | :class:`list` (:class:`int`)
        Number of unsteady iterations between saving checkpoint files.  Ignored
        if not an unsteady run, and no extra checkpoint files are created if
        this parameter is set to ``0`` or ``false``.
        
    *vizTD*: {``false``} | :class:`int` | :class:`list` (:class:`int`)
        Number of unsteady iterations between saving visualization files, i.e.
        cut plane files and surface files. Ignored if not an unsteady run, and
        no extra visualization files are created if this parameter is set to
        ``0`` or ``false``.
        
    *fc_clean*: {``false``} | ``true`` | :class:`list` (:class:`bool`)
        Whether or not to run an extra unsteady cycle using the ``-clean`` flag
        before starting unsteady iterations
        
    *fc_stats*: {``false``} | :class:`int` | :class:`list` (:class:`int`)
        Number of iterations to use for creating iterative average at the end
        of a run
        
    *binaryIO*: {``true``} | ``false`` | :class:`list` (:class:`bool`)
        Whether or not to write binary TecPlot output files (text files are
        created if this is set to ``false``)
        
    *tecO*: {``true``} | ``false`` | :class:`list` (:class:`bool`)
        Whether or not to write TecPlot surface output files (sets the
        command-line ``-T`` flag to `flowCart`
        
    *fmg*: {``true``} | ``false`` | :class:`list` (:class:`bool`)
        Whether or not to use full multigrid.  Unset this flag to increase
        robustness slightly.  Setting this parameter to ``false`` adds the
        ``-no_fmg`` flag to `flowCart`
        
    *pmg*: ``true`` | {``false``} | :class:`list` (:class:`bool`)
        Flag for poly multigrid.  If this is ``true``, the ``-pmg`` flag is
        added to `flowCart`
        
    *tm*: ``true`` | {``false``} | :class:`list` (:class:`bool`)
        Option that if ``true`` sets the ``-tm 0`` flag, which makes all cut
        cells first-order in `flowCart`
        
    *buffLim*: ``true`` | {``false``} | :class:`list` (:class:`bool`)
        Flag to set buffer limiter, which can be used to increase stability
        

Input File Options
==================

The following options set parameters in :file:`input.cntl`.  Because these
files are written when the case is set up (versus when the cases are run), the
input files already exist at the time that `flowCart` is called, and thus
changing these settings after the case's folder has been created has no effect.
Note, however, that manually changing the input files will of course have an
effect.

    *first_order*: ``true`` | {``false``} | :class:`list` (:class:`bool`)
        Whether or not to run the case in first-order mode.  This has the
        effect of turning off all gradient evaluations to the Runge-Kutta
        scheme.
        
    *robust_mode*: ``true`` | {``false``} | :class:`list` (:class:`bool`)
        Whether or not to run the case in robust mode
        
    *RKScheme*: {``""``} | ``"VL5"`` | ``"first-order"`` | ``"robust"`` |
            ``"VL3-1"`` | ``"VL3-2"`` | ``"VL4"`` |
            :class:`list` ([:class:`float`, ``0 | 1``])
            
        Manually specified Nx2 matrix of Runge-Kutta coefficients and either
        ``1`` to mark a gradient evaluation at a stage or ``0`` to mark no
        gradient evaluation.
        
    *nOrders*: {``12``} | :class:`int` | :class:`list` (:class:`int`)
        Number of orders of magnitude of residual drop at which to terminate
        `flowCart` early
        

