

----------------------------
Run Options for `flowCart`
----------------------------

The "flowCart" section of :file:`pyCart.json` controls settings for executing
the flow solver.  That includes settings specific to `flowCart` and
`mpix_flowCart` in additions to master settings needed for running the flow
solver in any mode.

Most important, there are options that specify command-line flags to
`flowCart` that specify some of the settings (or override values specified in
input files), options that determine how many iterations to run, options which
change settings in the input files, and options that specify what mode to
run the code in.

These settings are saved in a file called :file:`conditions.json` in each run
folder.  For most of these options, changing the value in
:file:`conditions.json` will affect the way that `flowCart` is run whenever the
next segment of the case is started. The full list of options with default
values is shown below.

    .. code-block:: javascript
    
        "flowCart": {
            // Run sequence
            "InputSeq": [0],
            "IterSeq": [200],
            "it_fc": 200
            // Command-line options
            "cfl": 1.1,
            "cflmin": 0.8,
            "mg_fc": 3,
            "dt": 0.1,
            "limiter": 2,
            "y_is_spanwise": true,
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


Run Sequence Options
====================

Three options in the "flowCart" section determine how many iterations to run
and which input set to use.  These options and their possible values are listed
below.

    *InputSeq*: {``[0]``} | ``[0, 1]`` | :class:`list` (:class:`int`)
        List of input sequences to use and order in which to use them.  These
        do not have to be in numerical order.  For example, ``[0, 1, 3]`` is an
        allowable sequence that may become useful if it's later determined that
        run number 2 is unnecessary or unproductive.
        
    *IterSeq*: {``[200]``} | ``[0, 0, 1000]`` | :class:`list` (:class:`int`)
        List of minimum number of total iterations after each run.  This list
        must have the same length as *InputSeq*, and a ``0`` tells pyCart to
        continue to the next run regardless of the current iteration count.
        
    *it_fc*: {``200``} | :class:`int` | :class:`list` (:class:`int`)
        Number of iterations to use in the next run of `flowCart`.  If this
        is a list, the number of iterations may vary from run to run, whereas a
        single integer value means that all runs will use the same number of
        iterations.
        
The following pseudocode describes approximately how pyCart uses these options
to determine which run number to use.

    .. code-block:: python
        
        import pyCart.case
        
        # Read the current options.
        fc = pyCart.case.ReadCaseJSON()
        # Determine the minimum total number of iterations
        nIter = fc.get_LastIter()
        # Get current number of iterations
        nCur = pyCart.case.GetCurrentIter()
        # Start with run index 0.
        j = 0
        
        # Loop until iteration count has been reached.
        while (nCur < nIter):
            # Get the run number.
            i = fc.get_InputSeq(j)
            # Number of iterations to use.
            it_fc = fc.get_it_fc(i)
            # Run flowCart for it_fc iterations...
            
            # Determine the current number of iterations.
            nCur = pyCart.case.GetCurrentiter()
            # Check it against current target number of iterations.
            if nCur >= fc.get_InputSeq(i):
                # If so, move to the next run in the sequence.
                i += 1
                # (If not, continue on current run number.)
            
        
`Example:`
Suppose the following values are part of :file:`pyCart.json`.

    .. code-block:: javascript
    
        "flowCart": {
            "InputSeq": [0, 1, 3],
            "IterSeq": [0, 1000, 1500],
            "it_fc": [500, 250],
            ...
        }

This would be interpreted in the following way:

    #. run input ``0`` for 500 iterations (i.e. ``it_fc[0]``) at a time once;
    #. run input ``1`` for 250 iterations (i.e. ``it_fc[1]``) at a time until
       at least 1000 total iterations (i.e. ``IterSeq[1]``) have been run; and
    #. run input ``3`` for 250 iterations (in this case, since there is no
       ``it_fc[3]``, use ``it_fc[-1]``) until 1500 total iterations have been
       run.
    
Thus there are 5 calls to `flowCart`, and the description of each run and the
cumulative progress is as shown in the table below.

    ======   =========   ====================   ===============
    Number   Run Index   Number of Iterations   Iteration Total
    ======   =========   ====================   ===============
      0          0               500                   500
      1          1               250                   750
      2          1               250                  1000
      3          2               250                  1250
      4          2               250                  1500
    ======   =========   ====================   ===============
    
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
        
    *nSteps*: {``10``} | :class:`int` | :class:`list` (:class:`int`)
        Number of subiterations if running in unsteady mode (otherwise ignored)
        
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
        

Run Method Options
==================

The remaining options determine basic aspects of how `flowCart` is called.  For
example, is the case submitted as a PBS job?  Is it run in MPI mode?  Is it run
unsteady (time-accurate)?  Is it an adaptive run?  The full list of options is
below

    *unsteady*: ``true`` | {:class:`false`} | :class;`list` (:class:`bool`)
        Whether or not to run unsteady simulation by adding the ``-steady``
        flag and switching the binary to either `td_flowCart` or
        `mpx_td_flowCart`
        
    *mpi_fc*: ``true`` | {``false``} | :class;`list` (:class:`bool`)
        Whether or not to use MPI version of the code.  Switches binary to
        `mpi_flowCart` or `mpix_td_flowCart`

    *qsub*: {``true``} | ``false`` | :class:`list` (:class:`bool`)
        Whether or not to submit cases as PBS jobs (as opposed to running them
        locally)
        
    *resub*: ``true`` | {``false``} | :class;`list` (:class:`bool`)
        Whether or not to resubmit a new job at the end of a run or simply
        continue on with the current Job ID
        
    *use-areo-shell*: ``true`` | {``false``} | :class;`list` (:class:`bool`)
        Whether or not the current index of the run is adaptive
        
    *jmpstart*: `true`` | {``false``} | :class;`list` (:class:`bool`)
        Whether or not to jumpstart adaptive cases with an existing volume
        meshes.
