
.. _test-cape-case:

Testing :mod:`cape.case`: Individual-Case Functions
=====================================================

This section tests generic calls to case utility functions.  That is, these are
functions that operate within individual case folders.  The are often called
while running a case rather than the master command-line call that sets up and
starts or submits cases.

The :mod:`case` module is one of the more heavily-customized modules for each
individual solver as it has responsibility for counting iterations, calling the
appropriate executables, and more.

.. testsetup:: *

    # System modules
    import os
    
    # Modules to import
    import cape.test
    import cape.case
    
.. _test-cape-case-conditions:

Conditions File
----------------
This function reads the ``conditions.json`` file, which is a simple JSON file
of all the values of CAPE run matrix (trajectory) keys for this particular
case.

.. testcode::

    # Case test folder
    fcape = os.path.join(cape.test.ftcape, "case")
    # Go to this folder.
    os.chdir(fcape)
    
    # Read conditions
    x = cape.case.ReadConditions()
    
    # Show conditions
    print(x["mach"])
    print(x["alpha"])
    
    # Read conditions directly
    beta = cape.case.ReadConditions('beta')
    # Display
    print(beta)
    
.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    0.4
    4.0
    -0.2
    
.. _test-cape-case-case:

CAPE Case Settings File
-------------------------
A subset of the global settings are copied into each case folder in the file
``case.json``.  This allows customizations of settings for individual cases.
For example, it allows neighboring cases to use a different number of
processors or (very importantly) the number of iterations each case must run
for.

.. testcode::

    # Case test folder
    fcape = os.path.join(cape.test.ftcape, "case")
    # Go to this folder.
    os.chdir(fcape)
    
    # Read conditions
    rc = cape.case.ReadCaseJSON()
    
    # Show settings
    print(rc.get_PhaseSequence())
    print(rc.get_PhaseIters(2))
    print(rc.get_qsub())
    
.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    [0, 1, 2, 3]
    1000
    False

.. _test-cape-case-time:

Timing Settings
----------------
Three functions in this module are responsible to reading and writing and CPU
time used by various programs.  Solver-specific customizations of these files
have fixed values for some arguments.

.. testcode::

    # Case test folder
    fcape = os.path.join(cape.test.ftcape, "case")
    # Go to this folder.
    os.chdir(fcape)
    
    # Example file names
    fstrt = "cape_start.dat"
    ftime = "cape_time.dat"
    
    # Delete test files if present
    for fn in [fstrt, ftime]:
        if os.path.isfile(fn):
            os.remove(fn)
            
    # Create initial time
    tic = cape.case.datetime.now()
    
    # Read settings
    rc = cape.case.ReadCaseJSON()
    
    # Write a flag for starting a program
    cape.case.WriteStartTimeProg(tic, rc, 0, fstrt, "prog")
    
    # Read it
    nProc, t0 = cape.case.ReadStartTimeProg(fstrt)
    
    # Calculate delta time
    dt = tic - t0
    
    # Test output
    print(nProc - rc.get_nProc())
    print(dt.seconds > 1)
    
    # Write output file
    cape.case.WriteUserTimeProg(tic, rc, 0, ftime, "cape")
    
    
.. testoutput::
    :options: +NORMALIZE_WHITESPACE
    
    0
    False
    

    
