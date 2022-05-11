
.. This documentation written by TestDriver()
   on 2022-05-11 at 01:40 PDT

Test ``07_case``: **FAIL** (command 1)
========================================

This test **FAILED** (command 1) on 2022-05-11 at 01:40 PDT

This test is run in the folder:

    ``test/cape/07_case/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_conds.py
        $ python3 test01_conds.py
        $ python2 test02_case.py
        $ python3 test02_case.py
        $ python2 test03_timing.py
        $ python3 test03_timing.py

**Included file:** ``conditions.json``

    .. code-block:: python

        {
            "mach": 0.4,
            "beta": -0.2,
            "alpha": 4.0
        }

**Included file:** ``case.json``

    .. code-block:: python

        {
            "PhaseSequence": [0, 1, 2],
            "PhaseIters": [500, 750, 1000],
            "nProc": 4,
            "MPI": false,
            "qsub": false
        }

**Included file:** ``test01_conds.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import cape module
        import cape.cfdx.case
        
        # Read conditions
        x = cape.cfdx.case.ReadConditions()
        
        # Show conditions
        print(x["mach"])
        print(x["alpha"])
        
        # Read conditions directly
        beta = cape.cfdx.case.ReadConditions('beta')
        # Display
        print(beta)

**Included file:** ``test02_case.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import cape module
        import cape.cfdx.case
        
        # Read conditions
        rc = cape.cfdx.case.ReadCaseJSON()
        
        # Show settings
        print(rc.get_PhaseSequence())
        print(rc.get_PhaseIters(1))
        print(rc.get_PhaseIters(2))
        print(rc.get_PhaseIters(3))
        print(rc.get_qsub())

**Included file:** ``test03_timing.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Standard library modules
        import datetime
        
        # Import cape module
        import cape.cfdx.case as case
        
        # Example file names
        fstrt = "cape_start.dat"
        ftime = "cape_time.dat"
        
        # Create initial time
        tic = datetime.datetime.now()
        
        # Read settings
        rc = case.ReadCaseJSON()
        
        # Write a flag for starting a program
        case.WriteStartTimeProg(tic, rc, 0, fstrt, "prog")
        
        # Read it
        nProc, t0 = case.ReadStartTimeProg(fstrt)
        
        # Calculate delta time
        dt = tic - t0
        
        # Test output
        print("%i cores, %.4f seconds" % (nProc, dt.seconds))
        
        # Write output file
        case.WriteUserTimeProg(tic, rc, 0, ftime, "cape")
        

Command 1: Conditions: Python 2 (**FAIL**)
-------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_conds.py

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.14 seconds
:STDOUT:
    * **FAIL**
    * Actual: (empty)
    * Target:

      .. code-block:: none

        0.4
        4.0
        -0.2
        

:STDERR:
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "test01_conds.py", line 5, in <module>
            import cape.cfdx.case
          File "/u/wk/ddalle/usr/cape/cape/__init__.py", line 87
        SyntaxError: Non-ASCII character '\xc2' in file /u/wk/ddalle/usr/cape/cape/__init__.py on line 88, but no encoding declared; see http://www.python.org/peps/pep-0263.html for details
        


