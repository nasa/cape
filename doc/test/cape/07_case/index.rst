
.. This documentation written by TestDriver()
   on 2019-09-18 at 12:34 PDT

Test ``07_case``
==================

This test is run in the folder:

    ``/u/wk/ddalle/usr/pycart/test/cape/07_case/``

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
        import cape.case
        
        # Read conditions
        x = cape.case.ReadConditions()
        
        # Show conditions
        print(x["mach"])
        print(x["alpha"])
        
        # Read conditions directly
        beta = cape.case.ReadConditions('beta')
        # Display
        print(beta)

**Included file:** ``test02_case.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import cape module
        import cape.case
        
        # Read conditions
        rc = cape.case.ReadCaseJSON()
        
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
        import cape.case
        
        # Example file names
        fstrt = "cape_start.dat"
        ftime = "cape_time.dat"
        
        # Create initial time
        tic = datetime.datetime.now()
        
        # Read settings
        rc = cape.case.ReadCaseJSON()
        
        # Write a flag for starting a program
        cape.case.WriteStartTimeProg(tic, rc, 0, fstrt, "prog")
        
        # Read it
        nProc, t0 = cape.case.ReadStartTimeProg(fstrt)
        
        # Calculate delta time
        dt = tic - t0
        
        # Test output
        print("%i cores, %.4f seconds" % (nProc, dt.seconds))
        
        # Write output file
        cape.case.WriteUserTimeProg(tic, rc, 0, ftime, "cape")
        

Command 1: Conditions: Python 2
--------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_conds.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.485867 seconds
    * Cumulative time: 0.485867 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        0.4
        4.0
        -0.2
        

:STDERR:
    * **PASS**

Command 2: Conditions: Python 3
--------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_conds.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.65128 seconds
    * Cumulative time: 1.13715 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        0.4
        4.0
        -0.2
        

:STDERR:
    * **PASS**

Command 3: ``case.json``: Python 2
-----------------------------------

:Command:
    .. code-block:: console

        $ python2 test02_case.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.361541 seconds
    * Cumulative time: 1.49869 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        [0, 1, 2]
        750
        1000
        1000
        False
        

:STDERR:
    * **PASS**

Command 4: ``case.json``: Python 3
-----------------------------------

:Command:
    .. code-block:: console

        $ python3 test02_case.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.574662 seconds
    * Cumulative time: 2.07335 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        [0, 1, 2]
        750
        1000
        1000
        False
        

:STDERR:
    * **PASS**

Command 5: Timing: Python 2
----------------------------

:Command:
    .. code-block:: console

        $ python2 test03_timing.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.379587 seconds
    * Cumulative time: 2.45294 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        4 cores, 0.0000 seconds
        

:STDERR:
    * **PASS**

Command 6: Timing: Python 3
----------------------------

:Command:
    .. code-block:: console

        $ python3 test03_timing.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.624004 seconds
    * Cumulative time: 3.07694 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        4 cores, 0.0000 seconds
        

:STDERR:
    * **PASS**

