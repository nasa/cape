
.. This documentation written by TestDriver()
   on 2022-04-19 at 01:45 PDT

Test ``09_mat``: PASS
=======================

This test PASSED on 2022-04-19 at 01:45 PDT

This test is run in the folder:

    ``test/ftypes/09_mat/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_clean.py
        $ python3 test01_clean.py
        $ python2 test02_dtypes.py
        $ python3 test02_dtypes.py

**Included file:** ``test01_clean.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import MAT module
        import cape.attdb.ftypes.matfile as matfile
        
        # Read MAT file
        db = matfile.MATFile("wt-sample.mat")
        
        # Case number
        i = 3
        
        # Get attributes
        run = db["run"][i]
        pt = db["pt"][i]
        mach = db["mach"][i]
        alph = db["alpha"][i]
        beta = db["beta"][i]
        
        # Create a string
        print("run%03i.%02i_m%.3fa%.1fb%.1f" % (run, pt, mach, alph, beta))
        

**Included file:** ``test02_dtypes.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import MAT module
        import cape.attdb.ftypes.matfile as matfile
        
        # Read MAT file
        db = matfile.MATFile("wt-sample.mat", DefaultType="int32")
        
        # Print data types
        for col in db.cols:
            print("%-5s: %s" % (col, db[col].dtype.name))
        

Command 1: Clean MAT read: Python 2 (PASS)
-------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_clean.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.40 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 2: Clean MAT read: Python 3 (PASS)
-------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_clean.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.51 seconds
    * Cumulative time: 0.92 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 3: MAT dtype check: Python 2 (PASS)
--------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test02_dtypes.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.39 seconds
    * Cumulative time: 1.31 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 4: MAT dtype check: Python 3 (PASS)
--------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test02_dtypes.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.50 seconds
    * Cumulative time: 1.81 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

