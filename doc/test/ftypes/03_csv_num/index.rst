
.. This documentation written by TestDriver()
   on 2022-04-14 at 01:45 PDT

Test ``03_csv_num``: PASS
===========================

This test PASSED on 2022-04-14 at 01:45 PDT

This test is run in the folder:

    ``test/ftypes/03_csv_num/``

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
        
        # Import CSV module
        import cape.attdb.ftypes.csvfile as csvfile
        
        # Read CSV file
        db = csvfile.CSVFile("wt-sample.csv")
        
        # Case number
        i = 2
        
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
        
        # Import CSV module
        import cape.attdb.ftypes.csvfile as csvfile
        
        # Read CSV file
        db = csvfile.CSVFile("wt-sample.csv", DefaultType="int32")
        
        # Print data types
        for col in db.cols:
            print("%-5s: %s" % (col, db[col].dtype.name))
        

Command 1: Clean CSV read: Python 2 (PASS)
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
    * Command took 0.48 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        run257.03_m0.974a2.0b-0.0
        

:STDERR:
    * **PASS**

Command 2: Clean CSV read: Python 3 (PASS)
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
    * Command took 0.56 seconds
    * Cumulative time: 1.04 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        run257.03_m0.974a2.0b-0.0
        

:STDERR:
    * **PASS**

Command 3: Specified :class:`float` types: Python 2 (PASS)
-----------------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test02_dtypes.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.48 seconds
    * Cumulative time: 1.52 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        run  : int32
        pt   : int32
        mach : float64
        alpha: float64
        beta : float64
        

:STDERR:
    * **PASS**

Command 4: Specified :class:`float` types: Python 3 (PASS)
-----------------------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test02_dtypes.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.56 seconds
    * Cumulative time: 2.08 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        run  : int32
        pt   : int32
        mach : float64
        alpha: float64
        beta : float64
        

:STDERR:
    * **PASS**

