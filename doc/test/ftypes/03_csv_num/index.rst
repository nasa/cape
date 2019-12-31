
.. This documentation written by TestDriver()
   on 2019-12-31 at 08:44 PST

Test ``03_csv_num``
=====================

This test is run in the folder:

    ``/u/wk/ddalle/usr/pycart/test/ftypes/03_csv_num/``

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
        import cape.attdb.ftypes.csv as csv
        
        # Read CSV file
        db = csv.CSVFile("wt-sample.csv")
        
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
        import cape.attdb.ftypes.csv as csv
        
        # Read CSV file
        db = csv.CSVFile("wt-sample.csv", DefaultType="int32")
        
        # Print data types
        for col in db.cols:
            print("%-5s: %s" % (col, db[col].dtype.name))
        

Command 1: Clean CSV read: Python 2
------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_clean.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.498928 seconds
    * Cumulative time: 0.498928 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        run257.03_m0.974a2.0b-0.0
        

:STDERR:
    * **PASS**

Command 2: Clean CSV read: Python 3
------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_clean.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.806942 seconds
    * Cumulative time: 1.30587 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        run257.03_m0.974a2.0b-0.0
        

:STDERR:
    * **PASS**

Command 3: Specified :class:`float` types: Python 2
----------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test02_dtypes.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.512848 seconds
    * Cumulative time: 1.81872 seconds
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

Command 4: Specified :class:`float` types: Python 3
----------------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test02_dtypes.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.772589 seconds
    * Cumulative time: 2.59131 seconds
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

