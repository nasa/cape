
.. This documentation written by TestDriver()
   on 2019-11-25 at 13:10 PST

Test ``02_csv_float``
=======================

This test is run in the folder:

    ``/u/wk/ddalle/usr/pycart/test/ftypes/02_csv_float/``

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
        db = csv.CSVFile("aeroenv.csv")
        
        # Case number
        i = 13
        
        # Get attributes
        mach = db["mach"][i]
        alph = db["alpha"][i]
        beta = db["beta"][i]
        
        # Create a string
        print("m%.2fa%.2fb%.2f" % (mach, alph, beta))
        

**Included file:** ``test02_dtypes.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import CSV module
        import cape.attdb.ftypes.csv as csv
        
        # Read CSV file
        db = csv.CSVFile("aeroenv.csv",
            DefaultType="float32",
            Definitions={"beta": {"Type": "float16"}})
        
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
    * Command took 0.354044 seconds
    * Cumulative time: 0.354044 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        m1.20a0.00b4.00
        

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
    * Command took 0.670097 seconds
    * Cumulative time: 1.02414 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        m1.20a0.00b4.00
        

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
    * Command took 0.372658 seconds
    * Cumulative time: 1.3968 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        mach : float32
        alpha: float32
        beta : float16
        

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
    * Command took 0.66423 seconds
    * Cumulative time: 2.06103 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        mach : float32
        alpha: float32
        beta : float16
        

:STDERR:
    * **PASS**

