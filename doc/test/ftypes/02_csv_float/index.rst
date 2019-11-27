
.. This documentation written by TestDriver()
   on 2019-11-27 at 11:11 PST

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
        $ python2 test03_simple.py
        $ python3 test03_simple.py

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
        

**Included file:** ``test03_simple.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import CSV module
        import cape.attdb.ftypes.csv as csv
        
        # Read CSV file
        db = csv.CSVSimple("aeroenv.csv")
        
        # Case number
        i = 13
        
        # Get attributes
        mach = db["mach"][i]
        alph = db["alpha"][i]
        beta = db["beta"][i]
        
        # Create a string
        print("m%.2fa%.2fb%.2f" % (mach, alph, beta))
        

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
    * Command took 0.476455 seconds
    * Cumulative time: 0.476455 seconds
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
    * Command took 0.702189 seconds
    * Cumulative time: 1.17864 seconds
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
    * Command took 0.47751 seconds
    * Cumulative time: 1.65615 seconds
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
    * Command took 0.71926 seconds
    * Cumulative time: 2.37541 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        mach : float32
        alpha: float32
        beta : float16
        

:STDERR:
    * **PASS**

Command 5: Simple CSV read: Python 2
-------------------------------------

:Command:
    .. code-block:: console

        $ python2 test03_simple.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.478939 seconds
    * Cumulative time: 2.85435 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        m1.20a0.00b4.00
        

:STDERR:
    * **PASS**

Command 6: Simple CSV read: Python 3
-------------------------------------

:Command:
    .. code-block:: console

        $ python3 test03_simple.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.717231 seconds
    * Cumulative time: 3.57158 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        m1.20a0.00b4.00
        

:STDERR:
    * **PASS**
