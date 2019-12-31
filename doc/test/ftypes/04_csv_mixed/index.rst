
.. This documentation written by TestDriver()
   on 2019-12-31 at 10:20 PST

Test ``04_csv_mixed``
=======================

This test is run in the folder:

    ``/u/wk/ddalle/usr/pycart/test/ftypes/04_csv_mixed/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_minimal.py
        $ python3 test01_minimal.py
        $ python2 test02_dtypes.py
        $ python3 test02_dtypes.py

**Included file:** ``test01_minimal.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import CSV module
        import cape.attdb.ftypes.csv as csv
        
        # Read CSV file
        db = csv.CSVFile("runmatrix.csv")
        
        # Case number
        i = 6
        
        # Get attributes
        for col in db.cols:
            print("%8s: %s" % (col, db[col][i]))
        

**Included file:** ``test02_dtypes.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import CSV module
        import cape.attdb.ftypes.csv as csv
        
        # Read CSV file
        db = csv.CSVFile("runmatrix.csv",
            DefaultType="float32",
            Types={
                "config": "str",
                "mach": "float16"
            })
        
        # Print data types
        for col in db.cols:
            # Get array
            V = db[col]
            # Check type
            clsname = V.__class__.__name__
            # Data type
            dtype = V[0].__class__.__name__
            # Status message
            print("%8s: %s (%s)" % (col, dtype, clsname))
        

Command 1: Minimal Definitions: Python 2
-----------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_minimal.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.393979 seconds
    * Cumulative time: 0.393979 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

            mach: 2.1
           alpha: 4.0
            beta: 1.5
          config: poweroff
           Label: 
            user: @user3
        

:STDERR:
    * **PASS**

Command 2: Minimal Definitions: Python 3
-----------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_minimal.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.727321 seconds
    * Cumulative time: 1.1213 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

            mach: 2.1
           alpha: 4.0
            beta: 1.5
          config: poweroff
           Label: 
            user: @user3
        

:STDERR:
    * **PASS**

Command 3: Specified dtypes: Python 2
--------------------------------------

:Command:
    .. code-block:: console

        $ python2 test02_dtypes.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.514785 seconds
    * Cumulative time: 1.63609 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

            mach: float16 (ndarray)
           alpha: float32 (ndarray)
            beta: float32 (ndarray)
          config: str (list)
           Label: str (list)
            user: str (list)
        

:STDERR:
    * **PASS**

Command 4: Specified dtypes: Python 3
--------------------------------------

:Command:
    .. code-block:: console

        $ python3 test02_dtypes.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.755956 seconds
    * Cumulative time: 2.39204 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

            mach: float16 (ndarray)
           alpha: float32 (ndarray)
            beta: float32 (ndarray)
          config: str (list)
           Label: str (list)
            user: str (list)
        

:STDERR:
    * **PASS**

