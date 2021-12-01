
.. This documentation written by TestDriver()
   on 2021-12-01 at 01:45 PST

Test ``04_csv_mixed``: PASS
=============================

This test PASSED on 2021-12-01 at 01:45 PST

This test is run in the folder:

    ``test/ftypes/04_csv_mixed/``

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
        import cape.attdb.ftypes.csvfile as csvfile
        
        # Read CSV file
        db = csvfile.CSVFile("runmatrix.csv")
        
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
        import cape.attdb.ftypes.csvfile as csvfile
        
        # Read CSV file
        db = csvfile.CSVFile("runmatrix.csv",
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
        

Command 1: Minimal Definitions: Python 2 (PASS)
------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_minimal.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.54 seconds
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

Command 2: Minimal Definitions: Python 3 (PASS)
------------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_minimal.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.78 seconds
    * Cumulative time: 1.33 seconds
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

Command 3: Specified dtypes: Python 2 (PASS)
---------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test02_dtypes.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.44 seconds
    * Cumulative time: 1.76 seconds
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

Command 4: Specified dtypes: Python 3 (PASS)
---------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test02_dtypes.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.51 seconds
    * Cumulative time: 2.28 seconds
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

