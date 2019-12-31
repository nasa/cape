
.. This documentation written by TestDriver()
   on 2019-12-31 at 09:36 PST

Test ``05_csv_noheader``
==========================

This test is run in the folder:

    ``/u/wk/ddalle/usr/pycart/test/ftypes/05_csv_noheader/``

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
            cols=["mach", "alpha", "beta", "config", "Label", "user"],
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
    * Command took 0.502375 seconds
    * Cumulative time: 0.502375 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

            col1: 2.1
            col2: 4.0
            col3: 1.5
            col4: poweroff
            col5: 
            col6: @user3
        

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
    * Command took 0.795373 seconds
    * Cumulative time: 1.29775 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

            col1: 2.1
            col2: 4.0
            col3: 1.5
            col4: poweroff
            col5: 
            col6: @user3
        

:STDERR:
    * **PASS**

Command 3: Specified Column Titles: Python 2
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
    * Command took 0.511948 seconds
    * Cumulative time: 1.8097 seconds
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

Command 4: Specified Column Titles: Python 3
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
    * Command took 0.769695 seconds
    * Cumulative time: 2.57939 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

